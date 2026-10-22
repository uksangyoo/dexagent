import numpy as np

TABLE = 0.75
# Thumb-index pinch target in the palm frame (geometry described in dexagent/hand.py):
# thumb toward the wrist (-z), index toward the fingertips (+z).
C = np.array([0.080, 0.040, 0.118])
SEED_THUMB = np.array([-0.432, -1.59, 1.049, 0.632])
SEED_INDEX = np.array([-0.184, 0.3, 0.768, 0.732])
THUMB_Z = 0.096           # thumb post pad, palm z
PEN_GAP = 0.0125          # thumb pad centre to pen axis (pad inset ~8 mm + pen radius)
GRASP_BELOW_TOP = 0.025


def smooth(u):
    return u * u * (3 - 2 * u)


def r3(v):
    return np.round(np.asarray(v, float) * 1000, 1).tolist()


# ------------------------------------------------------------ measurement-based finger IK (built here)
def measure(api, q, t_move=0.10, t_hold=0.14):
    yield from api.move(q=q, seconds=t_move)
    yield from api.hold(t_hold)
    return api.tips()


def jacobian(api, q, fingers, eps=0.08):
    """Finite-difference pad Jacobians (3x4 per finger), all fingers perturbed together."""
    lim = api.joint_limits
    base = yield from measure(api, q)
    J = {f: np.zeros((3, 4)) for f in fingers}
    for j in range(4):
        qp = q.copy()
        s = {}
        for f in fingers:
            k = 4 * f + j
            s[f] = eps if q[k] + eps <= lim[k, 1] else -eps
            qp[k] = q[k] + s[f]
        t = yield from measure(api, qp)
        for f in fingers:
            J[f][:, j] = (t[f] - base[f]) / s[f]
    return base, J


def solve(api, log, q, targets, iters=8, tol=0.0015, tag=""):
    """Gauss-Newton on measured pads with a Broyden-updated Jacobian."""
    lim = api.joint_limits
    q = np.array(q, float)
    fingers = list(targets)
    base, J = yield from jacobian(api, q, fingers)
    worst = 1.0
    for it in range(iters):
        errs = {f: targets[f] - base[f] for f in fingers}
        worst = max(np.linalg.norm(e) for e in errs.values())
        if worst < tol or it == iters - 1:
            log.note(f"ik{tag} it{it} err_mm={ {f: round(1000 * np.linalg.norm(e), 1) for f, e in errs.items()} }")
            break
        qn = q.copy()
        for f in fingers:
            Jf = J[f]
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-4 * np.eye(3), errs[f])
            n = np.linalg.norm(dq)
            if n > 0.35:
                dq *= 0.35 / n
            sl = slice(4 * f, 4 * f + 4)
            qn[sl] = np.clip(q[sl] + dq, lim[sl, 0], lim[sl, 1])
        new = yield from measure(api, qn)
        for f in fingers:
            sl = slice(4 * f, 4 * f + 4)
            d = qn[sl] - q[sl]
            if d @ d > 1e-8:
                J[f] += np.outer((new[f] - base[f]) - J[f] @ d, d) / (d @ d)
        q, base = qn, new
    return q, worst


# ------------------------------------------------------------ pen geometry
def pen_geom(api, Lh):
    p, R = api.object_pose("pen")
    a = R[:, 2] if R[2, 2] > 0 else -R[:, 2]      # unit axis pointing up
    return p, a, p - a * Lh, p + a * Lh            # centre, up axis, tip, top


class Log:
    def __init__(self, api):
        self.api, self.lines = api, []

    def note(self, text):
        self.lines.append(f"{self.api.t:6.2f} {text}")

    def flush(self):
        for l in self.lines[-55:]:
            self.api.note(l)


def run(api):
    log = Log(api)
    try:
        yield from _run(api, log)
    finally:
        log.flush()


def _run(api, log):
    lim = api.joint_limits
    R = api.hand_down()
    yield from api.hold(0.3)
    p0, Rp0 = api.object_pose("pen")
    lean0 = np.arccos(min(1.0, abs(Rp0[2, 2])))
    Lh = (p0[2] - TABLE) / np.cos(lean0)          # centre to end along the axis
    _, _, tip0, top0 = pen_geom(api, Lh)
    home = tip0[:2].copy()
    log.note(f"pen p={r3(p0)} lean={np.degrees(lean0):.1f}deg Lh={Lh*1000:.1f}mm home={r3(home)}")
    log.note(f"limits thumb={np.round(lim[0:4], 2).tolist()} index={np.round(lim[4:8], 2).tolist()}")

    # ---------------- calibrate the thumb-index pinch by measurement, in the air
    # The thumb is a post near the edge of its reach (pad ~ (80, 40, 96) mm); the pen sits
    # just above it along palm z and the index presses it down onto the thumb.
    q = np.array(api.OPEN, float)
    q[0:4] = SEED_THUMB                               # seeds from iteration 2's measured IK
    q[4:8] = SEED_INDEX
    T_open = {0: np.array([0.080, 0.040, 0.088]), 1: np.array([0.080, 0.040, 0.150])}
    q_open, _ = yield from solve(api, log, q, T_open, iters=6, tag="open")
    q_tc, _ = yield from solve(api, log, q_open, {0: np.array([0.080, 0.040, THUMB_Z])}, iters=6, tag="thumb")
    q_is, _ = yield from solve(api, log, q_open, {1: np.array([0.080, 0.040, 0.110])}, iters=6, tag="index")
    t_op = yield from measure(api, q_open)
    q_pre = q_open.copy(); q_pre[0:4] = q_tc[0:4]
    t_tc = yield from measure(api, q_pre)
    pinch = t_tc[0] + np.array([0, 0, PEN_GAP])       # where the pen axis should sit (palm frame)
    log.note(f"q_open={np.round(q_open[:8], 3).tolist()} thumb_close={np.round(q_tc[:4], 3).tolist()} "
             f"index_sq={np.round(q_is[4:8], 3).tolist()} open thumb={r3(t_op[0])} index={r3(t_op[1])} "
             f"thumb_close={r3(t_tc[0])} pinch={r3(pinch)}")
    dpin = pinch - t_op[0]                            # pinch point relative to the open thumb pad

    def index_closing(u):
        q = q_pre.copy()
        q[4:8] = q_open[4:8] + u * (q_is[4:8] - q_open[4:8])
        return np.clip(q, lim[:, 0], lim[:, 1])

    def align(n=3, zoff=0.0):
        """Put the pinch point on the pen axis, GRASP_BELOW_TOP under its top, using measured pads."""
        for i in range(n):
            tw = api.tips_world()
            P, Rm = api.measured_palm()
            th = Rm.T @ (tw[0] - P)
            Mc = tw[0] + Rm @ dpin
            p, a, tip, top = pen_geom(api, Lh)
            zg = top[2] - GRASP_BELOW_TOP * a[2]
            X = p + a * ((zg - p[2]) / a[2])
            d = X - Mc + np.array([0, 0, zoff])
            log.note(f"align{i} d_mm={r3(d)} palm-frame pen={r3(Rm.T @ (X - P))} thumb={r3(th)}")
            if np.linalg.norm(d) < 0.0015:
                break
            yield from api.move(xyz=api.palm_pose()[0] + d, R=R, seconds=0.4)
            yield from api.hold(0.2)

    # ---------------- approach the pen top from above with the pinch open
    yield from api.move(q=q_open, seconds=0.3)
    _, a, tip, top = pen_geom(api, Lh)
    g = top - a * GRASP_BELOW_TOP
    above = api.palm_for(g + np.array([0, 0, 0.07]), pinch, R)
    yield from api.move(xyz=above, R=R, seconds=2.0)
    yield from api.hold(0.2)
    yield from align(2, zoff=0.07)                    # xy alignment, 70 mm above the grasp height
    yield from api.move(xyz=api.palm_pose()[0] - np.array([0, 0, 0.07]), R=R, seconds=1.2)
    yield from api.hold(0.2)

    # ---------------- align, thumb to its post, index presses the pen onto it; retry if not held
    for attempt in range(3):
        yield from align(3)
        yield from api.move(q=q_pre, seconds=0.3)
        yield from api.hold(0.1)
        u, u_c = 0.0, None
        while u < 1.0:
            u += 0.025
            yield from api.move(q=index_closing(u), seconds=0.04)
            f = api.contact_forces("pen")
            if f[0] > 0.15 and f[1] > 0.15:
                u_c = u
                break
        u_fin = min(1.1, (u_c if u_c is not None else 0.8) + 0.2)
        yield from api.move(q=index_closing(u_fin), seconds=0.4)
        yield from api.hold(0.3)
        f = api.contact_forces("pen")
        log.note(f"grasp{attempt} u_contact={u_c} u_final={u_fin:.2f} forces={np.round(f, 2).tolist()} "
                 f"thumb={r3(api.tips()[0])} index={r3(api.tips()[1])} pen={r3(api.object_in_palm('pen')[0])}")
        if f[0] > 0.05 and f[1] > 0.05:
            break
        yield from api.move(q=q_open, seconds=0.4)
        yield from api.hold(0.2)

    # ---------------- lift out of the cup
    P = api.palm_pose()[0]
    yield from api.move(xyz=P + np.array([0, 0, 0.10]), R=R, seconds=1.5)
    yield from api.hold(0.3)
    _, a, tip, top = pen_geom(api, Lh)
    log.note(f"lifted tip h={(tip[2]-TABLE)*1000:.1f}mm lean={np.degrees(np.arccos(a[2])):.1f} forces={np.round(api.contact_forces('pen'), 2).tolist()}")

    # ---------------- carry to the drawing spot
    start = home + np.array([0.0, -0.08])
    # Drag along the pinch axis (palm z): a two-pad pinch is a hinge about that axis, so a drag
    # across it spins the pen over (iter 3 partial); along it the pads' friction resists tilting.
    dz = R[:2, 2] / np.linalg.norm(R[:2, 2])
    stroke = -0.042 * dz
    P = api.palm_pose()[0]
    _, a, tip, _ = pen_geom(api, Lh)
    off = P - tip                                    # palm relative to the pen tip
    yield from api.move(xyz=np.r_[start, TABLE + 0.012] + off, R=R, seconds=1.8)
    yield from api.hold(0.4)
    _, a, tip, _ = pen_geom(api, Lh)
    P = api.palm_pose()[0].copy()
    P[2] -= (tip[2] - TABLE) - 0.004                 # correct to 4 mm above the table
    yield from api.move(xyz=P.copy(), R=R, seconds=0.6)
    yield from api.hold(0.3)
    _, a, _, _ = pen_geom(api, Lh)
    lean_ref = np.degrees(np.arccos(min(1.0, a[2])))
    tip_pal0 = api.object_in_palm("pen")[0]          # free-hanging reference (palm frame)

    def state():
        _, a, tip, _ = pen_geom(api, Lh)
        c = api.object_in_palm("pen")[0]
        defl = tip_pal0[0] - c[0]                    # pen pushed toward the palm (palm x points down)
        return tip[2] - TABLE, defl, np.degrees(np.arccos(min(1.0, a[2]))), tip

    # slow descent until the tip touches (pen pose is exact; contact ~ h <= 0.1 mm)
    for k in range(200):
        h, defl, lean, tip = state()
        if h < 0.0001 or defl > 0.0003:
            break
        P[2] -= min(0.0003, max(0.00005, 0.25 * h))
        yield from api.move(xyz=P.copy(), R=R, seconds=0.02)
    yield from api.hold(0.2)
    h, defl, lean, tip = state()
    z_touch = P[2]
    log.note(f"touchdown k={k} h={h*1000:.2f}mm defl={defl*1000:.2f}mm lean={lean:.1f} ref={lean_ref:.1f} tip={r3(tip)}")

    # ---------------- stroke: light, force-limited contact (deflection of the compliant pinch)
    P0 = P.copy()
    N = 160
    zc = P[2]
    hs, ds, ls = [], [], []
    for k in range(1, N + 1):
        u = smooth(k / N)
        h, defl, lean, tip = state()
        hs.append(h); ds.append(defl); ls.append(lean)
        if lean - lean_ref > 10 or defl > 0.0006 or h < -0.0006:
            zc += 0.0002                             # pressing too hard / tilting: back off
        elif h > 0.0002:
            zc -= 0.00012                            # lost contact: come down
        elif h > -0.00005:
            zc -= 0.00004                            # barely touching: a little more
        zc = float(np.clip(zc, z_touch - 0.0015, z_touch + 0.004))
        xyz = P0 + np.r_[u * stroke, 0.0]
        xyz[2] = zc
        yield from api.move(xyz=xyz, R=R, seconds=0.02)
        if k % 40 == 0:
            log.note(f"stroke k={k} h={h*1000:.2f} defl={defl*1000:.2f} lean={lean:.1f} dz={(zc-z_touch)*1000:.2f}mm")
    h, defl, lean, tip = state()
    hs = np.array(hs)
    log.note(f"stroke done tip={r3(tip)} h mean={hs.mean()*1000:.2f} max={hs.max()*1000:.2f}mm "
             f"frac<0={np.mean(hs < 0.0):.2f} lean={lean:.1f} max_lean={max(ls):.1f} max_defl={max(ds)*1000:.2f}")

    # ---------------- lift and hold
    P = api.palm_pose()[0]
    yield from api.move(xyz=P + np.array([0, 0, 0.07]), R=R, seconds=1.2)
    yield from api.hold(1.5)
    _, a, tip, _ = pen_geom(api, Lh)
    log.note(f"final tip h={(tip[2]-TABLE)*1000:.1f}mm forces={np.round(api.contact_forces('pen'), 2).tolist()}")
