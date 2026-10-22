"""pen_pickup_pivot, synergies level (iteration 4 of 4 budgeted).

Plan:
(1) measure, in the air, a fixed 1-D synergy path s(lam) = S_O + lam (S_C - S_O) between two
    postures logged in earlier iterations: S_C = 19 mm thumb-index pinch, S_O = 38 mm pinch
    with the SAME pinch line (index almost fixed, thumb sweeps toward it); middle/ring/little
    sit >= 30 mm off the pinch line on one side.  Pads are measured at 5 points on the path.
(2) orient the palm so the measured pinch line is horizontal and perpendicular to the pen,
    the other fingers straight above it (rolled a little about the hinge), fingers pointing
    >= 25 deg down so the wrist/forearm stay high (iter 4: horizontal fingers -> arm could not
    get low).  Choose among candidates by reach heuristics, fall back on IK failure / arm
    not tracking.
(3) put the pen between the open pads (near the index), touch down on the table, close along
    the path while compensating the palm height from the measured path so the tips keep
    sliding on the table; check contact forces and a 10 mm test lift; retry if not held.
(4) lift slowly: the pen hangs on the frictionless thumb-index hinge, its far end drags on
    the table and the pen swings upright while standing on the table; lift clear; level.
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
BASE_XY = np.array([0.30, -0.58])
OFF = 0.046                  # grasp point from the pen centre toward the short end E1 [m]
S_C = np.array([1.8, -0.7, -0.4, 0.6, 0.3, -0.3])       # closed pinch (measured iter 4/5: 19.8 mm)
S_O = np.array([2.37, -0.38, -0.98, 0.8, 1.42, 0.01])   # open pinch (measured iter 4: 37.7 mm, same line)
LAMS = (-0.35, 0.0, 0.5, 1.0, 1.25)
LAM_GRIP = 1.25
SHIFTS = (0.002, 0.005, -0.001, 0.003)   # pen axis target past the open-pinch midpoint toward the index, per attempt
D = []


def note(msg):
    D.append(str(msg))


def unit(x):
    x = np.asarray(x, float)
    return x / (np.linalg.norm(x) + 1e-12)


def s_of(lam):
    return S_O + lam * (S_C - S_O)


def mm(x):
    return np.round(np.asarray(x, float) * 1000).astype(int).tolist()


def tilt_deg(R):
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def ang_deg(x, y):
    return float(np.degrees(np.arccos(np.clip(unit(x) @ unit(y), -1, 1))))


def pen_axis(api):
    p, R = api.object_pose("pen")
    a = R[:, 2].copy()
    a[2] = 0
    return p, R, unit(a)


def move_until(api, cond, **kw):
    g = api.move(**kw)
    for _ in g:
        yield
        if cond():
            g.close()
            return True
    return False


# ------------------------------------------------------------------ calibration (skill)
class Model:
    """Measured synergy path: pads at each lam (palm frame), hinge v, 'up' (other fingers' side)."""

    def __init__(self, P):
        self.P = P
        feats = {l: (np.linalg.norm(P[l][1] - P[l][0]), 0.5 * (P[l][0] + P[l][1]), unit(P[l][1] - P[l][0])) for l in P}
        self.feats = feats
        ap_c, self.c_c, v_c = feats[1.0]
        _, _, v_0 = feats[0.0]
        v = unit(v_c + v_0)
        ups = []
        for l in (0.0, 1.0):
            for f in (2, 3, 4):
                d = P[l][f] - self.c_c
                ups.append(unit(d - (d @ v) * v))
        up = np.mean(ups, axis=0)
        self.up = unit(up - (up @ v) * v)
        # wider opening if it keeps the pinch line and the other fingers clear
        ap_w, c_w, v_w = feats[-0.35]
        oth_w = min((P[-0.35][f] - self.c_c) @ self.up for f in (2, 3, 4))
        dh_w = abs((P[-0.35][0] - P[-0.35][1]) @ self.up)
        self.lam_open = -0.35 if (ap_w > 0.042 and ang_deg(v_w, v_c) < 25 and oth_w > 0.022 and dh_w < 0.012) else 0.0
        ap_o, self.c_o, v_o = feats[self.lam_open]
        self.v = unit(v_o + v_c)
        self.e3 = np.cross(self.v, self.up)
        self.s_open = s_of(self.lam_open)
        self.c_tgt = self.c_o + 0.002 * self.v       # where the pen axis goes (palm frame)
        self.oth = np.mean([P[1.0][f] for f in (2, 3, 4)], axis=0) - self.c_c
        note(f"path lam_open={self.lam_open} ap={[round(feats[l][0] * 1000, 1) for l in LAMS]} "
             f"ang(v_l,v_c)={[round(ang_deg(feats[l][2], v_c)) for l in LAMS]} oth_w={oth_w * 1000:.0f} dh_w={dh_w * 1000:.0f}")
        note(f"P_open={mm(P[self.lam_open])} P_c={mm(P[1.0])} v={np.round(self.v, 2).tolist()} up={np.round(self.up, 2).tolist()}")

    def hz(self, lam, R):
        """Mean world-z offset of thumb/index pads (palm at origin), piecewise linear in lam."""
        ls = [l for l in LAMS]
        zs = [float(np.mean([(R @ self.P[l][f])[2] for f in (0, 1)])) for l in ls]
        return float(np.interp(lam, ls, zs))


def calibrate(api):
    P = {}
    for i, lam in enumerate(LAMS):
        yield from api.move(s=s_of(lam), seconds=0.9 if i == 0 else 0.5)
        yield from api.hold(0.45)
        P[lam] = api.tips().copy()
    return Model(P)


# ------------------------------------------------------------------ orientation candidates (skill)
def candidates(M, p, a):
    """Palm rotations mapping the hinge v to the horizontal perpendicular of the pen, the
    other fingers' side 'up', then rolled phi about the hinge.  Sorted by a reach score."""
    u0 = unit(np.cross([0, 0, 1.0], a))
    radial = unit(p[:2] - BASE_XY)
    out = []
    for su in (1.0, -1.0):
        u = su * u0
        w = np.cross(u, [0, 0, 1.0])
        R0 = np.column_stack([u, [0, 0, 1.0], w]) @ np.column_stack([M.v, M.up, M.e3]).T
        for phi in (-10, 0, 10, 20, 30):
            R = Rotation.from_rotvec(u * np.radians(phi)).as_matrix() @ R0
            fz = R[2, 2]
            lean = (R @ M.oth) @ a
            score = (R[:2, 2] @ radial + 0.5 * min(abs(lean), 0.015) / 0.015
                     - 3.0 * max(0.0, fz + 0.45) - 1.0 * max(0.0, -fz - 0.85) - 2.0 * max(0.0, R[2, 0] + 0.3))
            side = -1.0 if lean > 0 else 1.0       # short end E1 away from the other fingers
            out.append((score, su, phi, R, side))
    out.sort(key=lambda c: -c[0])
    return out


def grasp_point(p, a, side):
    g = p + side * OFF * a
    g[2] = p[2]
    return g


def target_world(api, shift=0.002):
    T = api.tips_world()
    return 0.5 * (T[0] + T[1]) + shift * unit(T[1] - T[0])


# ------------------------------------------------------------------ grasp
def approach(api, M, cand, from_above=True, shift=0.002):
    """Hover the open pinch 24 mm above the grasp point. Returns (g, side, ok)."""
    _, su, phi, R, side = cand
    c_tgt = M.c_o + shift * M.v
    p, _, a = pen_axis(api)
    g = grasp_point(p, a, side)
    if from_above:
        yield from api.move(xyz=api.palm_for(g + np.array([0, 0, 0.07]), c_tgt, R), R=R, s=M.s_open, seconds=2.0)
        yield from api.hold(0.2)
    pre = api.palm_for(g + np.array([0, 0, 0.024]), c_tgt, R)
    yield from api.move(xyz=pre, R=R, s=M.s_open, seconds=1.0)
    yield from api.hold(0.3)
    for _ in range(2):
        p, _, a = pen_axis(api)
        g = grasp_point(p, a, side)
        err = g - target_world(api, shift)
        err[2] = 0
        P_cmd, _ = api.palm_pose()
        yield from api.move(xyz=P_cmd + err, seconds=0.4)
        yield from api.hold(0.25)
    P_cmd, _ = api.palm_pose()
    Pm, _ = api.measured_palm()
    track = float(np.linalg.norm(Pm - P_cmd))
    T = api.tips_world()
    note(f"approach su={su:+.0f} phi={phi} track_err={track * 1000:.1f}mm pads_z={mm(T[:, 2] - TABLE)} "
         f"tgt-g={mm(target_world(api, shift) - g)} shift={shift * 1000:.0f} fing={np.round(R[:, 2], 2).tolist()}")
    return g, side, track < 0.012


def touch_down(api, depth=0.04, secs=4.0):
    ref = api.tips().copy()
    pen0, _ = api.object_pose("pen")
    P_cmd, _ = api.palm_pose()

    def touched():
        dev = np.linalg.norm(api.tips() - ref, axis=1)
        if dev[0] > 0.0015 or dev[1] > 0.0015 or dev[2:].max() > 0.004:
            return True
        return np.linalg.norm(api.object_pose("pen")[0] - pen0) > 0.0015
    hit = yield from move_until(api, touched, xyz=P_cmd - np.array([0, 0, depth]), seconds=secs)
    yield from api.hold(0.15)
    return hit


def close_pinch(api, M, press):
    T = api.tips_world()
    zt = T[:2, 2].copy()
    P_cmd, R = api.palm_pose()
    x0 = P_cmd.copy()
    h0 = M.hz(M.lam_open, R)
    extra = 0.0
    n = 8
    for i in range(1, n + 1):
        lam = M.lam_open + (LAM_GRIP - M.lam_open) * i / n
        T = api.tips_world()
        lifted = min(T[0, 2] - zt[0], T[1, 2] - zt[1]) - 0.002
        extra += float(np.clip(lifted, 0.0, 0.003))
        z = x0[2] - press - (M.hz(lam, R) - h0) - extra
        yield from api.move(xyz=np.array([x0[0], x0[1], z]), s=s_of(lam), seconds=0.2)
    yield from api.hold(0.4)


def try_grasp(api, M, press):
    """Touch down, close, test-lift. Returns True if the pen comes up with the pinch."""
    for k in range(3):
        hit = yield from touch_down(api)
        f = api.contact_forces("pen")
        T = api.tips_world()
        note(f"touch{k} hit={hit} pads_z={mm(T[:, 2] - TABLE)} f={np.round(f, 2).tolist()}")
        if not hit:
            return None          # arm could not get down: try another orientation
        if max(f[0], f[1]) > 0.05 and k < 2:
            # a tip landed on the pen: back up and shift that tip away from the pen along the hinge
            vw = unit(T[1] - T[0])
            shift = -0.004 * vw if f[0] >= f[1] else 0.004 * vw
            shift[2] = 0.008
            P_cmd, _ = api.palm_pose()
            yield from api.move(xyz=P_cmd + shift, seconds=0.4)
            yield from api.hold(0.2)
            continue
        break
    yield from close_pinch(api, M, press)
    f = api.contact_forces("pen")
    p0, R0 = api.object_pose("pen")
    T = api.tips_world()
    note(f"grip press={press * 1000:.1f} f={np.round(f, 2).tolist()} pads_z={mm(T[:, 2] - TABLE)} hinge={np.round(unit(T[1] - T[0]), 2).tolist()}")
    P_cmd, _ = api.palm_pose()
    yield from api.move(xyz=P_cmd + np.array([0, 0, 0.010]), seconds=0.6)
    yield from api.hold(0.25)
    p1, R1 = api.object_pose("pen")
    f = api.contact_forces("pen")
    held = ((p1[2] - p0[2] > 0.002) or (tilt_deg(R0) - tilt_deg(R1) > 1.5)) and f[0] > 0.02 and f[1] > 0.02
    note(f"testlift dz={(p1[2] - p0[2]) * 1000:.1f}mm dtilt={tilt_deg(R0) - tilt_deg(R1):.1f} f={np.round(f, 2).tolist()} held={held}")
    return held


# ------------------------------------------------------------------ pivot / hang (skills)
def rotate_about(api, Rc, point, seconds, dz=0.0):
    P, R = api.palm_pose()
    yield from api.move(xyz=point + Rc @ (P - point) + np.array([0, 0, dz]), R=Rc @ R, seconds=seconds)


def level_hinge(api, max_deg=15.0, dz=0.004):
    T = api.tips_world()
    m = 0.5 * (T[0] + T[1])
    w = unit(T[1] - T[0])
    wh = w.copy()
    wh[2] = 0
    wh = unit(wh)
    rv = np.cross(w, wh)
    ang = min(np.arcsin(min(1.0, np.linalg.norm(rv))), np.radians(max_deg))
    Rc = Rotation.from_rotvec(unit(rv) * ang).as_matrix() if ang > 1e-3 else np.eye(3)
    yield from rotate_about(api, Rc, m, 0.8, dz)
    yield from api.hold(0.3)
    note(f"level_hinge {np.degrees(ang):.1f}deg")


def level_pen(api, rounds=2, seconds=0.8):
    """Rotate the hand about the pinch so the hanging pen's axis becomes vertical."""
    for _ in range(rounds):
        _, Rp = api.object_pose("pen")
        a = Rp[:, 2] * (-1 if Rp[2, 2] > 0 else 1)  # pointing down
        rv = np.cross(a, [0, 0, -1.0])
        ang = np.arcsin(min(1.0, np.linalg.norm(rv)))
        if ang < np.radians(1.5):
            break
        ang = min(ang, np.radians(35.0))
        Rc = Rotation.from_rotvec(unit(rv) * ang).as_matrix()
        T = api.tips_world()
        yield from rotate_about(api, Rc, 0.5 * (T[0] + T[1]), seconds)
        yield from api.hold(0.5)


# ------------------------------------------------------------------ policy
def run(api):
    try:
        yield from main(api)
    except Exception as e:
        tb = e.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        note(f"EXC line {tb.tb_lineno}: {type(e).__name__}: {str(e)[:300]}")
    finally:
        for line in D[-40:]:
            api.note(line)


def main(api):
    M = yield from calibrate(api)
    p, Rpen, a = pen_axis(api)
    r_pen = p[2] - TABLE
    cands = candidates(M, p, a)
    note("cands " + " ".join(f"({c[1]:+.0f},{c[2]},{c[0]:.2f})" for c in cands[:5]))

    # ---- grasp with orientation fallbacks and retries
    held = False
    ci = 0
    attempt = 0
    cand = None
    while not held and ci < len(cands) and api.t < 40.0:
        if cand is None:
            cand = cands[ci]
            try:
                g, side, ok = yield from approach(api, M, cand, from_above=True, shift=SHIFTS[attempt % 4])
            except ValueError as e:
                note(f"IK fail cand {ci}: {str(e)[:80]}")
                ok = False
            if not ok:
                ci += 1
                cand = None
                P_cmd, _ = api.palm_pose()
                try:
                    yield from api.move(xyz=P_cmd + np.array([0, 0, 0.06]), s=M.s_open, seconds=1.0)
                except ValueError:
                    pass
                continue
        else:
            try:
                g, side, ok = yield from approach(api, M, cand, from_above=False, shift=SHIFTS[attempt % 4])
            except ValueError as e:
                note(f"IK fail re-approach: {str(e)[:80]}")
                ci += 1
                cand = None
                continue
        press = 0.001 + 0.0015 * attempt
        held = yield from try_grasp(api, M, press)
        attempt += 1
        if held is None:
            held = False
            ci += 1
            cand = None
            P_cmd, _ = api.palm_pose()
            yield from api.move(xyz=P_cmd + np.array([0, 0, 0.06]), s=M.s_open, seconds=1.0)
            continue
        if not held:
            P_cmd, _ = api.palm_pose()
            yield from api.move(xyz=P_cmd + np.array([0, 0, 0.015]), s=M.s_open, seconds=0.6)
            yield from api.hold(0.3)
            p2, _, a2 = pen_axis(api)
            if ang_deg(a2, a) > 12 and ang_deg(-a2, a) > 12:
                note("pen yaw changed: re-plan orientation")
                a = a2
                cands = candidates(M, p2, a2)
                ci, cand = 0, None
                P_cmd, _ = api.palm_pose()
                yield from api.move(xyz=P_cmd + np.array([0, 0, 0.05]), seconds=0.8)
    if not held:
        note("no grasp")
        return

    # ---- level the hinge, then lift slowly: the pen swings upright on the table
    try:
        yield from level_hinge(api)
    except ValueError:
        note("level_hinge IK fail")
    st = {"min_tilt": 90.0, "half": None, "stood": False}

    def mon():
        p, Rp = api.object_pose("pen")
        t = tilt_deg(Rp)
        az = abs(Rp[2, 2])
        st["min_tilt"] = min(st["min_tilt"], t)
        if az > 0.5:
            h = (p[2] - r_pen - TABLE) / az
            if st["half"] is None or h < st["half"]:
                st["half"] = h
            if t < 18.0 and h - st["half"] < 0.002:
                st["stood"] = True
        return t < 4.0

    P_cmd, _ = api.palm_pose()
    up = yield from move_until(api, mon, xyz=P_cmd + np.array([0, 0, 0.17]), seconds=7.0)
    yield from api.hold(0.3)
    mon()
    p, Rp = api.object_pose("pen")
    T = api.tips_world()
    note(f"lift upright_stop={up} stood={st['stood']} tilt={tilt_deg(Rp):.1f} min_tilt={st['min_tilt']:.1f} "
         f"half={None if st['half'] is None else round(st['half'] * 1000, 1)} pinch_h={mm(0.5 * (T[0, 2] + T[1, 2]) - TABLE)} f={np.round(api.contact_forces('pen'), 2).tolist()}")

    if not st["stood"] and api.t < 50.0:
        # pen hangs tilted (or never touched): make it vertical, lower onto the table, lift again
        try:
            yield from level_pen(api, rounds=2)
        except ValueError:
            note("level_pen IK fail")
        _, Rp = api.object_pose("pen")
        t0 = tilt_deg(Rp)
        q0, _ = api.object_in_palm("pen")
        P_cmd, _ = api.palm_pose()

        def touch():
            mon()
            _, Rq = api.object_pose("pen")
            qq, _ = api.object_in_palm("pen")
            return tilt_deg(Rq) > t0 + 4 or np.linalg.norm(qq - q0) > 0.002
        hit = yield from move_until(api, touch, xyz=P_cmd - np.array([0, 0, 0.13]), seconds=6.0)
        yield from api.hold(0.2)
        mon()
        p, Rp = api.object_pose("pen")
        note(f"touchdown hit={hit} stood={st['stood']} tilt={tilt_deg(Rp):.1f}")

    # ---- lift clear and hang vertical
    p, Rp = api.object_pose("pen")
    half = st["half"] if st["half"] is not None else 0.07
    low = p[2] - half * abs(Rp[2, 2]) - r_pen - TABLE
    P_cmd, _ = api.palm_pose()
    dz = float(np.clip(0.07 - low, 0.03, 0.12))
    try:
        yield from api.move(xyz=P_cmd + np.array([0, 0, dz]), seconds=2.5)
    except ValueError:
        note("lift IK fail")
    yield from api.hold(0.5)
    try:
        yield from level_pen(api, rounds=2)
    except ValueError:
        note("level_pen IK fail")
    yield from api.hold(max(0.5, min(1.2, 57.0 - api.t)))
    p, Rp = api.object_pose("pen")
    low = p[2] - half * abs(Rp[2, 2]) - r_pen - TABLE
    note(f"final tilt={tilt_deg(Rp):.1f} low_est={low * 1000:.0f}mm f={np.round(api.contact_forces('pen'), 2).tolist()} t={api.t:.1f}")
