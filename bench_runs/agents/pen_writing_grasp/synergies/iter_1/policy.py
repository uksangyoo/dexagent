import numpy as np

TABLE = 0.75
R_PEN = 0.00375          # nominal pen radius (cancels out of the tip estimate)
PAD = 0.008              # pad centre sits ~8 mm inside the fingertip surface
PINCH_C = np.array([0.080, 0.040, 0.118])   # palm-frame thumb-index pinch centre (hand.py)
PINCH_AX = np.array([0.0, 0.0, -1.0])       # centre -> thumb
W_OPEN, W_PRE, SQUEEZE = 0.022, 0.0150, 0.005
GRASP_BELOW_TOP = 0.028
OTHER_X = 0.04           # other fingers' pads at most this far below the palm face
LIFT = 0.10
STROKE_OFF = np.array([-0.09, -0.01])        # stroke start relative to where the pen stood
DRAG = 0.13
LOG = []


def log(s):
    LOG.append(str(s))


def fmt(v):
    return np.round(np.asarray(v) * 1000, 1).tolist()


# ------------------------------------------------------------------ pen geometry
def pen_state(api, hseg):
    p, R = api.object_pose("pen")
    a = R[:, 2] if R[2, 2] > 0 else -R[:, 2]     # upward axis
    lowest = p[2] - hseg * a[2] - R_PEN            # lowest surface point z
    tip = p - a * (hseg + R_PEN)                   # lower axis end (scorer's tip)
    top = p + a * (hseg + R_PEN)
    lean = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    return p, a, tip, top, lowest, lean


# ------------------------------------------------------------------ synergy-space IK (skill)
def settle(api, s, sec=0.4, hold=0.2):
    """Command synergy coefficients s, wait, return measured palm-frame pads (5,3)."""
    yield from api.move(s=s, seconds=sec)
    yield from api.hold(hold)
    return api.tips()


def ik_step(J, s, tips, targets, lam=2e-5, maxstep=1.5):
    """One damped Gauss-Newton step in synergy space toward palm-frame pad targets."""
    k = len(s)
    A, b = [], []
    for f, T in targets.items():
        A.append(J[f]); b.append(np.asarray(T) - tips[f])
    for f in (2, 3, 4):   # weak: keep the other pads high (palm x small), clear of pen, cup and table
        A.append(0.3 * J[f][0:1]); b.append(0.3 * np.array([min(0.0, OTHER_X - tips[f][0])]))
    A.append(np.sqrt(lam) * np.eye(k)); b.append(-np.sqrt(lam) * s)
    ds = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    n = np.linalg.norm(ds)
    if n > maxstep:
        ds *= maxstep / n
    return np.clip(s + ds, -2.5, 2.5)


def broyden(J, ds, dt):
    """Rank-1 update of the measured pad Jacobian J (5,3,k)."""
    n = float(ds @ ds)
    if n < 1e-8:
        return J
    for f in range(5):
        J[f] += np.outer(dt[f] - J[f] @ ds, ds) / n
    return J


def solve_pads(api, J, s, tips, targets, iters=3):
    for _ in range(iters):
        s_new = ik_step(J, s, tips, targets)
        t_new = yield from settle(api, s_new)
        J = broyden(J, s_new - s, t_new - tips)
        s, tips = s_new, t_new
    return s, tips, J


# ------------------------------------------------------------------ policy
def run(api):
    try:
        yield from _run(api)
    finally:
        for line in LOG:
            api.note(line)


def _run(api):
    k = api.k
    log(f"k={k}")
    log("describe:\n" + api.describe())
    p_start, R_start = api.object_pose("pen")
    a0 = R_start[:, 2] if R_start[2, 2] > 0 else -R_start[:, 2]
    home = p_start[:2].copy() - a0[:2] * (p_start[2] - TABLE) / max(a0[2], 0.5)

    # ---- calibrate the pad Jacobian in free air (palm stays at its start pose)
    s0 = np.zeros(k)
    t0 = yield from settle(api, s0, 0.8, 0.3)
    d = 0.75
    J = np.zeros((5, 3, k))
    for j in range(k):
        e = np.zeros(k); e[j] = d
        tp = yield from settle(api, e, 0.4, 0.2)
        tm = yield from settle(api, -e, 0.5, 0.2)
        J[:, :, j] = (tp - tm) / (2 * d)
    log("mean tips mm " + str(fmt(t0)))
    for j in range(k):
        log(f"J[:, :, {j}] mm/sd " + str(np.round(J[:, :, j] * 1000).astype(int).tolist()))
    tips = yield from settle(api, s0, 0.4, 0.2)

    # ---- open pinch posture: thumb and index either side of the pinch centre
    tgt = {0: PINCH_C + W_OPEN * PINCH_AX, 1: PINCH_C - W_OPEN * PINCH_AX}
    s_o, tips, J = yield from solve_pads(api, J, s0, tips, tgt, 4)
    t_o = tips.copy()
    c_o = (t_o[0] + t_o[1]) / 2
    u_o = (t_o[0] - t_o[1]) / np.linalg.norm(t_o[0] - t_o[1])
    log(f"open s={np.round(s_o, 2).tolist()} tips={fmt(t_o)} gap={1000 * np.linalg.norm(t_o[0] - t_o[1]):.1f}")

    # ---- pre-grasp: pads just clear of a 7.5 mm pen at the same centre
    tgt = {0: c_o + W_PRE * u_o, 1: c_o - W_PRE * u_o}
    s_p, tips, J = yield from solve_pads(api, J, s_o, tips, tgt, 3)
    t_p = tips.copy()
    c_p = (t_p[0] + t_p[1]) / 2
    u_p = (t_p[0] - t_p[1]) / np.linalg.norm(t_p[0] - t_p[1])
    g_p = np.linalg.norm(t_p[0] - t_p[1])
    log(f"pre s={np.round(s_p, 2).tolist()} tips={fmt(t_p)} gap={1000 * g_p:.1f} c={fmt(c_p)}")
    # ---- squeeze: linear extrapolation past contact (pads would meet inside the pen)
    wc = R_PEN + PAD - SQUEEZE
    s_c = ik_step(J, s_p, tips, {0: c_p + wc * u_p, 1: c_p - wc * u_p}, lam=1e-5)
    log(f"close s={np.round(s_c, 2).tolist()} predicted thumb/index={fmt((tips + np.einsum('fik,k->fi', J, s_c - s_p))[:2])}")

    # pen has settled in the cup by now: tip-estimate calibration
    p, R = api.object_pose("pen")
    a = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
    hseg = (p[2] - TABLE - R_PEN) / a[2]
    _, a, tip, top, low, lean = pen_state(api, hseg)
    log(f"pen settled: hseg={hseg * 1000:.1f} lean={lean:.1f} top={fmt(top)} home={fmt(home)}")

    # ---- reach over the pen with the open pinch, descend, close
    Rd = api.hand_down()
    zg = top[2] - GRASP_BELOW_TOP
    q = top + a * (zg - top[2]) / a[2]               # axis point at grasp height
    yield from api.move(api.palm_for(q + [0, 0, 0.10], c_p, Rd), Rd, s=s_o, seconds=2.5)
    yield from api.move(api.palm_for(q + [0, 0, 0.04], c_p, Rd), Rd, seconds=1.0)
    yield from api.move(api.palm_for(q, c_p, Rd), Rd, seconds=1.0)
    yield from api.move(s=s_p, seconds=0.4)
    yield from api.move(s=s_c, seconds=0.8)
    yield from api.hold(0.4)
    f = api.contact_forces("pen")
    pp, Rp = api.object_in_palm("pen")
    log(f"grasp forces {np.round(f, 2).tolist()} pen_in_palm {fmt(pp)}")

    # ---- lift out of the cup
    P, _ = api.palm_pose()
    yield from api.move(P + [0, 0, LIFT], Rd, seconds=1.5)
    yield from api.hold(0.3)
    _, a, tip, top, low, lean = pen_state(api, hseg)
    log(f"lifted: tip h={1000 * (tip[2] - TABLE):.1f} lean={lean:.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # ---- carry to the stroke start
    S = home + STROKE_OFF
    P, _ = api.palm_pose()
    P1 = P + np.r_[S - tip[:2], 0]
    yield from api.move(P1, Rd, seconds=2.0)
    yield from api.hold(0.2)
    _, a, tip, top, low, lean = pen_state(api, hseg)
    P = P1 + np.r_[S - tip[:2], 0]
    P[2] -= max(0.0, low - TABLE - 0.012)
    yield from api.move(P, Rd, seconds=1.5)
    # fine descent to contact
    n = 0
    for n in range(400):
        _, a, tip, top, low, lean = pen_state(api, hseg)
        if low < TABLE + 0.0005:
            break
        P = P - [0, 0, 0.0002]
        yield from api.move(P, Rd, seconds=0.02)
    log(f"touchdown after {n} steps: low={1000 * (low - TABLE):.2f} lean={lean:.1f} tip={fmt(tip)}")
    P = P - [0, 0, 0.0025]
    yield from api.move(P, Rd, seconds=0.2)

    # ---- stroke: drag the hinge along -palm y (tip trails); keep the tip on the table
    dvec = -Rd[:, 1].copy(); dvec[2] = 0; dvec /= np.linalg.norm(dvec)
    Pb = P.copy()
    dz = 0.0
    N = 250
    inks, leans, path = 0, [], []
    for i in range(1, N + 1):
        u = i / N
        sm = u * u * (3 - 2 * u)
        _, a, tip, top, low, lean = pen_state(api, hseg)
        leans.append(lean)
        if low < TABLE + 0.001:
            inks += 1
        path.append(tip[:2].copy())
        if lean > 52:
            dz += 0.0002
        elif low > TABLE + 0.0008:
            dz -= 0.0004
        dz = max(dz, -0.05)
        yield from api.move(Pb + dvec * DRAG * sm + [0, 0, dz], Rd, seconds=0.02)
        if i % 25 == 0:
            log(f"drag {i}: low={1000 * (low - TABLE):.2f} lean={lean:.1f} dz={1000 * dz:.1f} tip={fmt(tip)}")
    path = np.array(path)
    log(f"stroke: ink steps {inks}/{N} max lean {max(leans):.1f} tip path {1000 * np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.1f}mm")

    # ---- lift and hold
    P, _ = api.palm_pose()
    yield from api.move(P + [0, 0, 0.08], Rd, seconds=1.5)
    yield from api.hold(1.0)
    _, a, tip, top, low, lean = pen_state(api, hseg)
    log(f"final: tip h={1000 * (tip[2] - TABLE):.1f} lean={lean:.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")
