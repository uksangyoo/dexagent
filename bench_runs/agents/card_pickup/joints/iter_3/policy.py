"""card_pickup, joints level.

Plan: (0) learn each finger's forward kinematics in-run from measured (q, pad) pairs
(exact multilinear cos/sin model, 81 features per finger) and build a pad IK on it;
(1) palm down, press index+middle pads on the tile and drag it toward the book's near
edge until it overhangs; (2) turn the hand fingers-down, palm facing the book edge,
slide index under / thumb over the overhang and pinch; (3) lift.
"""
import numpy as np
from scipy.optimize import least_squares

DT = 0.02
LOG = []
CURL = np.array([0, 1.6, 1.3, 1.1])
CURL_LITTLE = np.array([0.3, 1.4, 1.3, 1.1])
PAD = 0.008          # pad centre to finger surface
HALF_T = 0.0015      # tile half thickness

# tunables
S_PINCH = 0.031      # pinch midpoint distance from tile centre along long axis
M_COM = 0.007        # tile COM kept this far inside the book edge
LEVER = 0.004        # index under sits LEVER deeper (toward COM), thumb on top LEVER shallower
F_GRIP = 2.0         # target pad force [N]
TILT = 0.0           # fingers lean toward the book [rad]
SQUEEZE = 0.003
MIN_E_INDEX = 0.017  # index pad centre at least this far out from the book face


def note(api, s):
    LOG.append(f"{api.t:6.2f} {s}")


def smooth(u):
    return u * u * (3 - 2 * u)


def feats(Q):
    Q = np.atleast_2d(np.asarray(Q, float))
    n = Q.shape[0]
    F = np.ones((n, 1))
    for j in range(4):
        B = np.stack([np.ones(n), np.cos(Q[:, j]), np.sin(Q[:, j])], 1)
        F = (F[:, :, None] * B[:, None, :]).reshape(n, -1)
    return F


class Hand:
    """Pad forward kinematics learned from measurements, and IK on it."""

    def __init__(self, api):
        self.api = api
        lim = api.joint_limits
        self.lo, self.hi = lim[:, 0].copy(), lim[:, 1].copy()
        self.mid = (self.lo + self.hi) / 2
        self.amp = 0.47 * (self.hi - self.lo)
        self.blo, self.bhi = self.mid - self.amp, self.mid + self.amp
        self.ilo = self.blo.copy()
        self.ilo[6:8] = np.maximum(self.ilo[6:8], 0.0)
        self.W = [None] * 5

    def calibrate(self, T=5.0):
        api = self.api
        q0 = api.q()
        rng = np.random.default_rng(7)
        f = rng.uniform(0.3, 1.3, 20)
        ph = rng.uniform(0, 2 * np.pi, 20)
        Q, P = [], []
        n = int(T / DT)
        for k in range(n):
            t = k * DT
            e = min(1.0, t / 0.4)
            q = (1 - e) * q0 + e * (self.mid + self.amp * np.sin(2 * np.pi * f * t + ph))
            yield from api.move(q=q, seconds=DT)
            Q.append(api.q())
            P.append(api.tips())
        Q, P = np.array(Q), np.array(P)
        self.Q, self.P = Q, P
        te = np.arange(len(Q)) % 5 == 0
        errs = []
        for i in range(5):
            F = feats(Q[:, 4 * i:4 * i + 4])
            W = np.linalg.lstsq(F[~te], P[~te, i], rcond=1e-12)[0]
            errs.append(np.abs(F[te] @ W - P[te, i]).max() * 1000)
            self.W[i] = np.linalg.lstsq(F, P[:, i], rcond=1e-12)[0]
        note(api, f"calib test max err mm {np.round(errs, 3).tolist()}")
        note(api, f"limits {np.round(self.lo, 2).tolist()} {np.round(self.hi, 2).tolist()}")

    def fk(self, i, qi):
        return feats(qi)[0] @ self.W[i]

    def fk_all(self, q):
        return np.array([self.fk(i, q[4 * i:4 * i + 4]) for i in range(5)])

    def ik(self, i, target, seed=None, extra=None, reg=1e-3):
        sl = slice(4 * i, 4 * i + 4)
        lo, hi = (self.ilo if i == 1 else self.blo)[sl], self.bhi[sl]
        target = np.asarray(target, float)
        seeds = [] if seed is None else [np.asarray(seed, float)]
        d = np.linalg.norm(self.P[:, i] - target, axis=1)
        seeds += [self.Q[k, sl] for k in np.argsort(d)[:4]]
        best = None
        for s in seeds:
            s = np.clip(s, lo + 1e-6, hi - 1e-6)

            def res(q, s=s):
                r = [self.fk(i, q) - target, reg * (q - s)]
                if extra is not None:
                    r.append(extra(q))
                return np.concatenate(r)
            x = least_squares(res, s, bounds=(lo, hi), max_nfev=200).x
            err = float(np.linalg.norm(self.fk(i, x) - target))
            if best is None or err < best[1] - 2e-4:
                best = (x, err)
            if seed is not None and err < 5e-4:
                break
        return best


def index_extra(q):
    return np.array([0.01 * (q[1] + q[2] + q[3] - np.pi / 2)])


def r_pinch(a, tilt=0.0):
    """Fingers down, palm face toward -a (the book), tilted by `tilt` about palm y."""
    x = -np.array([a[0], a[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0, 0, -1.0])
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    c, s = np.cos(tilt), np.sin(tilt)
    return R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def finger_line(api, H, q, paths, seconds, xyz=None):
    n = max(1, round(seconds / DT))
    q = q.copy()
    worst = 0.0
    for k in range(1, n + 1):
        u = smooth(k / n)
        for i, (s, e, ex) in paths.items():
            sol = H.ik(i, (1 - u) * s + u * e, seed=q[4 * i:4 * i + 4], extra=ex)
            q[4 * i:4 * i + 4] = sol[0]
            worst = max(worst, sol[1])
        yield from api.move(q=q, seconds=DT)
    return q, worst


def run(api):
    try:
        yield from main(api)
    finally:
        for line in LOG[-45:]:
            api.note(line)


def main(api):
    H = Hand(api)
    p0, R0 = api.object_pose("tile")
    bp, _ = api.object_pose("book")
    edge_y = bp[1] - 0.11
    k_long = 0 if abs(R0[1, 0]) > abs(R0[1, 1]) else 1
    note(api, f"tile0 {np.round(p0, 4).tolist()} book {np.round(bp, 4).tolist()} klong {k_long}")
    yield from H.calibrate(5.0)

    # ---------------------------------------------------------------- press posture
    q = np.asarray(api.OPEN, float).copy()
    q[4:8] = [0, 0.7, 0.5, 0.4]
    q[8:12] = [0, 0.7, 0.5, 0.4]
    q[12:16] = CURL
    q[16:20] = CURL_LITTLE
    q = np.clip(q, H.blo, H.bhi)
    pads = H.fk_all(q)
    press_x = min(pads[1, 0], pads[2, 0])
    if pads[0, 0] > press_x - 0.03:   # thumb too low: pick the highest calibration thumb posture
        k = int(np.argmin(H.P[:, 0, 0]))
        q[0:4] = H.Q[k, 0:4]
        pads = H.fk_all(q)
    note(api, f"press pads palm {np.round(pads, 3).tolist()}")
    q_press = q.copy()

    c, R = api.object_pose("tile")
    a = R[:, k_long].copy(); a[2] = 0; a /= np.linalg.norm(a)
    if a[1] > 0:
        a = -a
    Rd = api.hand_down(0.7)
    mid_pad = (pads[1] + pads[2]) / 2
    tgt = c - 0.010 * a
    tgt[2] = c[2] + HALF_T + PAD + 0.010
    palm = tgt - Rd @ mid_pad
    yield from api.move(xyz=palm + [0, 0, 0.06], R=Rd, q=q_press, seconds=2.0)
    yield from api.move(xyz=palm, seconds=1.0)
    for k in range(80):
        f = api.contact_forces("tile")
        if f[1] + f[2] > 0.8:
            break
        cur = api.palm_pose()[0]
        yield from api.move(xyz=cur - [0, 0, 0.0005], seconds=DT)
    cur = api.palm_pose()[0]
    yield from api.move(xyz=cur - [0, 0, 0.004], seconds=0.2)
    note(api, f"pressed forces {np.round(api.contact_forces('tile'), 2).tolist()} book {np.round(api.contact_forces('book'), 2).tolist()}")

    # ---------------------------------------------------------------- drag to the edge
    y_goal = edge_y + M_COM
    start_palm = api.palm_pose()[0].copy()
    for k in range(300):
        c, R = api.object_pose("tile")
        if c[1] <= y_goal:
            break
        cur = api.palm_pose()[0]
        if start_palm[1] - cur[1] > (p0[1] - y_goal) + 0.04:
            note(api, "drag: palm travelled too far, tile slipping")
            break
        step = min(0.0008, c[1] - y_goal + 0.0002)
        yield from api.move(xyz=cur - [0, step, 0], seconds=DT)
    cur = api.palm_pose()[0]
    yield from api.move(xyz=cur + [0, 0.002, 0], seconds=0.3)
    c, R = api.object_pose("tile")
    note(api, f"dragged tile {np.round(c, 4).tolist()} goal_y {y_goal:.4f} forces {np.round(api.contact_forces('tile'), 2).tolist()}")
    cur = api.palm_pose()[0]
    yield from api.move(xyz=cur + [0, 0, 0.02], seconds=0.4)
    cur = api.palm_pose()[0]
    yield from api.move(xyz=[cur[0], cur[1], max(cur[2] + 0.05, 1.0)], seconds=0.8)

    # ---------------------------------------------------------------- plan the pinch
    c, R = api.object_pose("tile")
    a = R[:, k_long].copy(); a[2] = 0; a /= np.linalg.norm(a)
    if a[1] > 0:
        a = -a
    s_p = S_PINCH
    e_idx = (edge_y - (c + (s_p - LEVER) * a)[1])
    if e_idx < MIN_E_INDEX:
        s_p = min(0.037, s_p + (MIN_E_INDEX - e_idx) / max(0.5, abs(a[1])))
    P = c + s_p * a
    note(api, f"tile now {np.round(c, 4).tolist()} tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} pinch {np.round(P, 4).tolist()} edge_clear {1000 * (edge_y - P[1]):.1f}mm index_clear {1000 * (edge_y - (P - LEVER * a)[1]):.1f}mm s {1000 * s_p:.1f}")
    Rp = r_pinch(a, TILT)
    ez = np.array([0, 0, 1.0])
    ex = np.array([1.0, 0, 0])
    hw_t_pre, hw_i_pre = HALF_T + PAD + 0.006, HALF_T + PAD + 0.004
    hw_cl = HALF_T + PAD - SQUEEZE
    best = None
    C0 = np.array([0.08, 0.04, 0.118])
    for dx in (0, -0.012, 0.012):
        for dy in (0, -0.012, 0.012):
            for dz in (0, -0.012, 0.012):
                C = C0 + [dx, dy, dz]
                tp = H.ik(0, C - hw_t_pre * ez - LEVER * ex)
                tc = H.ik(0, C - (hw_cl - 0.004) * ez - LEVER * ex, seed=tp[0])
                ip = H.ik(1, C + hw_i_pre * ez + LEVER * ex, extra=index_extra)
                ic = H.ik(1, C + (hw_cl - 0.004) * ez + LEVER * ex, seed=ip[0], extra=index_extra)
                err = max(tp[1], tc[1], ip[1], ic[1])
                score = err + 0.05 * np.linalg.norm([dx, dy, dz])
                if best is None or score < best[0]:
                    best = (score, err, C, tp[0], ip[0])
    _, err, C, qt, qi = best
    note(api, f"pinch C {np.round(C, 3).tolist()} ik err {1000 * err:.1f}mm qt {np.round(qt, 2).tolist()} qi {np.round(qi, 2).tolist()}")
    q = q_press.copy()
    q[0:4], q[4:8] = qt, qi
    q[8:12] = CURL
    q[12:16] = CURL
    q[16:20] = CURL_LITTLE
    q = np.clip(q, H.blo, H.bhi)
    pads = H.fk_all(q)
    palm_f = P - Rp @ C
    stand = palm_f + 0.045 * a
    note(api, f"prepinch pads world {np.round(stand + pads @ Rp.T, 3).tolist()} palm_f {np.round(palm_f, 3).tolist()}")
    yield from api.move(xyz=stand + [0, 0, 0.07], R=Rp, q=q, seconds=2.0)
    yield from api.move(xyz=stand, seconds=1.0)
    yield from api.hold(0.3)
    want = stand + pads[:2] @ Rp.T
    meas = api.tips_world()[:2]
    corr = np.clip((want - meas).mean(0), -0.01, 0.01)
    note(api, f"prepinch tip err mm {np.round(1000 * (want - meas), 1).tolist()}")
    palm_f, stand = palm_f + corr, stand + corr
    yield from api.move(xyz=stand, seconds=0.3)
    yield from api.move(xyz=palm_f, seconds=1.2)
    note(api, f"at pinch forces tile {np.round(api.contact_forces('tile'), 2).tolist()} book {np.round(api.contact_forces('book'), 2).tolist()}")

    # ---------------------------------------------------------------- close
    T = lambda hw: C - hw * ez - LEVER * ex
    I = lambda hw: C + hw * ez + LEVER * ex
    q, e1 = yield from finger_line(api, H, q, {1: (I(hw_i_pre), I(hw_cl), index_extra)}, 0.5)
    q, e2 = yield from finger_line(api, H, q, {0: (T(hw_t_pre), T(hw_cl), None)}, 0.6)
    hw_i, hw_t = hw_cl, hw_cl
    for k in range(40):
        f = api.contact_forces("tile")
        if f[0] >= F_GRIP and f[1] >= F_GRIP:
            break
        if f[1] < F_GRIP and hw_i > hw_cl - 0.006:
            hw_i -= 0.0003
            q[4:8] = H.ik(1, I(hw_i), seed=q[4:8], extra=index_extra)[0]
        if f[0] < F_GRIP and hw_t > hw_cl - 0.006:
            hw_t -= 0.0003
            q[0:4] = H.ik(0, T(hw_t), seed=q[0:4])[0]
        yield from api.move(q=q, seconds=DT)
    yield from api.hold(0.3)
    c, R = api.object_pose("tile")
    note(api, f"closed ikerr {1000 * e1:.1f} {1000 * e2:.1f} hw_i {1000 * hw_i:.1f} hw_t {1000 * hw_t:.1f} forces {np.round(api.contact_forces('tile'), 2).tolist()} tile {np.round(c, 4).tolist()} book {np.round(api.contact_forces('book'), 2).tolist()}")

    # ---------------------------------------------------------------- lift
    yield from api.move(xyz=palm_f + [0, 0, 0.015], seconds=0.8)
    c, R = api.object_pose("tile")
    note(api, f"lift15 tile {np.round(c, 4).tolist()} tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} forces {np.round(api.contact_forces('tile'), 2).tolist()}")
    yield from api.move(xyz=palm_f + 0.03 * a + [0, 0, 0.14], seconds=1.8)
    yield from api.hold(0.5)
    c, R = api.object_pose("tile")
    tw = api.tips_world()
    note(api, f"lifted tile {np.round(c, 4).tolist()} tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} lift {1000 * (c[2] - p0[2]):.0f}mm d_pinch {1000 * np.linalg.norm(c - (tw[0] + tw[1]) / 2):.0f}mm forces {np.round(api.contact_forces('tile'), 2).tolist()}")
