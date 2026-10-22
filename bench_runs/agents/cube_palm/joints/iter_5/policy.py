"""cube_palm at the joints level (v3).

Fingers 1-4 are held at one fixed "wall" posture (MCP ~1.45 rad): palm down
they hang as a vertical wall on the cube's fingertip-side face, fingers down
they form a horizontal shelf, palm up a wall that cradles the cube against
the palm. Only the thumb moves (in-sim IK), pinching the cube against the wall.

Turning over is one pitch about palm y (palm z is the pinch axis, and any turn
about it slips the cube: the pinch has almost no torsional grip). Fingers
swing down (gravity puts the cube on the shelf), then on toward the robot to
palm-up. The thumb lets go at 30 deg past fingers-down and the cube slides
along the wall into the palm/wall corner; the hand is held palm-up with the
fingertips 30 deg down while the palm carries it. Back: pitch to fingers-down
(cube on the shelf), lower the shelf near the table and pitch palm-down about
the outermost pads so the cube slides off onto the table against the wall,
then back off and lift.

Joint-level IK is built here: `solve` servos the measured pad positions
(api.tips) onto palm-frame targets with a finite-difference Jacobian and
Broyden updates, with the hand in the air before any contact.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TABLE = 0.75
PAD = 0.008          # pad centre sits ~8 mm inside the fingertip surface
TILT = np.radians(30)   # palm-up carry: fingertips tilted down by this much
DIAG = []


def d(msg):
    DIAG.append(msg)


def mm(a):
    return np.round(np.asarray(a) * 1000).astype(int).tolist()


def r2(a):
    return np.round(np.asarray(a), 2).tolist()


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ----------------------------------------------------------------- skills
def probe(api, q, t=0.12, settle=0.1):
    """Command finger joints, let them settle, return measured pads (palm frame)."""
    yield from api.move(q=q, seconds=t)
    yield from api.hold(settle)
    return api.tips()


def clipq(api, q):
    lim = api.joint_limits
    return np.clip(q, lim[:, 0], lim[:, 1])


def solve(api, targets, q0, iters=10, tol=0.0015, fd=0.08, lam=0.01):
    """In-sim IK: joint targets putting finger pads at palm-frame `targets` {i: xyz}."""
    lim = api.joint_limits
    fingers = list(targets)
    q = clipq(api, np.asarray(q0, float).copy())
    T = yield from probe(api, q)
    J = {f: np.zeros((3, 4)) for f in fingers}
    for j in range(4):
        q2 = q.copy()
        for f in fingers:
            k = 4 * f + j
            q2[k] += fd if q[k] + fd <= lim[k, 1] else -fd
        q2 = clipq(api, q2)
        T2 = yield from probe(api, q2)
        for f in fingers:
            k = 4 * f + j
            dq = q2[k] - q[k]
            J[f][:, j] = (T2[f] - T[f]) / dq if abs(dq) > 1e-6 else 0.0
    for it in range(iters):
        errs = {f: np.asarray(targets[f]) - T[f] for f in fingers}
        if max(np.linalg.norm(e) for e in errs.values()) < tol:
            break
        qn = q.copy()
        for f in fingers:
            e = errs[f]
            if np.linalg.norm(e) < tol * 0.5:
                continue
            sl = slice(4 * f, 4 * f + 4)
            mask = np.ones(4)
            for _ in range(2):
                Jm = J[f] * mask
                dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam ** 2 * np.eye(3), e)
                lo = (q[sl] <= lim[sl, 0] + 1e-4) & (dq < 0)
                hi = (q[sl] >= lim[sl, 1] - 1e-4) & (dq > 0)
                if not (lo | hi).any():
                    break
                mask = mask * ~(lo | hi)
            qn[sl] += np.clip(dq, -0.3, 0.3)
        qn = clipq(api, qn)
        Tn = yield from probe(api, qn)
        for f in fingers:
            sl = slice(4 * f, 4 * f + 4)
            dq = qn[sl] - q[sl]
            if dq @ dq > 1e-8:
                J[f] += np.outer(Tn[f] - T[f] - J[f] @ dq, dq) / (dq @ dq)
        q, T = qn, Tn
    res = {f: float(np.linalg.norm(np.asarray(targets[f]) - T[f])) for f in fingers}
    return q, res, T


def pivot(api, R0, R1, piv, W0, W1, n=6, seconds=2.0, q=None):
    """Turn the palm from R0 to R1 (slerp) keeping palm-frame point `piv` on the
    world line W0 -> W1, in n arm waypoints."""
    sl = Slerp([0, 1], Rotation.from_matrix([R0, R1]))
    for k in range(1, n + 1):
        s = k / n
        R = sl(s).as_matrix()
        W = (1 - s) * np.asarray(W0) + s * np.asarray(W1)
        yield from api.move(W - R @ piv, R, q=q if k == 1 else None, seconds=seconds / n)


def try_move(api, xyz, R, seconds, q=None):
    """api.move that survives an unreachable arm pose: on IK failure it re-anchors
    the command at the measured palm and returns False."""
    try:
        yield from api.move(xyz, R, q=q, seconds=seconds)
        return True
    except ValueError as e:
        d(f"IK fail -> recover: {str(e)[:60]}")
        Pm, Rm = api.measured_palm()
        yield from api.move(Pm, Rm, seconds=0.02)
        yield from api.hold(0.3)
        return False


def forces(api):
    return np.asarray(api.contact_forces("cube"), float)


def state(api, tag, p0):
    p, _ = api.object_pose("cube")
    rp, _ = api.object_in_palm("cube")
    d(f"{tag} t={api.t:.1f} dz={(p[2]-p0[2])*1000:.0f} f={r2(forces(api))} rel={mm(rp)}")


# Thumb FK samples measured in iteration 2 (api.tips, palm down, in the air): (q[0:4], pad mm).
THUMB_SAMPLES = [
    ([-1.04, -1.9, 0.76, 0.21], [109, 51, 79]), ([0.83, -1.0, 0.46, 0.8], [98, 49, 94]),
    ([-0.94, -1.77, -0.02, 0.88], [123, 30, 61]), ([-0.05, -1.26, 0.69, -0.27], [122, 50, 64]),
    ([1.12, -0.6, 0.61, 0.79], [70, -6, 108]), ([0.04, -2.06, -0.14, 0.94], [115, -27, 50]),
    ([0.38, -0.92, 0.46, 1.18], [80, 35, 93]), ([-0.11, -1.34, 0.42, 1.57], [80, 36, 76]),
    ([-0.76, -2.67, 0.96, 1.57], [38, 37, 73]),
    # iteration 3
    ([-0.55, -1.57, -0.37, 1.57], [111, 30, 50]), ([0.76, -1.61, -1.2, 1.57], [113, 27, 15]),
]
Q_WALL = np.array([0.0, 1.45, 0.25, 0.15])     # index/middle/ring: bent down as a wall
Q_LITTLE = np.array([0.2, 1.45, 0.25, 0.15])   # little (own joint order; j2 flexes it)


def thumb_seed(target):
    best = min(THUMB_SAMPLES, key=lambda s: np.linalg.norm(np.array(s[1]) / 1000 - target))
    return np.array(best[0], float)


def ik_thumb(api, base, target, seed=None, iters=10):
    q = base.copy()
    q[0:4] = thumb_seed(target) if seed is None else seed
    q, res, T = yield from solve(api, {0: np.asarray(target, float)}, q, iters=iters)
    return q, res[0], T[0]


# ----------------------------------------------------------------- policy
def run(api):
    try:
        yield from main(api)
    finally:
        for line in DIAG[-60:]:
            api.note(line)


def on_palm_ok(api, p0):
    f = forces(api)
    rp, _ = api.object_in_palm("cube")
    p, _ = api.object_pose("cube")
    _, Rm = api.measured_palm()
    ok = (Rm[2, 0] > 0.7 and f[5] > 0.02 and f[:5].sum() < 0.5 and rp[0] < 0.06
          and abs(rp[1]) < 0.045 and p[2] - p0[2] > 0.1)
    return ok, f, rp, Rm[2, 0]


def main(api):
    p0, R0 = api.object_pose("cube")
    h = p0[2] - TABLE
    vert = int(np.argmax(np.abs(R0[2])))
    a = R0[:, (vert + 1) % 3]
    ya = np.arctan2(a[1], a[0])
    yaw = 0.7 + ((ya - 0.7 + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    Rd = api.hand_down(yaw)
    Fd = Rd @ Ry(np.pi / 2)            # fingers down, palm facing the robot
    Fd2 = Rz(np.pi) @ Fd               # fingers down, palm facing away
    d(f"start p0={mm(p0)} h={h*1000:.1f} yaw={yaw:.3f}")

    # ---- calibration in the air: wall posture, then thumb IK
    q_open = api.OPEN.copy()
    base = q_open.copy()
    base[4:8] = Q_WALL
    base[8:12] = Q_WALL
    base[12:16] = Q_WALL
    base[16:20] = Q_LITTLE
    base = clipq(api, base)
    T = yield from probe(api, base, t=0.6, settle=0.3)
    d(f"wall tips={mm(T)}")
    PI = T[1].copy()
    xp = float(max(T[1:5, 0]))         # deepest wall pad (palm x)
    zp = float(max(T[1:5, 2]))         # outermost wall pad (palm z)
    zf = PI[2] - PAD                   # cube's fingertip-side face, palm z
    zc = zf - h                        # cube centre, palm z
    yc = PI[1]
    x_hi = xp - 0.003                  # cube centre at pick (palm x)
    tz = zc - h - PAD                  # thumb pad z touching the wrist-side face

    q_g, r_g, Tg = yield from ik_thumb(api, base, [x_hi, yc, tz + 0.003])
    d(f"IK thumb grasp q={r2(q_g[:4])} res={r_g*1000:.1f} pad={mm(Tg)}")
    q_p, r_p, Tp = yield from ik_thumb(api, base, [x_hi, yc, tz - 0.03], seed=q_g[:4], iters=6)
    d(f"IK thumb pre q={r2(q_p[:4])} res={r_p*1000:.1f} pad={mm(Tp)} t={api.t:.1f}")
    q_rel = q_p.copy()
    q_rel[0:4] = q_open[0:4]           # thumb fully open, walls kept

    # ---- pick: descend beside the cube, slide the wall onto its face, squeeze with the thumb
    c_pick = np.array([x_hi, yc, zc])
    xyz_g = p0 - Rd @ c_pick
    shift = Rd @ np.array([0, 0, 0.012])
    yield from api.move(xyz_g + shift + [0, 0, 0.08], Rd, q=q_p, seconds=2.0)
    yield from api.move(xyz_g + shift, Rd, seconds=1.2)
    yield from api.hold(0.2)
    state(api, "descended", p0)
    yield from api.move(xyz_g + Rd @ np.array([0, 0, 0.002]), Rd, seconds=0.5)
    yield from api.move(q=q_g, seconds=0.6)
    yield from api.hold(0.3)
    state(api, "grasp", p0)
    P, _ = api.palm_pose()
    PL = P + [0, 0, 0.20]
    yield from api.move(PL, Rd, seconds=1.4)
    yield from api.hold(0.2)
    state(api, "lifted", p0)

    # ---- turn over: one pitch about palm y (fingers swing down, then back
    # toward the robot) so palm z, the pinch axis, never turns about itself
    def Rp(psi):
        return Rd @ Ry(psi)

    yield from api.move(PL, Rp(np.pi / 2), seconds=2.0)
    yield from api.hold(0.2)
    state(api, "fingers-down", p0)
    psi_rel = np.pi / 2 + np.radians(30)
    PC = PL
    ok = yield from try_move(api, PC, Rp(psi_rel), 1.0)
    if not ok:                                   # reach: carry closer to the robot base
        away = np.array([PL[0] - 0.30, PL[1] + 0.58, 0.0])
        PC = PL - 0.10 * away / np.linalg.norm(away)
        yield from api.move(PC, Rp(np.pi / 2), seconds=1.2)
        ok = yield from try_move(api, PC, Rp(psi_rel), 1.0)
    yield from api.hold(0.2)
    state(api, f"tilted ok={ok}", p0)
    # let go with the thumb: the cube rests on the (tilted) shelf
    yield from api.move(q=q_p, seconds=0.4)
    yield from api.move(q=q_rel, seconds=0.5)
    yield from api.hold(0.2)
    state(api, "thumb open", p0)
    ok = yield from try_move(api, PC, Rp(np.pi - TILT), 2.0)
    if not ok:
        ok = yield from try_move(api, PC, Rp(np.pi - TILT - 0.15), 1.0)
    d(f"cradle ok={ok} PC={mm(PC)}")

    # ---- palm carries the cube
    on, k, flat = 0, 0, False
    while k < 45:
        yield from api.hold(0.1)
        k += 1
        okp, f, rp, up = on_palm_ok(api, p0)
        on += okp
        if k % 5 == 0:
            d(f"carry k={k} ok={okp} up={up:.2f} f={r2(f)} rel={mm(rp)}")
        if on >= 16:
            break
        if k == 12 and on < 3 and not flat:     # stuck on the wall: lay the palm flat
            flat = True
            yield from try_move(api, PC, Rp(np.pi - 0.12), 0.8)
    d(f"carry on={on} (x0.1s) t={api.t:.1f}")

    # ---- back: pitch to fingers-down (cube on the shelf against the palm)
    yield from api.move(PC, Rp(np.pi / 2), seconds=1.8)
    yield from api.hold(0.2)
    state(api, "shelf", p0)

    # ---- set down: lower the shelf near the table, pitch palm-down about the pads
    delta = 0.008
    piv = np.array([xp, yc, zp])
    u = p0[:2] - np.array([0.30, -0.58])          # robot base -> cube, horizontal
    u = np.r_[u / np.linalg.norm(u), 0.0]
    for shift in (0.03, 0.06, 0.09):              # palm sits ~0.1 m past the cube here: reach
        Pfin = p0 + [0, 0, delta] - Rd @ c_pick - shift * u   # palm at the end (pick pose)
        Wend = Pfin + Rd @ piv
        Wstart = Wend + [0, 0, 0.015]
        P_sd = Wstart - Fd @ piv
        ok = yield from try_move(api, P_sd, Fd, 2.5)
        if ok:
            break
    d(f"set-down shift={shift*1000:.0f}mm ok={ok}")
    yield from api.hold(0.2)
    state(api, "over table", p0)
    yield from pivot(api, Fd, Rd, piv, Wstart, Wend, n=8, seconds=4.0)
    yield from api.hold(0.4)
    state(api, "set down", p0)
    P, _ = api.palm_pose()
    P = P + Rd @ np.array([0, 0, 0.006])
    yield from api.move(P, Rd, seconds=0.4)
    yield from api.move(P + [0, 0, 0.10], Rd, seconds=1.0)
    yield from api.move(q=q_open, seconds=0.5)
    yield from api.hold(0.3)
    p, R = api.object_pose("cube")
    d(f"end p-p0={mm(p - p0)} flat={np.abs(R[2]).max():.4f} f={r2(forces(api))} t={api.t:.1f}")
