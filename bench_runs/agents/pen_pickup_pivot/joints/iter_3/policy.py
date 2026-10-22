"""pen_pickup_pivot, joints level.

Plan: thumb-index pinch across the pen near one end (pinch line horizontal, along the
finger direction, perpendicular to the pen). The pen has no torsional friction, so the
pinch is a free hinge. Lift the hand slowly straight up: the far end drags on the table
and the pen swings up to vertical while still standing on that end, then hangs from the
hinge. Level the hinge from the measured pen tilt, touch the pen down on the table, lift
it clear, level again and hold.

Joint-level skills built here (no fingertip IK at this level):
  fd_jacobian  measured finite-difference Jacobian of pad centre vs the 4 joints of a finger
  servo_tips   closed-loop pad placement in free air: damped least squares + Broyden updates
"""
import numpy as np
from scipy.spatial.transform import Rotation

TABLE = 0.75
PINCH = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (hand.py doc)
S_GRIP = 0.052                             # grip this far from the pen centre toward one end
HALF_NOM = 0.0662
PAD_LOW = 0.008                            # stop descending when a pad centre is this high
NOTES = []


def note(text):
    NOTES.append(text)
    print("NOTE", text, flush=True)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def settle(api, q, t=0.25, hold=0.1):
    yield from api.move(q=q, seconds=t)
    yield from api.hold(hold)


def fd_jacobian(api, q, fingers, delta=0.12):
    lim = api.joint_limits
    yield from settle(api, q)
    base = api.tips()
    J = {f: np.zeros((3, 4)) for f in fingers}
    for j in range(4):
        qq = q.copy()
        d = {}
        for f in fingers:
            k = 4 * f + j
            dd = delta if q[k] + delta <= lim[k, 1] - 0.01 else -delta
            qq[k] += dd
            d[f] = qq[k] - q[k]
        yield from settle(api, qq)
        t = api.tips()
        for f in fingers:
            J[f][:, j] = (t[f] - base[f]) / d[f]
    yield from settle(api, q)
    return J


def servo_tips(api, q, J, targets, iters=8, tol=0.0012, lam=0.012, max_step=0.35):
    lim = api.joint_limits
    q = q.copy()
    for _ in range(iters):
        t = api.tips()
        err = {f: targets[f] - t[f] for f in targets}
        if max(np.linalg.norm(e) for e in err.values()) < tol:
            break
        qn = q.copy()
        for f, e in err.items():
            A = J[f]
            dq = A.T @ np.linalg.solve(A @ A.T + lam ** 2 * np.eye(3), e)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            sl = slice(4 * f, 4 * f + 4)
            qn[sl] = np.clip(q[sl] + dq, lim[sl, 0], lim[sl, 1])
        yield from settle(api, qn)
        t2 = api.tips()
        for f in targets:
            sl = slice(4 * f, 4 * f + 4)
            dqf = qn[sl] - q[sl]
            if dqf @ dqf > 1e-6:
                J[f] += np.outer((t2[f] - t[f]) - J[f] @ dqf, dqf) / (dqf @ dqf)
        q = qn
    t = api.tips()
    res = {f: float(np.linalg.norm(targets[f] - t[f])) for f in targets}
    return q, res


def pen_axis_down(api):
    p, R = api.object_pose("pen")
    a = R[:, 2]
    if a[2] > 0:
        a = -a
    return p, a


def tilt_deg(a):
    return float(np.degrees(np.arccos(min(1.0, abs(a[2])))))


def level_hinge(api, rounds=2, settle_s=0.6):
    """Rotate the palm about the pinch point so the hanging pen points straight down
    (rotation component about the hinge line removed: the pen turns freely about it)."""
    for _ in range(rounds):
        p, a = pen_axis_down(api)
        tl = tilt_deg(a)
        if tl < 1.5 or tl > 40:
            break
        k = np.cross(a, [0, 0, -1.0])
        k /= np.linalg.norm(k) + 1e-12
        w = np.radians(tl) * k
        tw = api.tips_world()
        h = tw[1] - tw[0]
        h /= np.linalg.norm(h)
        w = w - (w @ h) * h
        if np.linalg.norm(w) < np.radians(1.0):
            yield from api.hold(0.4)
            continue
        Rw = Rotation.from_rotvec(w).as_matrix()
        P, R = api.palm_pose()
        G = 0.5 * (tw[0] + tw[1])
        note(f"level: tilt {tl:.1f} deg, correction {np.degrees(np.linalg.norm(w)):.1f} deg")
        yield from api.move(xyz=G + Rw @ (P - G), R=Rw @ R, seconds=0.8)
        yield from api.hold(settle_s)


def run(api):
    p0, R0 = api.object_pose("pen")
    r_est = float(p0[2] - TABLE)
    a0 = R0[:, 2].copy()
    a0[2] = 0
    a0 /= np.linalg.norm(a0)
    # palm y (toward thumb) along the pen axis: hinge (palm z) horizontal and across the pen
    y1 = np.arctan2(-a0[0], a0[1])
    yaw = min([y1, y1 + np.pi], key=lambda y: abs(_wrap(y - api.DEFAULT_YAW)))
    yaw = api.DEFAULT_YAW + _wrap(yaw - api.DEFAULT_YAW)
    R = api.hand_down(yaw)
    grip = p0 + S_GRIP * R[:, 1]          # toward the thumb-side end; long part under the fingers
    grip[2] = p0[2]
    note(f"pen p0={np.round(p0, 4).tolist()} r_est={r_est * 1000:.2f}mm yaw={yaw:.3f}")

    q = api.OPEN.copy()
    for f in (2, 3, 4):
        q[4 * f + 1:4 * f + 4] = 0.1
    q[4:8] = [0.0, 0.7, 0.6, 0.5]           # index seed, half flexed
    q[0:4] = [0.0, -0.3, 0.4, 0.4]          # thumb seed
    high = api.palm_for(grip + [0, 0, 0.10], PINCH, R)
    yield from api.move(xyz=high, R=R, q=q, seconds=1.8)
    yield from api.hold(0.2)

    # --- calibrate and place the pads (free air)
    J = yield from fd_jacobian(api, q, (0, 1))
    tgt_open = {0: PINCH + [0, 0, -0.025], 1: PINCH + [0, 0, 0.025]}
    q50, res = yield from servo_tips(api, q, J, tgt_open, iters=10)
    note(f"servo open residual mm {[round(v * 1000, 1) for v in res.values()]} q={np.round(q50[:8], 2).tolist()}")
    if max(res.values()) > 0.003:
        J = yield from fd_jacobian(api, q50, (0, 1))
        q50, res = yield from servo_tips(api, q50, J, tgt_open, iters=8)
        note(f"servo open (2) residual mm {[round(v * 1000, 1) for v in res.values()]}")
    tgt_28 = {0: PINCH + [0, 0, -0.014], 1: PINCH + [0, 0, 0.014]}
    q28, res = yield from servo_tips(api, q50, J, tgt_28, iters=8)
    note(f"servo 28 residual mm {[round(v * 1000, 1) for v in res.values()]} q={np.round(q28[:8], 2).tolist()}")
    lim = api.joint_limits
    yield from settle(api, q50, t=0.4)
    t50 = api.tips()
    mid = 0.5 * (t50[0] + t50[1])
    note(f"open pads palm-frame thumb={np.round(t50[0], 4).tolist()} index={np.round(t50[1], 4).tolist()}")

    def closed(sep, dx):
        """Extrapolate from q28 (pads 28 mm apart) to pads `sep` apart and `dx` lower (palm x)."""
        qc = q28.copy()
        for f, sgn in ((0, 1), (1, -1)):
            d = np.array([dx, 0, sgn * (0.014 - sep / 2)])
            A = J[f]
            dq = A.T @ np.linalg.solve(A @ A.T + 0.012 ** 2 * np.eye(3), d)
            sl = slice(4 * f, 4 * f + 4)
            qc[sl] = np.clip(q28[sl] + dq, lim[sl, 0], lim[sl, 1])
        return qc

    grasped = False
    for attempt in range(3):
        press = 0.002 + 0.0015 * attempt
        sep = 0.006 - 0.002 * attempt
        # --- above the grip point, pads open, then descend until a pad touches the table
        p_now, _ = api.object_pose("pen")
        grip = p_now + S_GRIP * R[:, 1]
        grip[2] = p_now[2]
        start = api.palm_for(grip + [0, 0, 0.035], mid, R)
        yield from api.move(xyz=start, R=R, q=q50, seconds=1.0)
        yield from api.hold(0.2)
        ref = api.tips()
        P = start.copy()
        v = 0.0
        dev = 0.0
        for _ in range(400):
            tw = api.tips_world()
            if min(tw[0][2], tw[1][2]) - TABLE < 0.004:
                break
            v = min(0.0003, v + 0.00003)
            P[2] -= v
            yield from api.move(xyz=P, R=R, seconds=0.02)
            t = api.tips()
            dev = max(np.linalg.norm(t[f] - ref[f]) for f in (0, 1))
            if dev > 0.0015:
                break
        P[2] -= press
        yield from api.move(xyz=P, R=R, seconds=0.2)
        tw = api.tips_world()
        note(f"t={api.t:.2f} try {attempt}: descent pads z thumb {1000 * (tw[0][2] - TABLE):.1f} index {1000 * (tw[1][2] - TABLE):.1f} mm, "
             f"pen in palm {np.round(api.object_in_palm('pen')[0] * 1000, 1).tolist()}")
        # --- pinch: pads aimed lower than the table allows, so they stay pressed down
        qc = closed(sep, 0.004)
        yield from api.move(q=q28, seconds=0.4)
        yield from api.move(q=qc, seconds=0.5)
        yield from api.hold(0.4)
        f = api.contact_forces("pen")
        tw = api.tips_world()
        note(f"t={api.t:.2f} pinch: forces {np.round(f, 2).tolist()} pad z {1000 * (tw[0][2] - TABLE):.1f},{1000 * (tw[1][2] - TABLE):.1f} "
             f"pads palm {np.round(api.tips()[:2] * 1000, 1).tolist()} pen palm {np.round(api.object_in_palm('pen')[0] * 1000, 1).tolist()}")
        # --- test lift: 15 mm slowly, the grip end must rise
        P0 = P.copy()
        yield from api.move(xyz=P + [0, 0, 0.015], R=R, seconds=1.5)
        p, a = pen_axis_down(api)
        f = api.contact_forces("pen")
        note(f"t={api.t:.2f} test lift: pen tilt {tilt_deg(a):.1f} forces {np.round(f, 2).tolist()}")
        P = P + [0, 0, 0.015]
        if tilt_deg(a) < 85 and f[0] > 0.05 and f[1] > 0.05:
            grasped = True
            break
        yield from api.move(q=q50, seconds=0.4)
    if not grasped:
        note("grasp failed")

    # --- slow straight lift: far end drags on the table, pen swings up to vertical
    G0 = R @ mid + P
    target_h = TABLE + 0.175
    halves, min_tilt_on = [], 90.0
    last_k = 0
    v = 0.0
    while (R @ mid + P)[2] < target_h:
        v = min(0.0004, v + 0.00002)
        P[2] += v
        yield from api.move(xyz=P, R=R, seconds=0.02)
        p, a = pen_axis_down(api)
        az = abs(a[2])
        k_lift = int(((R @ mid + P)[2] - G0[2]) / 0.025)
        if k_lift > last_k:
            last_k = k_lift
            note(f"lift +{k_lift * 25}mm: pen tilt {tilt_deg(a):.1f} pen z {1000 * (p[2] - TABLE):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")
        if 0.3 < az < 0.92:
            halves.append((p[2] - TABLE - r_est) / az)
        if halves:
            hh = float(np.median(halves))
            low = p[2] - hh * az - r_est - TABLE
            if low < 0.003:
                min_tilt_on = min(min_tilt_on, tilt_deg(a))
    half = float(np.median(halves)) if len(halves) > 5 else HALF_NOM
    note(f"lift done: half_est {half * 1000:.1f}mm from {len(halves)} samples, est min tilt on table {min_tilt_on:.1f}")
    yield from api.hold(0.8)
    p, a = pen_axis_down(api)
    note(f"hanging tilt {tilt_deg(a):.1f} forces {np.round(api.contact_forces('pen'), 2).tolist()}")
    yield from level_hinge(api, rounds=2)

    # --- touch the lower end down on the table, then lift clear again
    P, R = api.palm_pose()
    Pm, _ = api.measured_palm()
    p, a = pen_axis_down(api)
    low = p[2] - half * abs(a[2]) - r_est - TABLE
    d0 = p[2] - Pm[2]
    z_stop = P[2] - low - 0.004
    P_hang = P.copy()
    v = 0.0
    while P[2] > z_stop:
        v = min(0.0003, v + 0.00002)
        P[2] -= v
        yield from api.move(xyz=P, R=R, seconds=0.02)
        p, a = pen_axis_down(api)
        Pm, _ = api.measured_palm()
        if p[2] - Pm[2] - d0 > 0.0008:
            break
    note(f"touchdown: est low before {low * 1000:.1f}mm, lowered {(P_hang[2] - P[2]) * 1000:.1f}mm, tilt {tilt_deg(a):.1f}")
    yield from api.hold(0.2)
    yield from api.move(xyz=P_hang + [0, 0, 0.005], R=R, seconds=1.5)
    yield from api.hold(0.6)
    yield from level_hinge(api, rounds=2)
    p, a = pen_axis_down(api)
    note(f"final: tilt {tilt_deg(a):.1f} low_est {(p[2] - half * abs(a[2]) - r_est - TABLE) * 1000:.1f}mm")
    yield from api.hold(0.3)
