import numpy as np

TABLE = 0.75
BLOCK_TOP = TABLE + 0.060
PSI = 0.7
C = np.array([0.080, 0.040, 0.118])   # nominal thumb-index pinch centre, palm frame
D_OPEN = 0.026                        # pad-centre half-separation, open
D_CLOSE = 0.0095                      # pad-centre half-separation: ~1 mm into a nominal pen (pad surface ~6.5 mm from site)
F_GRIP = 0.8                          # target per-finger grip force [N]

LOG = []


def note(api, s):
    LOG.append(f"{api.t:.2f} {s}")
    api.note(s)


def R_side(psi=PSI):
    """Palm vertical, fingers horizontal along yaw psi, thumb pointing DOWN (palm +y = world -z).
    The pen (vertical) then lies along palm y and can slide up through a thumb-index pinch."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[s, 0, c], [-c, 0, s], [0, -1, 0.]])


# ---------------------------------------------------------------- joint-level helpers (skills)
def settle_tips(api, q, move_s, hold_s):
    yield from api.move(q=q, seconds=move_s)
    yield from api.hold(hold_s)
    return api.tips()


def jacobian(api, q, fingers, h=0.08):
    """Finite-difference pad Jacobian (palm frame, 3x4 per finger), probing all named fingers at once."""
    lim = api.joint_limits
    T0 = yield from settle_tips(api, q, 0.08, 0.12)
    J = {i: np.zeros((3, 4)) for i in fingers}
    for j in range(4):
        qp = q.copy()
        st = {}
        for i in fingers:
            k = 4 * i + j
            hh = h if q[k] + h <= lim[k, 1] else -h
            qp[k] = q[k] + hh
            st[i] = hh
        T = yield from settle_tips(api, qp, 0.06, 0.12)
        for i in fingers:
            J[i][:, j] = (T[i] - T0[i]) / st[i]
    yield from api.move(q=q, seconds=0.06)
    yield from api.hold(0.08)
    return J, T0


def tips_ik(api, targets, q, iters=10, tol=0.0008, refresh=2, free_y=()):
    """Measured-tip Gauss-Newton IK: drive pads to palm-frame targets {finger: xyz}.
    Fingers in free_y ignore the palm-y error (only x, z matter)."""
    lim = api.joint_limits
    q = q.copy()
    fingers = list(targets)
    J = None
    yield from api.move(q=q, seconds=0.3)
    yield from api.hold(0.1)
    T = api.tips()
    for it in range(iters):
        if J is None or it % refresh == 0:
            J, T = yield from jacobian(api, q, fingers)
        err = {i: np.asarray(targets[i]) - T[i] for i in fingers}
        for i in free_y:
            err[i][1] = 0.0
        emax = max(np.linalg.norm(e) for e in err.values())
        if emax < tol:
            break
        for i in fingers:
            Ji = J[i]
            dq = Ji.T @ np.linalg.solve(Ji @ Ji.T + 2e-5 * np.eye(3), err[i])
            n = np.abs(dq).max()
            if n > 0.35:
                dq *= 0.35 / n
            sl = slice(4 * i, 4 * i + 4)
            q[sl] = np.clip(q[sl] + dq, lim[sl, 0], lim[sl, 1])
        T = yield from settle_tips(api, q, 0.15, 0.1)
    err = {}
    for i in fingers:
        e = np.asarray(targets[i]) - T[i]
        if i in free_y:
            e[1] = 0.0
        err[i] = float(np.linalg.norm(e))
    return q, J, T, err


def pad_step(J, d):
    """Joint step moving a pad by palm-frame d (damped least squares)."""
    return J.T @ np.linalg.solve(J @ J.T + 2e-5 * np.eye(3), d)


def pen_geom(api):
    p, R = api.object_pose("pen")
    a = R[:, 2] if R[2, 2] >= 0 else -R[:, 2]
    return p, a


def pen_axis_at_y(api, y):
    """Palm frame: the point of the pen axis at palm-y level y, and the pen's upper end."""
    pp, Rp = api.object_in_palm("pen")
    a = Rp[:, 2]
    _, Rw = api.object_pose("pen")
    if Rw[2, 2] < 0:
        a = -a
    top = pp + HALF[0] * a
    pt = pp + a * (y - pp[1]) / (a[1] if abs(a[1]) > 1e-6 else 1e-6)
    return pt, top


def pen_points(api, mid):
    """Palm frame: the pen axis point at the pinch's palm-y level, and the pen's lower end."""
    pp, Rp = api.object_in_palm("pen")
    a = Rp[:, 2]
    p, Rw = api.object_pose("pen")
    aw = Rw[:, 2]
    if aw[2] < 0:
        a = -a
    # a points (world) up; lower end = centre - half * a
    h = HALF[0]
    tip = pp - h * a
    pk = pp + a * (mid[1] - pp[1]) / (a[1] if abs(a[1]) > 1e-6 else 1e-6)
    return pk, tip


HALF = [0.07]


def run(api):
    try:
        yield from main(api)
    finally:
        for s in LOG[-25:]:
            api.note("SUMMARY " + s)


def main(api):
    lim = api.joint_limits
    R = R_side()
    p0, a0 = pen_geom(api)
    half = p0[2] - TABLE - 0.0005          # centre-to-end length
    top = p0[2] + half
    HALF[0] = half
    note(api, f"pen centre {np.round(p0, 4).tolist()} half {half:.4f} top {top:.4f}")

    # 1. rotate to thumb-down side posture in the air; middle/ring/little straight, out of the way
    q = api.OPEN.copy()
    for i in (2, 3, 4):
        q[4 * i:4 * i + 4] = 0.0
    q = np.clip(q, lim[:, 0], lim[:, 1])
    x_start, _ = api.palm_pose()
    yield from api.move(xyz=x_start + np.array([0.0, 0.05, -0.03]), R=R, q=q, seconds=1.5)

    # 2. joint-level IK for the open and closed pinch from measured pads
    tgt_open = {0: C - np.array([0, 0, D_OPEN]), 1: C + np.array([0, 0, D_OPEN])}
    q_open, _, T, err = yield from tips_ik(api, tgt_open, q, iters=12)
    note(api, f"open IK err mm {[round(1000 * e, 1) for e in err.values()]} q_th {np.round(q_open[0:4], 2).tolist()} q_ix {np.round(q_open[4:8], 2).tolist()}")
    tgt_close = {0: C - np.array([0, 0, D_CLOSE]), 1: C + np.array([0, 0, D_CLOSE])}
    q_close, J, T, err = yield from tips_ik(api, tgt_close, q_open, iters=8)
    note(api, f"close IK err mm {[round(1000 * e, 1) for e in err.values()]} q_th {np.round(q_close[0:4], 2).tolist()} q_ix {np.round(q_close[4:8], 2).tolist()}")
    J, Tc = yield from jacobian(api, q_close, [0, 1])
    mid = 0.5 * (Tc[0] + Tc[1])
    note(api, f"closed pads {np.round(Tc[0], 4).tolist()} {np.round(Tc[1], 4).tolist()} mid {np.round(mid, 4).tolist()}")

    # 3. above the pen with the pinch open
    z_g = top - 0.020
    D = float(np.clip(z_g - (BLOCK_TOP + 0.022), 0.018, 0.030))
    pen_xy = p0[:2]
    above = np.array([pen_xy[0], pen_xy[1], top + 0.035])
    yield from api.move(xyz=above - R @ mid, R=R, q=q_open, seconds=2.5)
    yield from api.hold(0.2)
    # 4. lower around the pen
    yield from api.move(xyz=np.array([pen_xy[0], pen_xy[1], z_g]) - R @ mid, seconds=1.2)
    yield from api.hold(0.3)
    # align the pen axis with the pinch midpoint (palm x, z)
    for _ in range(2):
        pp, _ = api.object_in_palm("pen")
        e = np.array([pp[0] - mid[0], 0.0, pp[2] - mid[2]])
        xyz, _ = api.palm_pose()
        yield from api.move(xyz=xyz + R @ e, seconds=0.3)
        yield from api.hold(0.15)
    pp, _ = api.object_in_palm("pen")
    note(api, f"pen in palm before close {np.round(pp, 4).tolist()} f {np.round(api.contact_forces('pen'), 2).tolist()}")

    # 5. close both pads together (joint interpolation to the IK'd closed posture), then squeeze equally.
    #    (A force-guarded one-finger-at-a-time close does not work: the free-standing pen is pushed
    #    with ~0 force until it hits the bore wall.)  Any resulting lean is removed by verticalize().
    yield from api.move(q=q_close, seconds=0.6)
    yield from api.hold(0.2)
    qg = q_close.copy()
    pk, tk = pen_points(api, mid)
    note(api, f"closed f {np.round(api.contact_forces('pen')[:2], 2).tolist()} pinch-pt {np.round(1000 * (pk - mid), 1).tolist()} tip-off {np.round(1000 * (tk - pk), 1).tolist()}")

    def squeeze(qg, ftarget, steps, dz=0.0006):
        for _ in range(steps):
            f = api.contact_forces("pen")
            need = [i for i in (0, 1) if f[i] < ftarget]
            if not need:
                break
            for i in need:
                d = np.array([0, 0, dz if i == 0 else -dz])
                sl = slice(4 * i, 4 * i + 4)
                qg[sl] = np.clip(qg[sl] + pad_step(J[i], d), lim[sl, 0], lim[sl, 1])
            yield from api.move(q=qg, seconds=0.12)
            yield from api.hold(0.12)
        return qg

    qg = yield from squeeze(qg, F_GRIP, 8)
    pk, tk = pen_points(api, mid)
    note(api, f"squeezed f {np.round(api.contact_forces('pen')[:2], 2).tolist()} pinch-pt {np.round(1000 * (pk - mid), 1).tolist()} tip-off {np.round(1000 * (tk - pk), 1).tolist()}")

    def verticalize(qg, xyz, gain=1.0, maxstep=0.003, iters=4, tol=0.0012, log=None):
        """Move the palm (palm x,z) so the pinch sits straight above the pen's lower end."""
        for _ in range(iters):
            pk, tk = pen_points(api, mid)
            e = (tk - pk) * gain
            e[1] = 0.0
            n = np.linalg.norm(e)
            if log is not None:
                log.append(round(1000 * float(n), 1))
            if n < tol:
                break
            if n > maxstep:
                e *= maxstep / n
            xyz = xyz + R @ e
            yield from api.move(xyz=xyz, seconds=0.25)
            yield from api.hold(0.05)
            f = api.contact_forces("pen")
            if f[0] < 0.4 or f[1] < 0.4:
                qg = yield from squeeze(qg, 0.6, 2)
        return qg, xyz

    xyz, _ = api.palm_pose()
    vlog = []
    qg, xyz = yield from verticalize(qg, xyz, iters=8, log=vlog)
    note(api, f"grasp verticalize offsets mm {vlog}")
    p, a = pen_geom(api)
    tip_h = p[2] - half * a[2] - TABLE
    th = api.tips_world()[0]
    note(api, f"grasped f {np.round(api.contact_forces('pen'), 2).tolist()} tip_h {1000 * tip_h:.1f}mm D {1000 * D:.0f} tilt {np.degrees(np.arccos(min(1, abs(a[2])))):.1f} pen-thumb axial {1000 * float((p - th) @ a):.1f}mm")

    def regrasp(qg, xyz, tag):
        """Release the pen in the bore (it stands on its own), re-centre the open pinch on the pen axis at
        the pinch level, and close again to the calibrated posture. Resets the slow sideways creep of the
        pen out of the two-pad pinch and any pinch-vs-pen torque, so the grasp starts symmetric again."""
        f0 = api.contact_forces("pen")[:2]
        q_rel = q_close.copy()
        for i, sgn in ((0, -1.0), (1, 1.0)):
            sl = slice(4 * i, 4 * i + 4)
            q_rel[sl] = np.clip(q_rel[sl] + pad_step(J[i], np.array([0.0, 0.0, sgn * 0.0035])), lim[sl, 0], lim[sl, 1])
        yield from api.move(q=q_rel, seconds=0.3)
        yield from api.hold(0.25)
        al = []
        for _ in range(3):
            pk, _ = pen_points(api, mid)
            e = np.array([pk[0] - mid[0], 0.0, pk[2] - mid[2]])
            al.append(round(1000 * float(np.linalg.norm(e)), 1))
            if np.linalg.norm(e) < 0.0006:
                break
            nrm = np.linalg.norm(e)
            if nrm > 0.012:
                e *= 0.012 / nrm
            xyz = xyz + R @ e
            yield from api.move(xyz=xyz, seconds=0.3)
            yield from api.hold(0.15)
        yield from api.move(q=q_close, seconds=0.5)
        yield from api.hold(0.2)
        qg = q_close.copy()
        qg = yield from squeeze(qg, F_GRIP, 6)
        pk, tk = pen_points(api, mid)
        p, a = pen_geom(api)
        note(api, f"regrasp[{tag}] f before {np.round(f0, 2).tolist()} align mm {al} after {np.round(api.contact_forces('pen')[:2], 2).tolist()} "
                  f"pinch-pt {np.round(1000 * (pk - mid), 1).tolist()} tilt {np.degrees(np.arccos(min(1, abs(a[2])))):.1f}")
        return qg, xyz

    # 6. slide: lower the hand in small chunks, the table pushes the pen up through the pinch
    n = 8
    slog = []
    regrasped = False
    for k in range(n):
        xyz = xyz + np.array([0, 0, -D / n])
        yield from api.move(xyz=xyz, seconds=0.3)
        f = api.contact_forces("pen")
        if f[0] < 0.4 or f[1] < 0.4:
            qg = yield from squeeze(qg, 0.6, 2)
        f = api.contact_forces("pen")
        slog.append((round(float(f[0]), 1), round(float(f[1]), 1)))
        qg, xyz = yield from verticalize(qg, xyz, gain=0.7, maxstep=0.0015, iters=1, tol=0.003, log=slog)
        if 2 <= k <= 5 and not regrasped:
            pk, tk = pen_points(api, mid)
            off = float(np.linalg.norm((tk - pk)[[0, 2]]))
            f = api.contact_forces("pen")
            if off > 0.003 or min(f[0], f[1]) < 0.5:
                regrasped = True
                slog.append("RG")
                qg, xyz = yield from regrasp(qg, xyz, f"slide k={k} off={1000 * off:.1f}")
    note(api, f"slide (fth, fix, offmm) {slog}")
    yield from api.hold(0.2)
    p, a = pen_geom(api)
    tip_h = p[2] - half * a[2] - TABLE
    th = api.tips_world()[0]
    note(api, f"after slide f {np.round(api.contact_forces('pen'), 2).tolist()} tip_h {1000 * tip_h:.1f}mm pen-thumb axial {1000 * float((p - th) @ a):.1f}mm tilt {np.degrees(np.arccos(min(1, abs(a[2])))):.1f}")

    # 7. fresh, torque-free grasp on the free-standing pen, then lift straight up (no palm
    #    verticalizing: a lean of <= 6 deg in the bore clears the wall when lifted vertically)
    qg, xyz = yield from regrasp(qg, xyz, "pre-lift")
    qg = yield from squeeze(qg, 1.0, 3)
    hist = []
    for k in range(14):
        xyz = xyz + np.array([0, 0, 0.01])
        yield from api.move(xyz=xyz, seconds=0.22)
        p, a = pen_geom(api)
        f = api.contact_forces("pen")
        tl = np.degrees(np.arccos(min(1, abs(a[2]))))
        hist.append((round(1000 * (p[2] - half * a[2] - TABLE)), round(float(tl), 1), round(float(f[0]), 1), round(float(f[1]), 1)))
        if f[0] < 0.5 or f[1:5].max() < 0.5:
            qg = yield from squeeze(qg, 0.9, 2)
    note(api, f"lift (tip_mm, tilt, fth, fix) {hist}")
    # 8. final hold: top up the grip so it survives the bench's 1 s hold
    flog = []
    for _ in range(4):
        yield from api.hold(0.2)
        f = api.contact_forces("pen")
        flog.append((round(float(f[0]), 2), round(float(f[1]), 2)))
        if f[0] < 1.0 or f[1:5].max() < 1.0:
            qg = yield from squeeze(qg, 1.2, 2)
    p, a = pen_geom(api)
    tip_h = p[2] - half * a[2] - TABLE
    note(api, f"lifted f {np.round(api.contact_forces('pen'), 2).tolist()} tip_h {1000 * tip_h:.1f}mm tilt {np.degrees(np.arccos(min(1, abs(a[2])))):.1f} hold f {flog}")
