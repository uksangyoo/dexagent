import numpy as np

# Scissors, joints level.
# Grasp: index in hole A and ring in hole B squeeze the inner walls (closing pinch),
# lift so the scissors hang from the fingers, hook the distal joints, then the ring
# opens/closes the blades by abduction with hinge-angle feedback.
# Finger placement uses a home-made in-air fingertip IK: finite-difference
# Jacobians from api.tips() plus Broyden updates (no kinematics model).

TABLE = 0.75
PINCH = np.array([0.080, 0.040, 0.118])
CURL = np.array([0.0, 1.6, 1.3, 1.1])
HINGE = "scissors_hinge"
# nominal extreme points (same as the scorer) for a clearance estimate
_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]


def mm(a):
    return np.round(np.asarray(a, float) * 1000).astype(int).tolist()


def clearance(api):
    p, R = api.object_pose("scissors")
    a = api.object_joint(HINGE)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    pts = np.vstack([_A, _B @ Rz.T]) @ R.T + p
    return float(pts[:, 2].min() - abs(R[2, 2]) * 0.0055 - TABLE)


def settle(api, q, sec=0.3, hold=0.12):
    yield from api.move(q=q, seconds=sec)
    yield from api.hold(hold)
    return api.tips()


def ik(api, targets, q0, lo, hi, iters=7, tol=0.0015):
    """In-air fingertip IK: targets {finger index: palm-frame xyz}."""
    q = np.clip(q0, lo + 1e-3, hi - 1e-3)
    P = yield from settle(api, q, 0.5, 0.2)
    J = {i: np.zeros((3, 4)) for i in targets}
    for j in range(4):
        qp = q.copy()
        st = {}
        for i in targets:
            k = 4 * i + j
            d = 0.12 if q[k] + 0.12 < hi[k] else -0.12
            qp[k] += d
            st[i] = d
        Pp = yield from settle(api, qp, 0.2, 0.1)
        for i in targets:
            J[i][:, j] = (Pp[i] - P[i]) / st[i]
    P = yield from settle(api, q, 0.2, 0.1)
    for _ in range(iters):
        errs = {i: np.asarray(targets[i]) - P[i] for i in targets}
        if max(np.linalg.norm(e) for e in errs.values()) < tol:
            break
        qn = q.copy()
        for i, e in errs.items():
            A = J[i]
            dq = A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(3), e)
            n = np.linalg.norm(dq)
            if n > 0.5:
                dq *= 0.5 / n
            sl = slice(4 * i, 4 * i + 4)
            qn[sl] = np.clip(q[sl] + dq, lo[sl] + 1e-3, hi[sl] - 1e-3)
        Pn = yield from settle(api, qn, 0.3, 0.12)
        for i in errs:
            sl = slice(4 * i, 4 * i + 4)
            s = qn[sl] - q[sl]
            if s @ s > 1e-8:
                J[i] += np.outer((Pn[i] - P[i]) - J[i] @ s, s) / (s @ s)
        q, P = qn, Pn
    res = {i: float(np.linalg.norm(np.asarray(targets[i]) - P[i])) for i in targets}
    return q, P, J, res


def sweep(api, q_goal, sec, cond):
    g = api.move(q=q_goal, seconds=sec)
    for _ in g:
        yield
        if cond():
            g.close()
            return True
    return False


def run(api):
    out = []

    def N(s):
        out.append(s)
        api.note(s)

    lim = api.joint_limits
    lo, hi = lim[:, 0], lim[:, 1]
    try:
        yield from body(api, N, lo, hi)
    finally:
        try:
            p, R = api.object_pose("scissors")
            N(f"END piv={mm(p)} flat={R[2, 2]:.3f} hinge={np.degrees(api.object_joint(HINGE)):.1f}")
        except Exception:
            pass


def goto(api, N, xyz, R, seconds, tag):
    yield from api.move(xyz=xyz, R=R, seconds=seconds)
    yield from api.hold(0.2)
    cmd = np.asarray(xyz, float).copy()
    for _ in range(2):
        m, _ = api.measured_palm()
        e = np.asarray(xyz) - m
        if np.linalg.norm(e) < 0.003:
            break
        N(f"{tag} palm err {mm(e)}")
        cmd = cmd + np.clip(e, -0.03, 0.03)
        yield from api.move(xyz=cmd, R=R, seconds=0.4)
        yield from api.hold(0.2)
    return cmd


def lin_step(J, e):
    return J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)


def body(api, N, lo, hi):
    p0, Rs = api.object_pose("scissors")
    psi = float(np.arctan2(Rs[1, 0], Rs[0, 0]))
    s = float(np.clip((p0[2] - TABLE) / 0.0068, 0.85, 1.15))
    N(f"SC p={mm(p0)} s={s:.3f} yaw={psi:.3f}")

    def W(x, y, h):
        w = p0 + Rs @ np.array([x, y, 0.0])
        w[2] = TABLE + h
        return w

    R = api.hand_down(psi)
    TH, IX, RG = 0, 1, 3
    ki, kr = 4 * IX, 4 * RG
    ix_pre = W(-0.065 * s, 0.024 * s, 0.009)
    rg_pre = W(-0.065 * s, -0.024 * s, 0.009)
    th_pre = W(-0.065 * s, 0.0425 * s + 0.020, 0.0068 * s)
    palm_g = api.palm_for(ix_pre, np.array([0.080, 0.0265, 0.120]), R)

    def P_(w):
        return R.T @ (w - palm_g)

    station = palm_g + np.array([0, 0, 0.14])
    yield from goto(api, N, station, R, 1.5, "station")
    q0 = api.OPEN.copy()
    q0[0:4] = [0.17, -0.8, 0.6, 0.6]
    q0[4:8] = [0.0, 0.45, 1.0, 0.8]
    q0[12:16] = [0.05, 0.45, 1.0, 0.8]
    q0[8:12] = CURL
    q0[16:20] = [0.8, -0.6, 0.0, 0.0]    # little spread and lifted, clear of the ring
    tg = {TH: P_(th_pre), IX: P_(ix_pre), RG: P_(rg_pre)}
    q, P, J, res = yield from ik(api, tg, q0, lo, hi)
    N(f"IK res_mm={ {k: round(v * 1000, 1) for k, v in res.items()} } q={np.round(q, 2).tolist()} tips th={mm(P[TH])} ix={mm(P[IX])} rg={mm(P[RG])}")
    N("J_T " + str(np.round(J[TH], 3).tolist()) + " J_I " + str(np.round(J[IX], 3).tolist()) + " J_R " + str(np.round(J[RG], 3).tolist()))
    q_th = q[0:4].copy()
    q_pre = q.copy()
    q_pre[0:4] = api.OPEN[0:4]            # thumb stays out of the way until the scissors are in the air
    dy = (0.0105 * s + 0.002) - 0.024 * s  # pads onto the inner walls, 6 mm past contact
    q_sq = q_pre.copy()
    # abduction moves the pad in palm -y at about 0.03-0.06 m/rad; distrust probes outside that
    jyi = J[IX][1, 0] if J[IX][1, 0] < -0.02 else -0.045
    jyr = J[RG][1, 0] if J[RG][1, 0] < -0.02 else -0.045
    q_sq[ki] = np.clip(q_pre[ki] + dy / jyi, lo[ki], hi[ki])
    q_sq[kr] = np.clip(q_pre[kr] - dy / jyr, lo[kr], hi[kr])
    sl = slice(kr, kr + 4)
    q_open = q_sq.copy()
    if J[RG][1, 0] < -0.015:
        q_open[sl] = np.clip(q_sq[sl] + lin_step(J[RG], np.array([0, -0.045, 0])), lo[sl], hi[sl])
    q_open[kr] = hi[kr]
    N(f"abd jy ix={jyi:.3f} rg={jyr:.3f} q_sq ix={q_sq[ki]:.2f} rg={q_sq[kr]:.2f} q_open_r={np.round(q_open[sl], 2).tolist()}")
    yield from settle(api, q_pre, 0.5, 0.1)

    hover = palm_g + np.array([0, 0, 0.035])
    hover = yield from goto(api, N, hover, R, 1.2, "hover")
    tw = api.tips_world()
    err = np.mean([(ix_pre - tw[IX])[:2], (rg_pre - tw[RG])[:2]], axis=0)
    err = np.clip(err, -0.01, 0.01)
    N(f"hover err ix={mm(ix_pre - tw[IX])} rg={mm(rg_pre - tw[RG])}")
    off = hover - (palm_g + np.array([0, 0, 0.035]))
    palm_c = palm_g + off + np.array([err[0], err[1], 0.0])
    yield from api.move(xyz=palm_c + np.array([0, 0, 0.035]), R=R, seconds=0.4)
    yield from api.move(xyz=palm_c, R=R, seconds=1.2)
    yield from api.hold(0.3)
    tw = api.tips_world()
    N(f"down err ix={mm(ix_pre - tw[IX])} rg={mm(rg_pre - tw[RG])} F={np.round(api.contact_forces('scissors'), 2).tolist()}")

    yield from api.move(q=q_sq, seconds=0.8)
    yield from api.hold(0.4)
    N(f"squeeze F={np.round(api.contact_forces('scissors'), 2).tolist()}")

    yield from api.move(xyz=palm_c + np.array([0, 0, 0.06]), R=R, seconds=1.2)
    yield from api.move(xyz=palm_c + np.array([0, 0, 0.25]), R=R, seconds=2.0)
    yield from api.hold(0.4)
    p, Rn = api.object_pose("scissors")
    N(f"lift piv={mm(p)} Rz={np.round(Rn[:, 2], 2).tolist()} clr={1000 * clearance(api):.0f} F={np.round(api.contact_forces('scissors'), 2).tolist()}")

    # thumb onto the outside of loop A's outer wall (from the measured pose)
    def a_pt(y):
        pp, Rp = api.object_in_palm("scissors")
        return pp + Rp @ np.array([-0.065 * s, y, 0.0])
    qt = api.q_target(); qt[0:4] = q_th
    Pm = yield from settle(api, qt, 0.6, 0.2)
    Jt = J[TH].copy()
    for _ in range(3):
        e = a_pt(0.0425 * s + 0.020) - Pm[TH]
        if np.linalg.norm(e) < 0.002:
            break
        qn = qt.copy()
        qn[0:4] = np.clip(qt[0:4] + lin_step(Jt, e), lo[0:4], hi[0:4])
        Pn = yield from settle(api, qn, 0.3, 0.15)
        st = qn[0:4] - qt[0:4]
        if st @ st > 1e-8:
            Jt += np.outer((Pn[TH] - Pm[TH]) - Jt @ st, st) / (st @ st)
        qt, Pm = qn, Pn
    e = a_pt(0.0425 * s + 0.020) - Pm[TH]
    N(f"thumb pre err={mm(e)}")
    qs = qt.copy()
    qs[0:4] = np.clip(qt[0:4] + lin_step(Jt, a_pt(0.0425 * s + 0.002) - Pm[TH]), lo[0:4], hi[0:4])
    yield from api.move(q=qs, seconds=0.6)
    yield from api.hold(0.3)
    N(f"thumb on F={np.round(api.contact_forces('scissors'), 2).tolist()}")
    # index across hole A onto the outer wall's inner face (push +y)
    qx = api.q_target()
    qx[ki] = np.clip(q_sq[ki] + (0.0375 * s - 0.002 - (0.0105 * s + 0.002)) / jyi, lo[ki], hi[ki])
    yield from api.move(q=qx, seconds=0.6)
    yield from api.hold(0.3)
    p, Rn = api.object_pose("scissors")
    N(f"A-pinch piv={mm(p)} clr={1000 * clearance(api):.0f} F={np.round(api.contact_forces('scissors'), 2).tolist()} hinge={np.degrees(api.object_joint(HINGE)):.1f}")

    peaks = []
    for c in range(3):
        mx = [0.0]

        def opened():
            a = api.object_joint(HINGE)
            mx[0] = max(mx[0], a)
            return a > 0.30
        qo = api.q_target(); qo[sl] = q_open[sl]
        yield from sweep(api, qo, 1.8, opened)
        yield from api.hold(0.2)
        a_open = api.object_joint(HINGE)

        def closed():
            return api.object_joint(HINGE) < 0.012
        # close toward the squeeze posture only (a hard push shoves the scissors out of the thumb pinch)
        qc = api.q_target(); qc[sl] = q_sq[sl]
        ok2 = yield from sweep(api, qc, 1.8, closed)
        if not ok2:
            qc = api.q_target(); qc[kr] = qc[kr] - 0.1 * np.sign(hi[kr] - q_sq[kr])
            ok2 = yield from sweep(api, qc, 0.6, closed)
        yield from api.hold(0.3)
        peaks.append((round(float(np.degrees(mx[0])), 1), round(float(np.degrees(a_open)), 1), bool(ok2),
                      round(float(np.degrees(api.object_joint(HINGE))), 1), round(1000 * clearance(api))))
        if c >= 1 and sum(1 for pk in peaks if pk[0] > 12 and pk[3] < 2.5) >= 2:
            break
    N(f"cycles (max,held,closed_ok,after,clr) {peaks}")

    # back to the closing pinch, thumb away, set down
    qb = api.q_target(); qb[ki] = q_sq[ki]; qb[kr] = q_sq[kr]
    yield from api.move(q=qb, seconds=0.6)
    qb[0:4] = api.OPEN[0:4]
    yield from api.move(q=qb, seconds=0.6)
    yield from api.hold(0.2)
    yield from api.move(xyz=palm_c + np.array([0, 0, 0.004]), R=R, seconds=3.0)
    yield from api.hold(0.4)
    p, Rn = api.object_pose("scissors")
    N(f"down piv={mm(p)} flat={Rn[2, 2]:.3f} hinge={np.degrees(api.object_joint(HINGE)):.1f}")
    yield from api.move(q=q_pre, seconds=0.5)
    yield from api.move(xyz=palm_c + np.array([0, 0, 0.08]), R=R, seconds=1.0)
    yield from api.hold(0.3)
