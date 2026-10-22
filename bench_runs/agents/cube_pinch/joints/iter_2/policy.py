import numpy as np

TABLE_Z = 0.75
TH, IX = 0, 1
# Palm-frame pinch centre for thumb/index (palm x out of the face, z along fingers).
C = np.array([0.080, 0.040, 0.118])
CURL = np.array([0, 1.6, 1.3, 1.1])
LITTLE_CURL = np.array([0.3, 1.4, 1.3, 1.1])


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


# ---------------------------------------------------------------- skills
def settle(api, q, tm=0.2, th=0.15):
    """Command finger joints q, wait, return measured palm-frame pads (5,3)."""
    yield from api.move(q=q, seconds=tm)
    yield from api.hold(th)
    return api.tips()


def finger_ik(api, q, targets, J=None, iters=12, tol=0.0015, lam=0.008, maxstep=0.35):
    """Online IK: drive the pads of the fingers in `targets` {i: palm xyz} there,
    measuring pads with api.tips(). Jacobian from finite differences (if J is None)
    then Broyden updates. Returns (q, tips, J, worst_err)."""
    lim = api.joint_limits
    fs = list(targets)
    q = q.copy()
    tips = yield from settle(api, q)
    if J is None:
        J = {f: np.zeros((3, 4)) for f in fs}
        d = 0.12
        for j in range(4):
            qp = q.copy()
            sg = {}
            for f in fs:
                k = 4 * f + j
                sg[f] = 1.0 if q[k] + d < lim[k, 1] - 1e-3 else -1.0
                qp[k] = q[k] + sg[f] * d
            tp = yield from settle(api, qp)
            for f in fs:
                J[f][:, j] = (tp[f] - tips[f]) / (sg[f] * d)
            q, tips = qp, tp
    worst = 1.0
    for _ in range(iters):
        err = {f: np.asarray(targets[f]) - tips[f] for f in fs}
        worst = max(np.linalg.norm(e) for e in err.values())
        if worst < tol:
            break
        qn = q.copy()
        for f in fs:
            Jf = J[f]
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam ** 2 * np.eye(3), err[f])
            n = np.linalg.norm(dq)
            if n > maxstep:
                dq *= maxstep / n
            sl = slice(4 * f, 4 * f + 4)
            qn[sl] = np.clip(q[sl] + dq, lim[sl, 0], lim[sl, 1])
        tn = yield from settle(api, qn)
        for f in fs:
            sl = slice(4 * f, 4 * f + 4)
            dqa = qn[sl] - q[sl]
            if dqa @ dqa > 1e-6:
                dp = tn[f] - tips[f]
                J[f] += np.outer(dp - J[f] @ dqa, dqa) / (dqa @ dqa)
        q, tips = qn, tn
    err = {f: np.asarray(targets[f]) - tips[f] for f in fs}
    worst = max(np.linalg.norm(e) for e in err.values())
    return q, tips, J, worst


def r3(v):
    return np.round(np.asarray(v) * 1000).astype(int).tolist()


# ---------------------------------------------------------------- policy
def run(api):
    lim = api.joint_limits
    p0, R0 = api.object_pose("cube")
    h = p0[2] - TABLE_Z                      # half edge (cube rests on the table)
    yaw_c = float(np.arctan2(R0[1, 0], R0[0, 0]))
    yaw_g = api.DEFAULT_YAW + wrap(yaw_c - api.DEFAULT_YAW, np.pi / 2)
    Rg = api.hand_down(yaw_g)
    api.note(f"cube p0={r3(p0)} h={h*1000:.1f}mm yaw={np.degrees(yaw_c):.1f} grasp_yaw={np.degrees(yaw_g):.1f}")

    q = api.OPEN.copy()
    q[8:12], q[12:16], q[16:20] = CURL, CURL, LITTLE_CURL
    q = np.clip(q, lim[:, 0], lim[:, 1])
    start_xyz, _ = api.palm_pose()
    yield from api.move(xyz=start_xyz, R=Rg, q=q, seconds=1.0)

    # --- finger IK in the air: open (pre-grasp) and squeeze postures
    ho = min(h + 0.008 + 0.009, 0.036)
    hs = h + 0.008 - 0.006
    z = np.array([0, 0, 1.0])
    q_o, tips_o, J, eo = yield from finger_ik(api, q, {TH: C - ho * z, IX: C + ho * z}, iters=14)
    # Fresh Jacobian at the open posture and stronger damping/smaller steps keep the
    # squeeze posture on the same thumb branch as the open one.
    q_s, tips_s, J, es = yield from finger_ik(api, q_o, {TH: C - hs * z, IX: C + hs * z}, iters=8,
                                              lam=0.015, maxstep=0.2)
    api.note(f"IK open err={eo*1000:.1f}mm q={np.round(q_o[:8], 2).tolist()} tips={r3(tips_o[:2])}")
    api.note(f"IK squeeze err={es*1000:.1f}mm q={np.round(q_s[:8], 2).tolist()} tips={r3(tips_s[:2])} others={r3(tips_s[2:])}")
    tips_o = yield from settle(api, q_o, 0.3, 0.2)

    # --- approach above the cube with the open pinch
    m_o = 0.5 * (tips_o[TH] + tips_o[IX])
    m_s = 0.5 * (tips_s[TH] + tips_s[IX])
    grasp_pt = p0 + np.array([0, 0, 0.004])
    palm_g = api.palm_for(grasp_pt, m_s, Rg)
    above = palm_g + np.array([0, 0, 0.05])
    yield from api.move(xyz=above, R=Rg, seconds=1.5)
    yield from api.hold(0.4)
    tw = api.tips_world()
    mid = 0.5 * (tw[TH] + tw[IX])
    xyz_c, _ = api.palm_pose()
    expected = xyz_c + Rg @ m_o
    corr = expected - mid
    corr[2] = 0.0
    # Keep every fingertip clear of the table at grasp height.
    low = tw[:, 2].min() - 0.05
    lift_fix = max(0.0, TABLE_Z + 0.006 - low)
    api.note(f"approach corr={r3(corr)} lowest_tip_at_grasp={low*1000:.0f} raise={lift_fix*1000:.1f}mm")
    palm_g = palm_g + corr + np.array([0, 0, lift_fix])
    yield from api.move(xyz=palm_g + np.array([0, 0, 0.05]), R=Rg, seconds=0.4)
    yield from api.move(xyz=palm_g, R=Rg, seconds=1.0)
    yield from api.hold(0.3)

    # --- close
    yield from api.move(q=q_s, seconds=0.8)
    yield from api.hold(0.3)
    f = api.contact_forces("cube")
    if min(f[TH], f[IX]) < 0.3:
        q_s = np.clip(q_o + 1.6 * (q_s - q_o), lim[:, 0], lim[:, 1])
        yield from api.move(q=q_s, seconds=0.4)
        yield from api.hold(0.3)
        f2 = api.contact_forces("cube")
        api.note(f"weak grip {np.round(f, 2).tolist()} -> squeezed more {np.round(f2, 2).tolist()}")
        f = f2
    api.note(f"grip forces {np.round(f, 2).tolist()} cube_in_palm={r3(api.object_in_palm('cube')[0])}")

    # --- lift
    up = palm_g + np.array([0, 0, 0.11])
    yield from api.move(xyz=up, R=Rg, seconds=1.5)
    yield from api.hold(0.3)
    pc, _ = api.object_pose("cube")
    api.note(f"lifted cube dz={(pc[2]-p0[2])*1000:.0f}mm forces={np.round(api.contact_forces('cube'), 2).tolist()}")

    # --- turn about the vertical axis through the cube
    theta = -0.7 if yaw_g > api.DEFAULT_YAW else 0.7
    xyz_c, Rc = api.palm_pose()
    for k in range(1, 5):
        Rk = rz(theta * k / 4)
        yield from api.move(xyz=pc + Rk @ (xyz_c - pc), R=Rk @ Rc, seconds=0.5)
    yield from api.hold(0.3)
    pc2, Rc2 = api.object_pose("cube")
    dyaw = wrap(np.arctan2(Rc2[1, 0], Rc2[0, 0]) - yaw_c, np.pi / 2)
    api.note(f"turned: cube dz={(pc2[2]-p0[2])*1000:.0f}mm dyaw={np.degrees(dyaw):.1f}")
    # Closed-loop top-up if the cube slipped in the grasp (need > 25 deg, aim 38).
    extra_total = 0.0
    for _ in range(2):
        short = np.radians(38) - abs(dyaw)
        if short < np.radians(4) or extra_total > 0.39:
            break
        step = np.sign(theta) * min(short + np.radians(3), 0.4 - extra_total)
        extra_total += abs(step)
        xyz_c, Rc = api.palm_pose()
        yield from api.move(xyz=pc2 + rz(step) @ (xyz_c - pc2), R=rz(step) @ Rc, seconds=0.5)
        yield from api.hold(0.3)
        pc2, Rc2 = api.object_pose("cube")
        dyaw = wrap(np.arctan2(Rc2[1, 0], Rc2[0, 0]) - yaw_c, np.pi / 2)
        api.note(f"top-up turn {np.degrees(step):.1f} -> dyaw={np.degrees(dyaw):.1f}")

    # --- put down at the start xy
    for dz, sec in ((0.015, 1.5), (0.002, 0.8)):
        pc, _ = api.object_pose("cube")
        xyz_c, Rc = api.palm_pose()
        target = xyz_c + (p0 + np.array([0, 0, dz]) - pc)
        yield from api.move(xyz=target, R=Rc, seconds=sec)
        yield from api.hold(0.3)
    pc, _ = api.object_pose("cube")
    api.note(f"placed cube off={r3(pc - p0)}")

    # --- release and retreat
    yield from api.move(q=q_o, seconds=0.6)
    yield from api.hold(0.2)
    xyz_c, Rc = api.palm_pose()
    yield from api.move(xyz=xyz_c + np.array([0, 0, 0.08]), R=Rc, seconds=1.0)
    pc, _ = api.object_pose("cube")
    api.note(f"final cube off={r3(pc - p0)}")
