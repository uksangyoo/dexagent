"""cap_twist, joints level.

Strategy: palm held still, palm-down, fingers reaching past the lid. Index /
middle / ring hook the FAR side of the lid rim, pull back toward the axis
(force-regulated through PIP/DIP flexion) and sweep sideways toward the
thumb with their abduction joints: contacts on the far rim moving toward +y
(thumb side) turn the lid counter-clockwise seen from above, with the full
lid radius as moment arm. Lift, swing back, repeat.

Joint-level helpers built here (no IK is available at this level):
  calibrate(): holds a few postures in the air and reads api.tips() to learn
               where the pads are and which abduction sign moves a pad to +y.
"""
import numpy as np

ACTIVE = (1, 2, 3)          # index, middle, ring
F_TARGET = 2.5              # N, pull-in force per engaged finger
G_PIP = 0.005               # rad per N per step, PIP force loop gain
S_WIN = 0.028               # m, a pad pulls on the rim only while within this lateral offset of the axis
R_MAX = 0.0385              # m, largest lid radius in the variation range
PAD_R = 0.008               # m, pad centre to pad surface
PAD_DEPTH = 0.004           # m, pad centre below the lid top
SWEEP = 0.45                # rad, abduction half-range
GOAL = 2 * np.pi + 0.2      # rad, stop turning beyond this


def run(api):
    notes = []
    try:
        yield from _run(api, notes)
    finally:
        for line in (notes[:10] + notes[10:][-45:]):
            api.note(line)


def _run(api, notes):

    # buffer notes: the bench keeps only the last 60 log lines, so they are flushed at the end
    note = lambda txt: notes.append(f"{api.t:6.2f} {txt}")
    lim = api.joint_limits
    note(f"limits {np.round(lim, 2).tolist()}")
    P0, R = api.palm_pose()
    fwd, side, down = R[:, 2], R[:, 1], R[:, 0]

    q = api.OPEN.copy()

    def step(qt):
        yield from api.move(q=qt, seconds=0.02)

    # ---------------------------------------------------------------- calibration
    hook = api.OPEN.copy()
    for f in ACTIVE:
        hook[4 * f:4 * f + 4] = [0.0, 0.55, 0.45, 0.3]
    hook = np.clip(hook, lim[:, 0], lim[:, 1])
    yield from api.move(q=hook, seconds=0.8)
    yield from api.hold(0.4)
    T_hook = api.tips()
    note(f"tips hook {np.round(T_hook * 1000).astype(int).tolist()}")
    qa = hook.copy()
    for f in ACTIVE:
        qa[4 * f] = 0.3
    yield from api.move(q=qa, seconds=0.5)
    yield from api.hold(0.3)
    T_ab = api.tips()
    note(f"tips abd+0.3 {np.round(T_ab * 1000).astype(int).tolist()} q {np.round(api.q(), 2).tolist()}")
    ql = hook.copy()
    for f in ACTIVE:
        ql[4 * f + 1] -= 0.35
    ql = np.clip(ql, lim[:, 0], lim[:, 1])
    yield from api.move(q=ql, seconds=0.5)
    yield from api.hold(0.3)
    T_lift = api.tips()
    note(f"tips lift {np.round(T_lift * 1000).astype(int).tolist()}")
    dy = (T_ab - T_hook)[:, 1] / 0.3          # pad y per rad abduction
    sgn = {f: (1.0 if dy[f] > 0 else -1.0) for f in ACTIVE}
    note(f"dy/dabd mm/rad {np.round(dy * 1000).astype(int).tolist()}")

    # PIP/DIP calibration: with the distal link pointing down, PIP+DIP flexion pulls the pad back (toward the palm)
    qp = hook.copy()
    for f in ACTIVE:
        qp[4 * f + 2] += 0.2
        qp[4 * f + 3] += 0.1
    yield from api.move(q=qp, seconds=0.4)
    yield from api.hold(0.3)
    T_pip = api.tips()
    note(f"tips pip+0.2 {np.round(T_pip * 1000).astype(int).tolist()}")

    # ---------------------------------------------------------------- placement
    # Rim strategy: pads hook the FAR side of the lid (forward of the axis), pull back toward the axis with
    # PIP/DIP (force loop), and sweep toward +y (thumb side) with abduction -> CCW seen from above.
    pj, _ = api.object_pose("jar")
    top = pj[2] + 0.123
    c = np.array([pj[0], pj[1], top])
    fh = fwd - fwd[2] * np.array([0, 0, 1.0]); fh /= np.linalg.norm(fh)
    sh = np.cross(np.array([0, 0, 1.0]), fh)      # horizontal, toward the thumb side (+y palm)
    if np.dot(sh, side) < 0:
        sh = -sh
    W = c + (R_MAX + PAD_R) * fh + np.array([0, 0, -PAD_DEPTH])
    palm_touch = W - R @ T_hook[2]
    yield from api.move(xyz=palm_touch + np.array([0, 0, 0.08]), q=ql, seconds=2.0)
    yield from api.move(xyz=palm_touch + np.array([0, 0, 0.004]), seconds=1.2)
    yield from api.hold(0.3)
    note(f"placed palm {np.round(api.measured_palm()[0], 3).tolist()} cmd {np.round(palm_touch, 3).tolist()}")
    tw = api.tips_world()
    note(f"thumb world {np.round(tw[0], 3).tolist()} little {np.round(tw[4], 3).tolist()} lid c {np.round(c, 3).tolist()}")

    # ---------------------------------------------------------------- strokes
    q = api.q_target().copy()
    lift2 = {f: ql[4 * f + 1] for f in ACTIVE}
    mcp = {f: hook[4 * f + 1] for f in ACTIVE}
    pip0 = {f: hook[4 * f + 2] for f in ACTIVE}
    dip0 = {f: hook[4 * f + 3] for f in ACTIVE}
    pip_out = {f: pip0[f] - 0.12 for f in ACTIVE}
    a_lo = {f: float(np.clip(-SWEEP * sgn[f], lim[4 * f, 0] + 0.02, lim[4 * f, 1] - 0.02)) for f in ACTIVE}
    a_hi = {f: float(np.clip(SWEEP * sgn[f], lim[4 * f, 0] + 0.02, lim[4 * f, 1] - 0.02)) for f in ACTIVE}
    note(f"abd lo {np.round(list(a_lo.values()), 2).tolist()} hi {np.round(list(a_hi.values()), 2).tolist()}")

    def set_pip(qv, f, pip):
        qv[4 * f + 2] = pip
        qv[4 * f + 3] = dip0[f] + 0.5 * (pip - pip0[f])

    a0 = api.object_joint("lid_twist")
    stroke = 0
    while api.object_joint("lid_twist") - a0 < GOAL and api.t < 53:
        stroke += 1
        s0 = api.object_joint("lid_twist")
        # swing back, lifted and opened
        tgt = q.copy()
        for f in ACTIVE:
            tgt[4 * f] = a_lo[f]
            tgt[4 * f + 1] = lift2[f]
            set_pip(tgt, f, pip_out[f])
        yield from api.move(q=tgt, seconds=0.45)
        # lower beside the rim
        for f in ACTIVE:
            tgt[4 * f + 1] = mcp[f]
        yield from api.move(q=tgt, seconds=0.35)
        q = tgt.copy()
        n, pre = 60, 15
        fsum = np.zeros(6)
        neng = {f: 0 for f in ACTIVE}
        rc = []
        for k in range(pre + n + 5):
            u = min(1.0, max(0.0, (k - pre) / n))
            us = u * u * (3 - 2 * u)
            F = api.contact_forces("lid")
            fsum += F
            tw = api.tips_world()
            for f in ACTIVE:
                rel = tw[f] - c
                s = float(rel @ sh)
                eng = abs(s) < S_WIN
                neng[f] += eng
                q[4 * f] = a_lo[f] + us * (a_hi[f] - a_lo[f])
                if eng:
                    pip = q[4 * f + 2] + G_PIP * float(np.clip(F_TARGET - F[f], -2, 2))
                    set_pip(q, f, float(np.clip(pip, pip_out[f], pip0[f] + 0.7)))
                    if F[f] > 0.5:
                        rc.append(float(np.linalg.norm(rel[:2])))
                else:
                    set_pip(q, f, q[4 * f + 2] + 0.2 * (pip_out[f] - q[4 * f + 2]))
            yield from step(q)
        # lift off
        tgt = q.copy()
        for f in ACTIVE:
            tgt[4 * f + 1] = lift2[f]
            set_pip(tgt, f, pip_out[f])
        yield from api.move(q=tgt, seconds=0.3)
        q = tgt.copy()
        s1 = api.object_joint("lid_twist")
        note(f"stroke {stroke}: +{np.degrees(s1 - s0):.1f} deg, total {np.degrees(s1 - a0):.1f}, "
             f"mean F {np.round(fsum / (pre + n + 5), 2).tolist()} eng {[int(v) for v in neng.values()]} "
             f"pad r {np.round(np.percentile(rc, [10, 50, 90]) * 1000, 1).tolist() if rc else None}")
    note(f"done: total {np.degrees(api.object_joint('lid_twist') - a0):.1f} deg at t={api.t:.1f}")
    yield from api.hold(0.3)
