import numpy as np

# cube_palm, force level.
# Plan: thumb-index pinch along the palm z axis (thumb toward the wrist), held by the
# grip-force servo; lift; roll the palm about its own z axis (pure wrist roll) to
# palm-up; lower the pinched cube toward the palm with the pads until the palm feels it;
# release the pads (palm alone carries); regrasp with the servo; roll back; place; release.

PINCH = np.array([0.080, 0.040, 0.118])   # palm-frame thumb/index pinch centre
EZ = np.array([0.0, 0.0, 1.0])
GRIP_N = 1.2
TABLE_Z = 0.75


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def r3(v):
    return np.round(np.asarray(v, float) * 1000, 1).tolist()


def face_axis(R, d):
    """Cube axis (columns of R, palm frame) most aligned with palm-frame direction d, signed toward d."""
    k = int(np.argmax(np.abs(R.T @ d)))
    a = R[:, k]
    return a * np.sign(a @ d)


def feasible(api, targets, tol=0.003):
    return all(api.reach(f, x) < tol for f, x in targets.items())


def forces(api):
    return np.round(api.contact_forces("cube"), 3)


def pinch_tips(c, a, w):
    return {"thumb": c - w * a, "index": c + w * a}


def run(api):
    p0, R0 = api.object_pose("cube")
    h = p0[2] - TABLE_Z                     # half edge
    yc = np.arctan2(R0[1, 0], R0[0, 0])
    yaw = 0.7 + ((yc - 0.7 + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    Rd = api.hand_down(yaw)
    api.note(f"cube p0={r3(p0)} half_edge={h*1000:.1f}mm yaw_cube={yc:.3f} palm_yaw={yaw:.3f}")
    lim = api.joint_limits
    api.note(f"finger limits lo={np.round(lim[:, 0], 2).tolist()} hi={np.round(lim[:, 1], 2).tolist()}")
    api.note(f"start tips(palm,mm)={r3(api.tips())}")

    # Other fingers flat / slightly hyperextended, out of the cube's way palm-down and palm-up.
    others = {}
    for i, f in ((2, "middle"), (3, "ring"), (4, "little")):
        q = np.array([api.OPEN[4 * i], -0.15, 0.0, 0.0])
        others[f] = np.clip(q, lim[4 * i:4 * i + 4, 0], lim[4 * i:4 * i + 4, 1])

    w_touch = h + 0.008
    w_open = w_touch + 0.015
    w_close = w_touch - 0.002
    c = PINCH.copy()
    for cy in (0.04, 0.035, 0.045, 0.03):
        cc = np.array([PINCH[0], cy, PINCH[2]])
        if feasible(api, pinch_tips(cc, EZ, w_open)) and feasible(api, pinch_tips(cc, EZ, w_close)):
            c = cc
            break
    api.note(f"grasp centre palm={r3(c)}")

    # ---------------------------------------------------------------- pick
    above = api.palm_for(p0 + np.array([0, 0, 0.06]), c, Rd)
    yield from api.move_joints(above, Rd, q=others, seconds=2.0)
    yield from api.move_tips(tips=pinch_tips(c, EZ, w_open), seconds=0.6)
    for attempt in range(2):
        p, R = api.object_pose("cube")
        at = api.palm_for(np.array([p[0], p[1], p0[2]]), c, Rd)
        yield from api.carry(at, Rd, seconds=1.0)
        pc, Rc = api.object_in_palm("cube")
        a = face_axis(Rc, EZ)
        yield from api.move_tips(tips=pinch_tips(pc, a, w_close), seconds=0.5, straight=True)
        r = yield from api.grip("cube", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.0)
        api.note(f"grip {r} cube_in_palm={r3(api.object_in_palm('cube')[0])}")
        lift_xyz = at + np.array([0, 0, 0.20])
        yield from api.carry(lift_xyz, Rd, seconds=1.5)
        p, _ = api.object_pose("cube")
        api.note(f"lifted cube z={p[2]:.3f} forces={forces(api).tolist()} grip={api.grip_state()}")
        if p[2] > p0[2] + 0.1:
            break
        api.note("LIFT FAILED, retrying")
        yield from api.ungrip()
        yield from api.move_tips(tips=pinch_tips(c, EZ, w_open), seconds=0.5)
        yield from api.carry(above, Rd, seconds=1.0)

    # ---------------------------------------------------------------- roll palm-up
    for k in range(1, 5):
        yield from api.carry(lift_xyz, Rd @ rz(k * np.pi / 4), seconds=0.8)
    yield from api.hold(0.3)
    Ru = Rd @ rz(np.pi)
    pc, Rc = api.object_in_palm("cube")
    _, Rp = api.measured_palm()
    api.note(f"palm up: palm_up={Rp[2, 0]:.3f} cube_in_palm={r3(pc)} forces={forces(api).tolist()} grip={api.grip_state()}")

    # ---------------------------------------------------------------- lower onto palm
    tt = api.tip_targets()
    T, I = tt[0].copy(), tt[1].copy()
    best, best_s = np.zeros(3), 1e9
    deep = np.array([-0.02, 0.0, 0.0])
    for dz in (0.0, -0.01, -0.02, -0.03, -0.04):
        for dy in (0.0, -0.01, -0.02, -0.03):
            d = np.array([0.0, dy, dz])
            if not feasible(api, {"thumb": T + d, "index": I + d}):
                continue
            if not feasible(api, {"thumb": T + d + deep, "index": I + d + deep}):
                continue
            # toward the wrist (over the palm, off the finger bases), y about 20 mm
            s = abs(pc[2] + dz - 0.08) + abs(pc[1] + dy - 0.02)
            if s < best_s:
                best, best_s = d, s
    api.note(f"in-hand shift chosen {r3(best)}")
    if np.any(best):
        yield from api.translate("cube", best, ["thumb", "index"], seconds=0.8)
    T, I = T + best, I + best
    dx = 0.0
    reason = "limit"
    for _ in range(18):
        f = forces(api)
        if f[5] > 0.03:
            reason = f"palm contact {f[5]:.3f}N"
            break
        nd = dx - 0.004
        d = np.array([nd, 0, 0])
        if not feasible(api, {"thumb": T + d, "index": I + d}):
            reason = "ik limit"
            break
        yield from api.translate("cube", np.array([-0.004, 0, 0]), ["thumb", "index"], seconds=0.2)
        dx = nd
    api.note(f"lowered dx={dx*1000:.0f}mm stop={reason} cube_in_palm={r3(api.object_in_palm('cube')[0])} forces={forces(api).tolist()}")

    # ---------------------------------------------------------------- palm alone
    yield from api.ungrip()
    yield from api.release(["thumb", "index"], "cube", clear=0.012, seconds=0.6)
    # Extend the index at its base so no index link stays under / against the cube.
    q_index = api.q_target()[4:8].copy()
    qi = q_index.copy()
    qi[1] -= 0.4
    yield from api.move_joints(q={"index": qi}, seconds=0.5)
    for _ in range(8):
        yield from api.hold(0.2)
        pc, _ = api.object_in_palm("cube")
        api.note(f"on palm: cube_in_palm={r3(pc)} forces={forces(api).tolist()}")

    # ---------------------------------------------------------------- regrasp
    yield from api.move_joints(q={"index": q_index}, seconds=0.5)
    pc, Rc = api.object_in_palm("cube")
    a = face_axis(Rc, EZ)
    yield from api.move_tips(tips=pinch_tips(pc, a, w_open - 0.004), seconds=0.5)
    pc, Rc = api.object_in_palm("cube")
    a = face_axis(Rc, EZ)
    yield from api.move_tips(tips=pinch_tips(pc, a, w_close), seconds=0.5, straight=True)
    r = yield from api.grip("cube", {"thumb": GRIP_N, "index": GRIP_N}, seconds=1.0)
    api.note(f"regrip {r}")
    up = np.array([0.015, 0, 0])
    if feasible(api, {"thumb": api.tip_targets()[0] + up, "index": api.tip_targets()[1] + up}):
        yield from api.translate("cube", up, ["thumb", "index"], seconds=0.5)

    # ---------------------------------------------------------------- roll back palm-down
    for k in range(3, -1, -1):
        yield from api.carry(lift_xyz, Rd @ rz(k * np.pi / 4), seconds=0.8)
    yield from api.hold(0.3)
    p, _ = api.object_pose("cube")
    api.note(f"palm down again cube z={p[2]:.3f} forces={forces(api).tolist()}")

    # ---------------------------------------------------------------- place
    pc, _ = api.object_in_palm("cube")
    tgt = np.array([p0[0], p0[1], p0[2] + 0.03])
    yield from api.carry(api.palm_for(tgt, pc, Rd), Rd, seconds=1.5)
    pc, _ = api.object_in_palm("cube")
    tgt = np.array([p0[0], p0[1], p0[2] + 0.002])
    yield from api.carry(api.palm_for(tgt, pc, Rd), Rd, seconds=1.2)
    yield from api.hold(0.3)
    # Guarded extra descent: lower 2 mm at a time while the cube still follows.
    for _ in range(6):
        p, _ = api.object_pose("cube")
        if p[2] - p0[2] < 0.0025:
            break
        P, _ = api.palm_pose()
        yield from api.carry(P - np.array([0, 0, 0.002]), Rd, seconds=0.15)
        yield from api.hold(0.1)
        p2, _ = api.object_pose("cube")
        if p[2] - p2[2] < 0.0007:
            api.note("descent blocked")
            break
    p, _ = api.object_pose("cube")
    api.note(f"placed cube={r3(p)} dz={1000*(p[2]-p0[2]):.1f}mm forces={forces(api).tolist()}")
    yield from api.ungrip()
    yield from api.release(["thumb", "index"], "cube", clear=0.015, seconds=0.6)
    P, _ = api.palm_pose()
    yield from api.carry(P + np.array([0, 0, 0.08]), Rd, seconds=1.0)
    yield from api.hold(0.3)
    p, _ = api.object_pose("cube")
    api.note(f"end cube={r3(p)} off={1000*np.linalg.norm(p[:2]-p0[:2]):.1f}mm forces={forces(api).tolist()}")
