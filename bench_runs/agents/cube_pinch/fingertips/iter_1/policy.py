import numpy as np

# Thumb-index pinch geometry (palm frame), from dexagent/hand.py docs:
# pads oppose along the palm z axis (finger direction); thumb toward the wrist.
PC = np.array([0.080, 0.040, 0.118])   # pinch centre
AX = np.array([0.0, 0.0, -1.0])        # centre -> thumb
TABLE = 0.75
TURN = np.radians(38)


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def pinch_tips(hw):
    return {"thumb": PC + hw * AX, "index": PC - hw * AX}


def fmt(v):
    return np.round(np.asarray(v, float), 4).tolist()


def run(api):
    p0, R0 = api.object_pose("cube")
    cy = yaw_of(R0)
    edge = 2 * (p0[2] - TABLE)
    api.note(f"cube p0={fmt(p0)} yaw={np.degrees(cy):.1f} edge~{edge*1000:.1f}mm")
    api.note(f"start tips={fmt(api.tip_targets())} palm={fmt(api.palm_pose()[0])}")

    # Palm yaw square to a cube face, the one closest to the default yaw.
    yaw = api.DEFAULT_YAW + wrap(cy - api.DEFAULT_YAW, np.pi / 2)
    R = api.hand_down(yaw)
    hw = edge / 2
    open_hw = min(hw + 0.013, 0.035)
    for f, x in pinch_tips(open_hw).items():
        api.note(f"reach {f} open: {api.reach(f, x)*1000:.1f}mm")

    g = np.array([p0[0], p0[1], p0[2] + 0.004])  # grasp point, slightly above the centre
    palm_g = api.palm_for(g, PC, R)

    # Pre-grasp above, fingers open, then descend.
    r = yield from api.move(palm_g + [0, 0, 0.08], R, tips=pinch_tips(open_hw), seconds=2.0)
    api.note(f"open ik={r}")
    yield from api.move(palm_g, R, seconds=1.5)
    yield from api.hold(0.2)

    # Centre the measured pads on the cube (xy).
    tw = api.tips_world()
    mid = (tw[0] + tw[1]) / 2
    pc, _ = api.object_pose("cube")
    err = np.array([pc[0] - mid[0], pc[1] - mid[1], 0.0])
    api.note(f"pad-mid vs cube err={fmt(err)} tips_world thumb={fmt(tw[0])} index={fmt(tw[1])}")
    if np.linalg.norm(err) < 0.015:
        palm_g = palm_g + err
        yield from api.move(palm_g, R, seconds=0.5)

    # Close.
    sq = 0.007
    r = yield from api.move(tips=pinch_tips(hw - sq), seconds=1.0, straight=True)
    yield from api.hold(0.3)
    f = api.contact_forces("cube")
    api.note(f"closed ik={r} forces={fmt(f)}")
    for _ in range(2):
        if f[0] > 0.3 and f[1] > 0.3:
            break
        sq += 0.004
        yield from api.move(tips=pinch_tips(hw - sq), seconds=0.5, straight=True)
        yield from api.hold(0.2)
        f = api.contact_forces("cube")
        api.note(f"squeeze {sq*1000:.0f}mm forces={fmt(f)}")

    # Lift.
    lifted = palm_g + [0, 0, 0.12]
    yield from api.move(lifted, R, seconds=2.0)
    yield from api.hold(0.3)
    pc, Rc = api.object_pose("cube")
    api.note(f"lifted cube z+={1000*(pc[2]-p0[2]):.1f}mm forces={fmt(api.contact_forces('cube'))}")

    # Turn the palm about the vertical through the cube, toward the default yaw.
    sign = -1.0 if yaw >= api.DEFAULT_YAW else 1.0
    xyz, Rp = lifted.copy(), R.copy()
    n = 4
    for _ in range(n):
        pc, _ = api.object_pose("cube")
        Rz = rz(sign * TURN / n)
        c = np.array([pc[0], pc[1], 0.0])
        xyz = c + Rz @ (xyz - c)
        Rp = Rz @ Rp
        yield from api.move(xyz, Rp, seconds=0.7)
    yield from api.hold(0.3)
    pc, Rc = api.object_pose("cube")
    api.note(f"turned cube yaw change={np.degrees(wrap(yaw_of(Rc) - cy, np.pi / 2)):.1f} z+={1000*(pc[2]-p0[2]):.1f}mm")

    # Carry the cube back over its start, then lower until it nearly touches.
    dxy = np.array([p0[0] - pc[0], p0[1] - pc[1], 0.0])
    xyz = xyz + dxy
    yield from api.move(xyz, Rp, seconds=1.5)
    pc, _ = api.object_pose("cube")
    dxy = np.array([p0[0] - pc[0], p0[1] - pc[1], 0.0])
    xyz = xyz + dxy + [0, 0, -(pc[2] - p0[2]) + 0.012]
    yield from api.move(xyz, Rp, seconds=1.5)
    pc, _ = api.object_pose("cube")
    dxy = np.array([p0[0] - pc[0], p0[1] - pc[1], 0.0])
    xyz = xyz + dxy + [0, 0, -(pc[2] - p0[2]) + 0.002]
    yield from api.move(xyz, Rp, seconds=1.0)
    yield from api.hold(0.3)
    pc, Rc = api.object_pose("cube")
    api.note(f"placed-held off={fmt(pc - p0)}")

    # Release and back off.
    yield from api.move(tips=pinch_tips(min(hw + 0.012, 0.035)), seconds=0.8, straight=True)
    yield from api.move(xyz + [0, 0, 0.08], Rp, seconds=1.0)
    yield from api.hold(0.3)
    pc, Rc = api.object_pose("cube")
    api.note(f"final off={fmt(pc - p0)} Rzz={Rc[2,2]:.3f}")
