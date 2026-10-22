# cube_pinch at level layered
#
# Plan: yaw the palm to square with the cube, preshape a wide thumb-index
# pinch (fingertip layer), descend, close onto the faces (contact layer
# grasp/squeeze, force-checked), lift with the arm, turn the whole palm about
# the vertical through the cube (arm), lower back to the start xy with a
# measured correction, release (contact layer) and back off.
import numpy as np

# Thumb-index pinch geometry measured on this hand (from dexagent/hand.py, palm frame):
# thumb toward the wrist (-z), index toward the fingertips (+z).
PINCH_C = np.array([0.080, 0.040, 0.118])
Z = np.array([0.0, 0.0, 1.0])
HW_ASSUMED = 0.018     # cube half-edge assumed for the contact points (min is 17 mm)
OPEN_HALF = 0.033      # preshape pad-centre distance from the pinch centre (aperture 66 mm)
LIFT = 0.11
TURN = 0.65            # rad of palm yaw while lifted (37 deg; need > 25)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def cube_yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap90(d):
    return (d + np.pi / 4) % (np.pi / 2) - np.pi / 4


def forces(api):
    try:
        return np.asarray(api.contact_forces("cube"), float)
    except Exception:
        return np.zeros(6)


def grasp_points(api, theta):
    """Object-frame points on the two faces normal to the palm's finger direction."""
    _, R = api.object_pose("cube")
    zw = np.array([np.cos(theta), np.sin(theta), 0.0])
    loc = R.T @ zw
    k = int(np.argmax(np.abs(loc)))
    e = np.zeros(3)
    e[k] = np.sign(loc[k])
    pts = {"index": HW_ASSUMED * e, "thumb": -HW_ASSUMED * e}
    nrm = {"index": e.copy(), "thumb": -e.copy()}
    return pts, nrm


def attempt(api, p0, theta, R_g):
    """One pick attempt. Returns True if the cube came up with the hand."""
    # preshape: open, tuck the three spare fingers, wide pinch aperture
    yield from api.open(0.5)
    yield from api.tuck(["middle", "ring", "little"], 0.5)
    # widest aperture each finger reaches cleanly (IK check only, no motion), so a
    # 46 mm cube still gets a few mm of clearance on the way down
    half = {}
    for f, sgn in (("thumb", -1.0), ("index", 1.0)):
        half[f] = OPEN_HALF
        for h in (0.040, 0.038, 0.036, 0.034):
            if api.reach(f, PINCH_C + sgn * h * Z) < 0.0015:
                half[f] = h
                break
    api.note(f"preshape half-apertures {half}")
    pre = {"thumb": PINCH_C - half["thumb"] * Z, "index": PINCH_C + half["index"] * Z}
    p, _ = api.object_pose("cube")
    above = api.palm_for(p + np.array([0, 0, 0.07]), PINCH_C, R_g)
    r = yield from api.move_tips(above, R_g, tips=pre, seconds=2.0)
    api.note(f"preshape ik {r}")
    # descend so the pinch centre sits at the cube centre
    p, _ = api.object_pose("cube")
    at = api.palm_for(p, PINCH_C, R_g)
    yield from api.carry(at, R_g, 1.5)
    yield from api.hold(0.2)
    # close
    pts, nrm = grasp_points(api, theta)
    r = yield from api.grasp("cube", pts, squeeze=0.002, seconds=1.0, normals=nrm)
    api.note(f"grasp {r}")
    yield from api.hold(0.3)
    for k in range(4):
        f = forces(api)
        api.note(f"forces after grasp/squeeze {k}: {np.round(f, 3).tolist()}")
        if f[0] > 0.15 and f[1] > 0.15:
            break
        yield from api.squeeze(["thumb", "index"], "cube", 0.002, 0.4)
        yield from api.hold(0.2)
    # small test lift
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + np.array([0, 0, 0.03]), R, 0.8)
    yield from api.hold(0.2)
    p, _ = api.object_pose("cube")
    api.note(f"test lift: cube rose {1000 * (p[2] - p0[2]):.1f} mm, forces {np.round(forces(api), 3).tolist()}")
    return p[2] - p0[2] > 0.02


def run(api):
    p0, R0 = api.object_pose("cube")
    yaw0 = cube_yaw(R0)
    api.note(f"cube start {np.round(p0, 4).tolist()} yaw {np.degrees(yaw0):.1f}")
    # palm yaw: the cube face direction nearest the default yaw
    theta = yaw0 + np.round((api.DEFAULT_YAW - yaw0) / (np.pi / 2)) * (np.pi / 2)
    R_g = api.hand_down(theta)
    api.note(f"palm yaw {np.degrees(theta):.1f} deg")

    ok = False
    for tries in range(3):
        ok = yield from attempt(api, p0, theta, R_g)
        if ok:
            break
        api.note(f"attempt {tries} failed; reopening")
        yield from api.open(0.6)
        xyz, R = api.palm_pose()
        yield from api.carry(xyz + np.array([0, 0, 0.05]), R, 0.8)
        yield from api.hold(0.5)
        p0b, Rb = api.object_pose("cube")
        theta = cube_yaw(Rb) + np.round((api.DEFAULT_YAW - cube_yaw(Rb)) / (np.pi / 2)) * (np.pi / 2)
        R_g = api.hand_down(theta)

    # lift the rest of the way
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + np.array([0, 0, LIFT - 0.03]), R, 1.6)
    yield from api.hold(0.3)
    p, Rc = api.object_pose("cube")
    api.note(f"lifted {1000 * (p[2] - p0[2]):.1f} mm")

    # turn the palm about the vertical through the cube, toward the default yaw
    sgn = -1.0 if theta > api.DEFAULT_YAW else 1.0
    steps = 4
    for i in range(steps):
        xyz, R = api.palm_pose()
        c, _ = api.object_pose("cube")
        Rm = Rz(sgn * TURN / steps)
        xyz1 = c + Rm @ (xyz - c)
        yield from api.carry(xyz1, Rm @ R, 0.7)
    yield from api.hold(0.3)
    p, Rc = api.object_pose("cube")
    api.note(f"turned: cube yaw change {np.degrees(wrap90(cube_yaw(Rc) - yaw0)):.1f} deg, height {1000 * (p[2] - p0[2]):.1f} mm")

    # lower back over the start, correcting xy from the measured cube position
    for h, sec in ((0.03, 1.6), (0.008, 1.0)):
        xyz, R = api.palm_pose()
        p, _ = api.object_pose("cube")
        yield from api.carry(xyz + (p0 + np.array([0, 0, h]) - p), R, sec)
        yield from api.hold(0.3)
    p, _ = api.object_pose("cube")
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + (p0 + np.array([0, 0, 0.001]) - p), R, 0.8)
    yield from api.hold(0.3)
    p, _ = api.object_pose("cube")
    api.note(f"placed: offset {1000 * np.linalg.norm(p[:2] - p0[:2]):.1f} mm, dz {1000 * (p[2] - p0[2]):.1f} mm")

    # release and back off
    yield from api.release(["thumb", "index"], "cube", clear=0.02, seconds=0.8)
    yield from api.hold(0.3)
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + np.array([0, 0, 0.08]), R, 1.2)
    yield from api.hold(0.5)
    p, Rc = api.object_pose("cube")
    api.note(f"final: offset {1000 * np.linalg.norm(p[:2] - p0[:2]):.1f} mm, dz {1000 * (p[2] - p0[2]):.1f} mm, Rzz {Rc[2, 2]:.3f}")
