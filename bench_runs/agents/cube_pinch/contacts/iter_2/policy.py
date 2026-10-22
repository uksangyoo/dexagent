import numpy as np

TABLE_Z = 0.75
PINCH_CENTRE = np.array([0.080, 0.040, 0.118])  # palm-frame thumb-index pinch centre (from dexagent/hand.py docs)
UP_OFFSET = 0.005      # pinch this far above the cube centre (keeps fingertip ends off the table)
STANDOFF = 0.008
SQUEEZE = 0.004
LIFT = 0.115
TURN = np.radians(35)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def face_points(api, obj, h, centre_palm=None, shift_palm=np.zeros(3)):
    """Thumb/index contact points + normals in the OBJECT frame for an opposition pinch
    along the palm z axis (index on the fingertip-side face, thumb on the wrist-side face).
    If centre_palm is given, the cube is assumed (virtually) centred there in the palm
    frame (used for pre-shaping before the palm descends)."""
    p, R = api.object_in_palm(obj)
    _, Rp = api.measured_palm()
    up = Rp.T @ np.array([0, 0, 1.0])            # world up in palm frame
    # cube face normal (palm frame) closest to palm +z
    k = int(np.argmax(np.abs(R[2, :])))
    n = R[:, k] * np.sign(R[2, k])
    c = p if centre_palm is None else np.asarray(centre_palm, float)
    pts, nrm = {}, {}
    for f, s in (("index", 1.0), ("thumb", -1.0)):
        x = c + s * h * n + UP_OFFSET * up + shift_palm
        pts[f] = R.T @ (x - p)
        nrm[f] = R.T @ (s * n)
    return pts, nrm


def run(api):
    obj = "cube"
    p0, R0 = api.object_pose(obj)
    h = float(np.clip(p0[2] - TABLE_Z, 0.015, 0.025))   # resting cube: centre height = half edge
    psi = yaw_of(R0)
    cands = psi + np.pi / 2 * np.arange(-4, 5)
    yaw = float(cands[np.argmin(np.abs(cands - api.DEFAULT_YAW))])
    api.note(f"p0={np.round(p0, 4).tolist()} h={h:.4f} cube_yaw={np.degrees(psi):.1f} palm_yaw={np.degrees(yaw):.1f}")

    Rg = api.hand_down(yaw)
    cw = p0 + np.array([0, 0, UP_OFFSET])
    grasp_xyz = api.palm_for(cw, PINCH_CENTRE, Rg)
    pre_xyz = grasp_xyz + np.array([0, 0, 0.05])

    # tuck the unused fingers, move above the cube
    yield from api.tuck(["middle", "ring", "little"], seconds=0.6)
    yield from api.carry(pre_xyz, Rg, seconds=2.0)
    yield from api.hold(0.2)

    # pre-shape thumb/index around where the cube will be once the palm descends
    # (face_points adds UP_OFFSET; PINCH_CENTRE already is the contact centre, so undo it)
    _, Rp = api.measured_palm()
    up = Rp.T @ np.array([0, 0, 1.0])
    pts, nrm = face_points(api, obj, h, centre_palm=PINCH_CENTRE - UP_OFFSET * up)
    standoff = float(np.clip(0.037 - h - 0.008, 0.004, STANDOFF))  # stay inside the ~72 mm pad aperture for big cubes
    r = yield from api.approach(obj, pts, standoff=standoff, seconds=1.0, normals=nrm)
    api.note(f"approach {r}")

    yield from api.carry(grasp_xyz, Rg, seconds=1.5)
    yield from api.hold(0.3)
    p, _ = api.object_pose(obj)
    api.note(f"cube before grasp {np.round(p, 4).tolist()} tips_w={np.round(api.tips_world()[:2], 4).tolist()}")

    pts, nrm = face_points(api, obj, h)
    r = yield from api.grasp(obj, pts, squeeze=SQUEEZE, seconds=0.8, normals=nrm)
    yield from api.hold(0.3)
    f = api.contact_forces(obj)
    api.note(f"grasp forces={np.round(f, 2).tolist()} ik={r.get('ik_residual_mm')}")
    for _ in range(3):
        if min(f[0], f[1]) > 0.4:
            break
        yield from api.squeeze(["thumb", "index"], obj, 0.002, seconds=0.4)
        yield from api.hold(0.2)
        f = api.contact_forces(obj)
        api.note(f"extra squeeze forces={np.round(f, 2).tolist()}")

    # lift
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + np.array([0, 0, LIFT]), R, seconds=2.0)
    yield from api.hold(0.3)
    p, _ = api.object_pose(obj)
    api.note(f"lifted cube dz={1000 * (p[2] - p0[2]):.1f}mm forces={np.round(api.contact_forces(obj), 2).tolist()}")
    if p[2] - p0[2] < 0.09:
        xyz, R = api.palm_pose()
        yield from api.carry(xyz + np.array([0, 0, 0.095 - (p[2] - p0[2])]), R, seconds=1.0)
        yield from api.hold(0.2)

    # turn about the vertical through the cube by rotating the wrist
    sign = 1.0 if yaw <= api.DEFAULT_YAW else -1.0
    n_steps = 5
    for _ in range(n_steps):
        c, _ = api.object_pose(obj)
        xyz, R = api.palm_pose()
        Rs = Rz(sign * TURN / n_steps)
        yield from api.carry(c + Rs @ (xyz - c), Rs @ R, seconds=0.5)
    yield from api.hold(0.4)
    p, Rc = api.object_pose(obj)
    api.note(f"turned: cube yaw={np.degrees(yaw_of(Rc)):.1f} dz={1000 * (p[2] - p0[2]):.1f}mm")

    # put it back: closed-loop on the cube position
    for target_dz, secs in ((0.02, 2.0), (0.002, 1.2)):
        p, _ = api.object_pose(obj)
        xyz, R = api.palm_pose()
        d = np.array([p0[0] - p[0], p0[1] - p[1], p0[2] + target_dz - p[2]])
        yield from api.carry(xyz + d, R, seconds=secs)
        yield from api.hold(0.3)
    p, _ = api.object_pose(obj)
    api.note(f"placed: off={1000 * np.linalg.norm(p[:2] - p0[:2]):.1f}mm dz={1000 * (p[2] - p0[2]):.1f}mm")

    # release and retreat
    yield from api.release(["thumb", "index"], obj, clear=0.02, seconds=0.6)
    yield from api.hold(0.3)
    xyz, R = api.palm_pose()
    yield from api.carry(xyz + np.array([0, 0, 0.08]), R, seconds=1.0)
    yield from api.hold(0.3)
    p, _ = api.object_pose(obj)
    api.note(f"final: off={1000 * np.linalg.norm(p[:2] - p0[:2]):.1f}mm dz={1000 * (p[2] - p0[2]):.1f}mm")
