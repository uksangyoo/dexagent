import numpy as np
from scipy.spatial.transform import Rotation

# Thumb-index pinch centre in the palm frame (measured on this hand, dexagent/hand.py).
PC = np.array([0.080, 0.040, 0.118])
X, Y, Z = np.eye(3)


def Rz(t):
    return Rotation.from_rotvec([0, 0, t]).as_matrix()


def tilt_h(api):
    """Pen axis angle from horizontal [deg]."""
    _, R = api.object_pose("pen")
    return 90.0 - np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))


def pen_state(api):
    p, R = api.object_pose("pen")
    pp, Rop = api.object_in_palm("pen")
    f = api.contact_forces("pen")
    return f"tilt_h={tilt_h(api):.0f} z={p[2]:.3f} axis_palm={np.round(Rop[:, 2], 2).tolist()} F={np.round(f[:3], 2).tolist()}"


def watch(api, gen, tag, every=15):
    """Run a motion generator, logging the pen state every `every` steps."""
    k = 0
    for _ in gen:
        k += 1
        if k % every == 0:
            api.note(f"{tag} {pen_state(api)}")
        yield


def roll_about(api, xyz0, R0, qp, th0, th1, seconds, n=6):
    """Roll the palm about its own z axis through palm-frame point qp (kept fixed in the world)."""
    Pw = xyz0 + R0 @ qp
    for k in range(1, n + 1):
        th = th0 + (th1 - th0) * k / n
        Rk = R0 @ Rz(th)
        yield from api.carry(Pw - Rk @ qp, Rk, seconds / n)


def run(api):
    p0, R0 = api.object_pose("pen")
    a = R0[:, 2]
    ah = np.array([a[0], a[1], 0.0])
    ah /= np.linalg.norm(ah)
    if ah[1] < 0:
        ah = -ah
    try:
        pr, _ = api.object_pose("pen_rest")
        c_r = float((pr - p0) @ ah)
    except Exception as e:
        api.note(f"no rest pose: {e}")
        c_r = -0.012
    r = float(np.clip(p0[2] - 0.78, 0.003, 0.0046))
    # s=-1: long side toward palm -y (pinch at the +ah end); s=+1: long side toward +y.
    s = 1 if c_r > 0 else -1
    d = 0.033 if s > 0 else 0.048
    pinch_w = p0 - s * d * ah
    psi = float(np.arctan2(-ah[0], ah[1]))
    Rg = api.hand_down(psi)
    api.note(f"c_r={c_r:.4f} r={r:.4f} s={s} d={d} psi={psi:.3f} p0={np.round(p0, 4).tolist()}")

    grasp_xyz = pinch_w - Rg @ PC
    yield from api.open(0.6)
    yield from api.carry(grasp_xyz + [0, 0, 0.05], Rg, 2.0)
    yield from api.carry(grasp_xyz, Rg, 1.0)
    yield from api.hold(0.3)

    # Contact points from the measured pen pose in the palm frame.
    pp, Rop = api.object_in_palm("pen")
    u = Rop[:, 2] * (1 if Rop[1, 2] > 0 else -1)
    q = pp - s * d * u
    qm = q - 0.021 * u
    to_o = lambda x: Rop.T @ (x - pp)
    to_n = lambda n: Rop.T @ n
    # Middle presses DOWN on the top of the stub (palm -x is world up): its normal force, not
    # friction, balances the gravity torque about the pinch hinge, and it adds no yaw torque.
    pts = {"thumb": to_o(q - r * Z), "index": to_o(q + r * Z), "middle": to_o(qm - r * X)}
    nrm = {"thumb": to_n(-Z), "index": to_n(Z), "middle": to_n(-X)}
    api.note(f"pen in palm {np.round(pp, 4).tolist()} pinch {np.round(q, 4).tolist()}")

    res = yield from api.approach("pen", pts, standoff=0.012, seconds=1.2, normals=nrm)
    api.note(f"approach {res}")
    res = yield from api.grasp("pen", {f: pts[f] for f in ("thumb", "index")}, squeeze=0.004, seconds=0.8,
                               normals={f: nrm[f] for f in ("thumb", "index")})
    api.note(f"pinch {res}")
    res = yield from api.grasp("pen", {"middle": pts["middle"]}, squeeze=0.002, seconds=0.6,
                               normals={"middle": nrm["middle"]})
    api.note(f"middle {res}")
    yield from api.hold(0.2)

    # Level hold: lift the clamped pen ~110 mm.
    lift_xyz = grasp_xyz + [0, 0, 0.11]
    yield from watch(api, api.carry(lift_xyz, Rg, 1.5), "lift", 25)
    yield from api.hold(0.7)
    api.note(f"lifted {pen_state(api)}")

    # Move forward (palm z, perpendicular to the pen) to clear the rest, then lower the pinch to
    # 25 mm above the table so the long end can rest on the table once the middle lets go.
    tt = api.tip_targets()
    qp = 0.5 * (tt[0] + tt[1])
    fwd = Rg[:, 2] * np.array([1, 1, 0])
    fwd /= np.linalg.norm(fwd)
    xyz = lift_xyz + 0.045 * fwd
    yield from api.carry(xyz, Rg, 0.8)
    h = xyz + Rg @ qp
    xyz = xyz + [0, 0, (0.775 - h[2])]
    yield from watch(api, api.carry(xyz, Rg, 1.5), "lower", 25)
    api.note(f"lowered {pen_state(api)}")
    res = yield from api.release(["middle"], "pen", clear=0.0, lift=(-0.03, -0.004, 0.02), seconds=0.6)
    yield from api.hold(0.6)
    api.note(f"tip down {pen_state(api)}")

    # Raise the pinch along an arc about the resting tip: the pen turns in the thumb-index hinge
    # (relative to the palm) under gravity, quasi-statically, and leaves the table hanging vertical.
    p, R = api.object_pose("pen")
    h = xyz + Rg @ qp
    at = R[:, 2] if (p - h) @ R[:, 2] > 0 else -R[:, 2]
    sz = max(-at[2], 0.05)
    sd = float(np.clip((h[2] - (0.75 + r)) / sz, 0.07, 0.13))
    ct = h + sd * at
    e = np.array([at[0], at[1], 0.0])
    e /= np.linalg.norm(e)
    b0 = float(np.arcsin(np.clip((h[2] - ct[2]) / sd, 0.0, 0.9)))
    api.note(f"arc sd={sd:.3f} b0={np.degrees(b0):.0f} ct={np.round(ct, 3).tolist()}")
    n = 10
    for k in range(1, n + 1):
        b = b0 + (np.pi / 2 + 0.05 - b0) * k / n
        hk = ct - sd * np.cos(b) * e + [0, 0, sd * np.sin(b)]
        yield from api.carry(hk - Rg @ qp, Rg, 0.35)
        if k % 3 == 0:
            api.note(f"arc{k} {pen_state(api)}")
    hk = ct + [0, 0, max(sd, 0.10) + 0.09]
    yield from api.carry(hk - Rg @ qp, Rg, 1.5)
    yield from api.hold(0.5)
    api.note(f"up {pen_state(api)}")
    yield from api.hold(1.0)
    api.note(f"final {pen_state(api)}")
