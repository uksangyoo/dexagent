"""pen_gravity_pivot, force level (v2: palm-down horizontal pinch).

Plan (quasi-static gravity pivoting):
 1. Palm-down hand, fingers across the pen. Thumb (wrist side) and index (far side)
    pinch the pen's free overhang at the hand's natural pinch point; the pinch line
    is horizontal and normal to the pen, i.e. a frictionless hinge (pen contacts have
    no torsional friction). The middle presses the pen 22 mm further out, on the
    index side: its friction (lever 22 mm) holds the pen level against gravity.
    Thumb and middle are force servos, the index is a position anchor.
 2. Lift and hold level.
 3. Roll the hand -90 deg about the pinch line with the pen clamped: pen vertical.
 4. Loosen: ungrip the middle and back it off -> pen hangs from the thumb-index hinge.
 5. Roll the hand back about the hinge so the palm faces down again; gravity keeps
    the pen vertical, so it turns ~90 deg in the fingers.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

C = np.array([0.080, 0.040, 0.118])   # palm-frame pinch point (pen axis)
RR, PAD = 0.00375, 0.008              # pen radius, pad-centre inset
V = np.array([0.0, 0.0, 1.0])
SG = 0.032        # pinch point from the pen centre, toward the free end
DM = 0.022        # middle pad, further toward the free end
LIFT = 0.16
F_TH, F_MD = 1.8, 0.4
TRAVEL = 0.022


def smooth(u):
    return u * u * (3 - 2 * u)


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def hor(v):
    v = np.array(v, float)
    v[2] = 0
    return unit(v)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(unit(u) @ unit(v)), 0, 1))))


def heading(v):
    return float(np.arctan2(v[1], v[0]))


def pen_axis(api):
    p, R = api.object_pose("pen")
    return p, R[:, 2]


def palm_axis(api):
    return api.object_in_palm("pen")[1][:, 2]


def obj_dir(api, d_palm):
    """A palm-frame direction expressed in the pen frame (for grip directions)."""
    _, Rop = api.object_in_palm("pen")
    return Rop.T @ np.asarray(d_palm, float)


def rot_about(api, W, mid, R0, rotvec, seconds, monitor=None, refresh=None):
    """Rotate the commanded palm by `rotvec` (world) about world point W (palm-frame
    point `mid`), one arm step per control step. `refresh` (a generator function) runs
    every 5 steps. Returns the fraction done."""
    n = max(1, round(seconds / 0.02))
    rv = np.asarray(rotvec, float)
    for k in range(1, n + 1):
        u = smooth(k / n)
        R = Rot.from_rotvec(u * rv).as_matrix() @ R0
        yield from api.carry(W - R @ mid, R, seconds=0.02)
        if refresh is not None and k % 5 == 0:
            yield from refresh()
        if monitor is not None and monitor(u):
            return u
    return 1.0


def run(api):
    notes = []
    try:
        yield from _run(api, notes)
    except Exception as ex:
        notes.append(f"EXC {type(ex).__name__}: {ex}")
        raise
    finally:
        for s in notes:
            api.note(s)


def _run(api, N):
    note = N.append
    f2 = lambda v: np.round(np.asarray(v) * 1000, 1).tolist()
    p0, a0 = pen_axis(api)
    a = hor(a0)
    try:
        pr, _ = api.object_pose("pen_rest")
        sr = float((pr - p0) @ a)
    except Exception as ex:
        sr = 0.012
        note(f"rest pose unavailable {ex}")
    e = -a if sr > 0 else a          # the end with the long free overhang
    G = p0 + SG * e
    # palm down (x = -V), palm y = -e (middle, at -y, goes toward the free end), fingers z = V x e
    R0 = np.column_stack([-V, -e, np.cross(V, e)])
    note(f"start pen {np.round(p0, 3).tolist()} axis {np.degrees(heading(a)):.1f} rest_s {sr * 1000:.1f}mm "
         f"fingers heading {np.degrees(heading(R0[:, 2])):.1f} det {np.linalg.det(R0):.3f}")

    zc = RR + PAD
    # Pinch point along the fingers: the thumb must be able to reach 6 mm past the pen surface
    # (servo headroom), the index and middle the far surface.
    global C
    C0 = C.copy()
    scan = []
    for dz in (0.0, -0.003, -0.006, -0.009, -0.012, -0.015):
        c = C0 + np.array([0, 0, dz])
        r_t = api.reach("thumb", c + np.array([0, 0, -(zc - 0.006)]))
        r_i = api.reach("index", c + np.array([0, 0, zc]))
        r_m = api.reach("middle", c + np.array([0, -DM, zc - 0.002]))
        scan.append((round(dz * 1000), round(r_t * 1000, 1), round(r_i * 1000, 1), round(r_m * 1000, 1)))
        if max(r_t, r_i, r_m) < 0.0015:
            C = c
            break
    else:
        C = C0 + np.array([0, 0, min(scan, key=lambda x: max(x[1:]))[0] / 1000])
    note(f"pinch scan dz,thumb,index,middle residual mm {scan} -> C {f2(C)}")
    T_th = C + np.array([0, 0, -(zc + 0.004)])
    T_ix = C + np.array([0, 0, zc - 0.0015])   # index preloaded 1.5 mm into the pen: an anchor that pushes back
    T_md = C + np.array([0, -DM, zc + 0.004])
    yield from api.tuck(["ring", "little"], seconds=0.5)
    r = yield from api.move_tips(api.palm_for(G + 0.06 * V, C, R0), R0,
                                 tips={"thumb": T_th, "index": T_ix, "middle": T_md}, seconds=3.0)
    note(f"preshape ik {r.get('ik_residual_mm')}")
    yield from api.carry(api.palm_for(G, C, R0), R0, seconds=1.5)
    yield from api.hold(0.2)
    pp, Rp = api.object_in_palm("pen")
    note(f"at grasp: pen in palm {f2(pp)} axis {np.round(Rp[:, 2], 2).tolist()} tips {f2(api.tips()[:3])}")

    def regrip(seconds=0.02, fingers=("thumb", "middle")):
        F = {"thumb": F_TH, "middle": F_MD}
        D = {"thumb": obj_dir(api, [0, 0, 1]), "middle": obj_dir(api, [0, 0, -1])}
        return (yield from api.grip("pen", {f: F[f] for f in fingers}, directions={f: D[f] for f in fingers},
                                    seconds=seconds, max_travel=TRAVEL))
    g = yield from regrip(1.5)
    note(f"grip {g}")

    P, Rc = api.palm_pose()
    yield from api.carry(P + LIFT * V, Rc, seconds=2.5)
    yield from api.hold(0.6)
    pc, ax = pen_axis(api)
    a_lvl = palm_axis(api)
    note(f"after lift: tilt_h {90 - ang(ax, V):.1f} lift {1000 * (pc[2] - p0[2]):.0f}mm "
         f"forces {np.round(api.contact_forces('pen'), 2).tolist()} grip {api.grip_state()}")

    # Clamped roll about the pinch line: the pen's centre side (palm +y) goes down.
    P, Rc = api.palm_pose()
    t = api.tips()
    mid = 0.5 * (t[0] + t[1])
    W = P + Rc @ mid
    log = []

    def mon1(u):
        if len(log) < 4 and u > 0.2 * (len(log) + 1):
            _, ax = pen_axis(api)
            log.append((round(u, 2), round(ang(palm_axis(api), a_lvl), 1), np.round(api.contact_forces('pen')[:3], 2).tolist()))
        return False
    yield from rot_about(api, W, mid, Rc, Rc @ np.array([0, 0, -np.pi / 2]), seconds=5.0, monitor=mon1, refresh=regrip)
    note(f"clamped roll (u, pivot, forces): {log}")
    yield from api.hold(0.4)
    pc, ax = pen_axis(api)
    note(f"after clamped roll: tilt_v {ang(ax, V):.1f} pivot {ang(palm_axis(api), a_lvl):.1f} "
         f"lift {1000 * (pc[2] - p0[2]):.0f}mm forces {np.round(api.contact_forces('pen'), 2).tolist()}")

    # Loosen: the middle lets go, the pen hangs from the thumb-index hinge.
    yield from api.ungrip(["middle"])
    tt = api.tip_targets()
    yield from api.move_tips(tips={"middle": tt[2] + np.array([0, 0, 0.015])}, seconds=0.5, straight=True)
    g = yield from regrip(0.3, ("thumb",))
    yield from api.hold(0.5)
    for _ in range(20):          # let any swing die out (the pen hangs from the hinge)
        _, ax = pen_axis(api)
        if ang(ax, V) < 6.0:
            break
        yield from regrip(0.1, ("thumb",))
    _, ax = pen_axis(api)
    note(f"middle released: tilt_v {ang(ax, V):.1f} grip {g}")

    # Roll back about the hinge until the palm faces down.
    P, Rc = api.palm_pose()
    t = api.tips()
    mid = 0.5 * (t[0] + t[1])
    k = Rc @ unit(t[1] - t[0])
    W = P + Rc @ mid
    x0 = Rc[:, 0]
    A = -((x0 - k * (k @ x0)) @ V)
    B = -(np.cross(k, x0) @ V)
    phi = float(np.arctan2(B, A))
    note(f"hinge axis {np.round(k, 2).tolist()} roll {np.degrees(phi):.0f}deg")
    worst = [0.0]

    def mon(u):
        _, ax = pen_axis(api)
        tv = ang(ax, V)
        pv = ang(palm_axis(api), a_lvl)
        worst[0] = max(worst[0], tv)
        return tv > 12.0 and pv > 72.0
    u = yield from rot_about(api, W, mid, Rc, k * phi, seconds=4.0, monitor=mon,
                             refresh=lambda: regrip(0.02, ("thumb",)))
    yield from api.hold(1.2)
    pc, ax = pen_axis(api)
    note(f"after roll (u={u:.2f}, max tilt_v {worst[0]:.1f}): tilt_v {ang(ax, V):.1f} pivot {ang(palm_axis(api), a_lvl):.1f} "
         f"lift {1000 * (pc[2] - p0[2]):.0f}mm forces {np.round(api.contact_forces('pen'), 2).tolist()} grip {api.grip_state()}")
