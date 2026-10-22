"""Scissors v3: index pad deep in hole A, ring pad deep in hole B, closing squeeze
(index pushes A's inner wall -y, ring pushes B's inner wall +y; the closed stop carries
the load), thumb on A's outer face. Lift. If the pitch torque wins the scissors hang
from the y-directed contacts like an axle, which is fine (blade tips stay high).
Cycles: index relocates to A's outer wall (+y) so A is pinched by thumb+index; the ring
pushes B's outer wall and rotates about the pivot to open, then B's inner wall to close.
Pads are placed by a measure-and-correct loop before they go down into the holes.
"""
import numpy as np

OBJ = "scissors"
TABLE = 0.75
H_NOM = 0.00465       # settled pivot height at nominal size
X_HOLE = -0.065
X_PAD = -0.060        # pad x inside the holes (front part of the hole: shorter lever to the COM)
Y_HOLE = 0.024
Y_INNER = 0.0105      # hole-side face of the inner wall
Y_OUTER_IN = 0.0375   # hole-side face of the outer wall
Y_OUTER = 0.0425      # outside face of the outer wall
X_THUMB = -0.065

_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]
SUMMARY = []
FI = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
UP = np.array([0, 0, 1.0])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def clearance(p, R, a):
    pts = np.vstack([_A, _B @ Rz(a).T]) @ R.T + p
    return float(pts[:, 2].min() - abs(R[2, 2]) * 0.0055 - TABLE)


def hinge(api):
    return api.object_joint("scissors_hinge")


def pads_obj(api):
    """Measured pad centres in the scissors (part A) frame, rows = fingers."""
    p, R = api.object_in_palm(OBJ)
    return (api.tips() - p) @ R


def status(api, tag):
    p, R = api.object_pose(OBJ)
    a = hinge(api)
    f = np.round(api.contact_forces(OBJ), 1).tolist()
    c = clearance(p, R, a)
    q = np.round(pads_obj(api)[[0, 1, 3]] * 1000, 1).tolist()
    SUMMARY.append(f"{tag}:h={np.degrees(a):.1f} c={1000 * c:.0f} z={1000 * (p[2] - TABLE):.1f} zR={R[2, 2]:.2f} f={f}")
    api.note(f"[{tag}] hinge={np.degrees(a):.1f} clear={1000 * c:.1f}mm piv_z={1000 * (p[2] - TABLE):.1f} "
             f"zR={R[2, 2]:.3f} f={f} pads(t,i,r)={q}")
    return p, R, a, c


def goto(api, targets, seconds=0.8):
    """Pad centres to object-frame points (straight lines), no squeeze."""
    pts = {f: np.asarray(q, float) - UP * api.pad_offset for f, q in targets.items()}
    r = yield from api.grasp(OBJ, pts, squeeze=0.0, seconds=seconds, normals={f: UP for f in targets})
    return r


def z_axis_palm(api):
    _, R = api.object_in_palm(OBJ)
    return R[:, 2]


def b_wall(api, s, which, finger="ring", z=None):
    """Point and outward normal (part A frame) on part B's hole-side wall face, at the pad's x/z."""
    a = hinge(api)
    q = Rz(-a) @ pads_obj(api)[FI[finger]]
    if z is not None:
        q[2] = z
    if which == "outer":
        pt, n = np.array([q[0], -Y_OUTER_IN * s, q[2]]), np.array([0, 1.0, 0])
    else:
        pt, n = np.array([q[0], -Y_INNER * s, q[2]]), np.array([0, -1.0, 0])
    return Rz(a) @ pt, Rz(a) @ n


def run(api):
    yield from api.hold(0.6)                         # let the scissors settle
    p0, R0 = api.object_pose(OBJ)
    psi = float(np.arctan2(R0[1, 0], R0[0, 0]))
    h0 = p0[2] - TABLE
    s = float(np.clip(h0 / H_NOM, 0.88, 1.12))       # size from the resting pivot height
    Rp = api.hand_down(psi)
    zlo = TABLE + 0.0075 - p0[2]                     # pad centre 7.5 mm above the table
    zhi = TABLE + 0.011 * s + 0.014 - p0[2]          # pad bottom ~5 mm above the wall tops
    zth = TABLE + 0.010 - p0[2]
    xp, xt = X_PAD * s, X_THUMB * s
    yth = Y_OUTER * s + 0.016
    low = {"thumb": np.array([xt, yth, zth]), "index": np.array([xp, Y_HOLE * s, zlo]),
           "ring": np.array([xp, -Y_HOLE * s, zlo])}
    high = {f: v + np.array([0, 0, zhi - zlo]) for f, v in low.items()}

    # ---- placement search over the pivot position o in the palm frame; palm = (-z_obj, y_obj, x_obj) + o
    def P(o, q):
        return np.array([o[0] - q[2], o[1] + q[1], o[2] + q[0]])

    extra = {"index": [np.array([xp, Y_OUTER_IN * s - 0.004, zlo])],
             "ring": [Rz(0.35) @ np.array([xp, -Y_OUTER_IN * s - 0.004, zlo]),
                      np.array([xp, -Y_INNER * s - 0.004, zlo])]}
    best = None
    for hx in (0.07, 0.08, 0.09, 0.10, 0.11):
        for oy in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02):
            for oz in (0.13, 0.15, 0.17, 0.19, 0.21):
                o = np.array([hx + zlo, oy, oz])
                r = [api.reach(f, P(o, q)) for f, q in low.items()]
                r += [api.reach(f, P(o, q)) for f, ql in extra.items() for q in ql]
                r += [api.reach(f, P(o, q)) for f, q in high.items()]
                score = max(r) + 0.3 * np.mean(r)
                if best is None or score < best[0]:
                    best = (score, o, r)
    _, o, r = best
    SUMMARY.append(f"h0={1000 * h0:.2f} s={s:.3f} o={np.round(o, 3).tolist()} r={np.round(np.array(r) * 1000, 1).tolist()}")
    api.note(SUMMARY[-1])
    X_final = p0 - Rp @ o

    # ---- pre-shape above, come down, correct pad placement by measurement
    yield from api.tuck(["middle", "little"], seconds=0.5)
    lift0 = 0.04
    yield from api.carry(X_final + np.array([0, 0, lift0]), Rp, seconds=2.0)
    yield from goto(api, {f: v + np.array([0, 0, lift0]) for f, v in high.items()}, seconds=1.0)
    yield from api.carry(X_final, Rp, seconds=1.0)
    yield from api.hold(0.2)
    corr = {f: np.zeros(3) for f in low}
    for k in range(2):
        q = pads_obj(api)
        for f in low:
            e = q[FI[f]] - high[f]
            corr[f] -= np.array([e[0], e[1], 0.0])
        api.note(f"place{k} err_mm=" + str({f: np.round((q[FI[f]] - high[f]) * 1000, 1).tolist() for f in low}))
        yield from goto(api, {f: high[f] + corr[f] for f in low}, seconds=0.5)
        yield from api.hold(0.15)
    status(api, "above")

    # ---- into the holes
    for attempt in range(2):
        yield from goto(api, {f: low[f] + corr[f] for f in low}, seconds=0.8)
        yield from api.hold(0.25)
        q = pads_obj(api)
        bad = [f for f in ("index", "ring") if q[FI[f]][2] > TABLE + 0.011 * s + 0.006 - p0[2]]
        status(api, f"in{attempt}")
        if not bad:
            break
        api.note(f"pads not in holes: {bad}; retry")
        yield from goto(api, {f: high[f] + corr[f] for f in low}, seconds=0.5)
        q = pads_obj(api)
        for f in low:
            e = q[FI[f]] - high[f]
            corr[f] -= np.array([e[0], e[1], 0.0])

    # ---- closing squeeze: index on A's inner wall, ring on B's inner wall, thumb on A's outer face
    def squeeze_grip(sq):
        q = pads_obj(api)
        zi, zr = min(q[1][2], zlo + 0.002) - 0.0015, min(q[3][2], zlo + 0.002) - 0.0015
        pts = {"index": np.array([q[1][0], Y_INNER * s, zi]),
               "ring": np.array([q[3][0], -Y_INNER * s, zr])}
        nrm = {"index": np.array([0, 1.0, 0]), "ring": np.array([0, -1.0, 0])}
        r = yield from api.grasp(OBJ, pts, squeeze=sq, seconds=0.8, normals=nrm)
        # thumb only lightly
        return r

    r = yield from squeeze_grip(0.02)
    yield from api.hold(0.3)
    api.note(f"squeeze {r}")
    status(api, "squeezed")

    # ---- lift test, then lift
    for attempt in range(2):
        X, _ = api.palm_pose()
        yield from api.carry(X + np.array([0, 0, 0.015]), Rp, seconds=0.6)
        yield from api.hold(0.2)
        p, R, a, c = status(api, f"lifttest{attempt}")
        if p[2] - p0[2] > 0.005:
            break
        yield from api.carry(X, Rp, seconds=0.6)
        yield from api.hold(0.2)
        r = yield from squeeze_grip(0.02)
        yield from api.hold(0.3)
        api.note(f"resqueeze {r}")
    X, _ = api.palm_pose()
    yield from api.carry(X + np.array([0, 0, 0.13]), Rp, seconds=2.2)
    yield from api.hold(0.4)
    p, R, a, c = status(api, "lifted")
    if c < 0.035 and p[2] - p0[2] > 0.02:
        X, _ = api.palm_pose()
        yield from api.carry(X + np.array([0, 0, 0.08]), Rp, seconds=1.2)
        yield from api.hold(0.3)
        p, R, a, c = status(api, "lifted2")

    # ---- in the air the table no longer blocks the fingertips: work at mid-wall height
    zmid = 0.001
    # thumb onto A's outer face (pushes -y), mid-wall height: first straight down OUTSIDE the wall
    # (iter 4: a diagonal path hit the wall top and stripped the scissors out of the grip)
    q = pads_obj(api)
    yth_out = max(q[0][1], Y_OUTER * s + 0.008 + 0.006)
    yield from goto(api, {"thumb": np.array([q[0][0], yth_out, q[0][2]])}, seconds=0.3)
    yield from goto(api, {"thumb": np.array([X_THUMB * s, yth_out, zmid])}, seconds=0.6)
    status(api, "thumbdown")
    thumb_ok = False
    for sq in (0.011, 0.018):
        r = yield from api.grasp(OBJ, {"thumb": np.array([X_THUMB * s, Y_OUTER * s, zmid])}, squeeze=sq,
                                 seconds=0.6, normals={"thumb": np.array([0, 1.0, 0])})
        yield from api.hold(0.2)
        api.note(f"thumb on A sq={sq} {r}")
        p, R, a, c = status(api, "thumbA")
        if api.contact_forces(OBJ)[0] > 1.0:
            thumb_ok = True
            break
    # middle under the inner walls (a shelf near the COM line y=0), if reachable
    def Pm(qo):
        po, Ro = api.object_in_palm(OBJ)
        return po + Ro @ qo
    path = [np.array([-0.106 * s, 0.0, 0.025]), np.array([-0.106 * s, 0.0, -0.0165]),
            np.array([-0.070 * s, 0.0, -0.0115])]
    rr = [api.reach("middle", Pm(qo)) for qo in path]
    api.note(f"middle shelf reach_mm={np.round(np.array(rr) * 1000, 1).tolist()}")
    if max(rr) < 0.004:
        for qo, sec in zip(path, (0.6, 0.5, 0.6)):
            yield from goto(api, {"middle": qo}, seconds=sec)
        yield from api.hold(0.2)
        status(api, "shelf")
    # index to A's outer wall (pushes +y) against the thumb -- only once the thumb is confirmed
    if thumb_ok:
        q = pads_obj(api)
        r = yield from api.grasp(OBJ, {"index": np.array([q[1][0], Y_OUTER_IN * s, zmid])},
                                 squeeze=0.011, seconds=0.8, normals={"index": np.array([0, -1.0, 0])})
        yield from api.hold(0.3)
        api.note(f"A pinch {r}")
        p, R, a, c = status(api, "Apinch")
    else:
        api.note("thumb never made contact; skip cycles to keep the grip")

    # ---- open / close cycles with the ring finger in hole B
    done = 0
    for k in range(3 if thumb_ok else 0):
        if done >= 2 or api.t > 44:
            break
        p, R = api.object_pose(OBJ)
        if clearance(p, R, hinge(api)) < 0.03:
            api.note("too low for a cycle; abort cycles")
            break
        pt, n = b_wall(api, s, "outer", z=zmid)
        yield from api.grasp(OBJ, {"ring": pt}, squeeze=0.006, seconds=0.6, normals={"ring": n})
        r = yield from api.rotate(OBJ, z_axis_palm(api), 0.32, ["ring"], seconds=1.0)
        yield from api.hold(0.2)
        p, R, a, c = status(api, f"open{k}")
        if np.degrees(a) < 13:
            r = yield from api.rotate(OBJ, z_axis_palm(api), 0.22, ["ring"], seconds=0.7)
            yield from api.hold(0.2)
            p, R, a, c = status(api, f"open{k}b")
        opened = np.degrees(a) > 10.5 and c > 0.02
        api.note(f"open ik {r.get('ik_residual_mm')}")
        pt, n = b_wall(api, s, "inner", z=zmid)
        yield from api.grasp(OBJ, {"ring": pt}, squeeze=0.006, seconds=0.6, normals={"ring": n})
        a = hinge(api)
        r = yield from api.rotate(OBJ, z_axis_palm(api), -(a + 0.12), ["ring"], seconds=1.0)
        yield from api.hold(0.2)
        p, R, a, c = status(api, f"close{k}")
        if np.degrees(a) > 2.0:
            pt, n = b_wall(api, s, "inner", z=zmid)
            yield from api.grasp(OBJ, {"ring": pt}, squeeze=0.004, seconds=0.4, normals={"ring": n})
            yield from api.rotate(OBJ, z_axis_palm(api), -(a + 0.12), ["ring"], seconds=0.6)
            yield from api.hold(0.2)
            p, R, a, c = status(api, f"close{k}b")
        if opened and np.degrees(a) < 2.8 and c > 0.02:
            done += 1
        api.note(f"cycle {k}: counted={done}")

    # ---- keep it closed: ring pushes B's inner wall
    pt, n = b_wall(api, s, "inner", z=zmid)
    yield from api.grasp(OBJ, {"ring": pt}, squeeze=0.008, seconds=0.5, normals={"ring": n})

    # ---- set down: lower (and drift back if pitched so it lies down blades-forward)
    back = -R0[:, 0] * np.array([1, 1, 0])
    back /= np.linalg.norm(back) + 1e-9
    for k in range(10):
        if api.t > 54 or api.tips_world()[[0, 1, 3], 2].min() < TABLE + 0.0085:
            api.note("stop lowering: pads at the table or out of time")
            break
        p, R = api.object_pose(OBJ)
        a = hinge(api)
        c = clearance(p, R, a)
        hz = p[2] - TABLE
        flat = abs(R[2, 2]) > 0.97
        if flat and hz < 0.0075:
            break
        X, _ = api.palm_pose()
        if flat:
            dz = min(hz - 0.0045 * s / 1.0, 0.10)
            yield from api.carry(X - np.array([0, 0, max(dz, 0.002)]), Rp, seconds=max(0.5, dz * 15))
        else:
            dz = float(np.clip(c - 0.002, 0.008, 0.06))
            dx = 0.012 * float(np.sqrt(max(0.0, 1 - R[2, 2] ** 2)))
            yield from api.carry(X - np.array([0, 0, dz]) + back * dx, Rp, seconds=max(0.5, dz * 15))
        yield from api.hold(0.2)
        if k % 2 == 0:
            status(api, f"down{k}")
    yield from api.hold(0.3)
    p, R, a, c = status(api, "lowered")
    if np.degrees(a) > 2.0:
        pt, n = b_wall(api, s, "inner")
        yield from api.grasp(OBJ, {"ring": pt}, squeeze=0.004, seconds=0.4, normals={"ring": n})
        yield from api.rotate(OBJ, z_axis_palm(api), -(a + 0.1), ["ring"], seconds=0.6)
        status(api, "reclosed")

    # ---- release: pads to the hole centres / away from the face, then palm up
    q = pads_obj(api)
    a = hinge(api)
    tgt = {"index": np.array([q[1][0], Y_HOLE * s, q[1][2] + 0.002]),
           "ring": Rz(a) @ np.array([(Rz(-a) @ q[3])[0], -Y_HOLE * s, q[3][2] + 0.002]),
           "thumb": q[0] + np.array([0, 0.012, 0.004])}
    yield from goto(api, tgt, seconds=0.6)
    X, _ = api.palm_pose()
    yield from api.carry(X + np.array([0, 0, 0.03]), Rp, seconds=0.6)
    X, _ = api.palm_pose()
    yield from api.carry(X + np.array([0, 0, 0.06]), Rp, seconds=0.8)
    yield from api.hold(0.3)
    status(api, "released")
    api.note("SUMMARY " + " | ".join(SUMMARY))
