"""card_pickup, contacts level.

Plan: (1) palm down, press the index pad on the tile centre and drag the tile
toward the robot (-y) until its centre is just inside the book's near edge,
so its near end overhangs; (2) droop the hand (fingers down, palm facing -x,
the thumb-index pinch axis vertical and the index finger in a plane parallel
to the book edge), slide in from -y with the pinch open around the overhang,
pinch thumb-on-top / index-below, (3) lift.
"""
import numpy as np

PC = np.array([0.080, 0.040, 0.118])      # thumb-index pinch centre, palm frame (hand.py)
RA = np.diag([-1.0, 1.0, -1.0])           # fingers down, palm face -x, thumb +y
T_HALF = 0.0015                           # tile half thickness


def r3(v):
    return np.round(np.asarray(v, float) * 1000).astype(int).tolist()


def pads_world(api, targets, seconds, xyz=None, R=None):
    """Place pads at world points (as seen from the current palm) via approach with virtual tile points."""
    c, Rt = api.object_pose("tile")
    n = np.array([0, 0, 1.0])
    pts = {f: Rt.T @ (np.asarray(W, float) - c) - n * api.pad_offset for f, W in targets.items()}
    return (yield from api.approach("tile", pts, standoff=0.0, seconds=seconds,
                                    normals={f: n for f in targets}, xyz=xyz, R=R))


def pads_palm(api, targets, seconds, xyz=None, R=None):
    """Pads to palm-frame points (kept in the palm frame while the palm moves to xyz, R)."""
    P, Rp = api.measured_palm()
    W = {f: P + Rp @ np.asarray(v, float) for f, v in targets.items()}
    return (yield from pads_world(api, W, seconds, xyz, R))


def run(api):
    c0, Rt0 = api.object_pose("tile")
    book, _ = api.object_pose("book")
    edge = book[1] - 0.110
    sz = 1.0 if Rt0[2, 2] > 0 else -1.0
    tips0 = api.tips()
    api.note(f"tile0 {r3(c0)} Rt0 {np.round(Rt0, 2).tolist()} book {r3(book)} edge {edge*1000:.0f} "
             f"tips0 {r3(tips0)}")

    Rd = api.hand_down(0.7)
    # ---- index press point: a moderately flexed index, pad facing down
    yi = tips0[1][1]
    cands = [(0.05, yi, 0.17), (0.05, yi, 0.16), (0.06, yi, 0.15), (0.06, yi, 0.14),
             (0.07, yi, 0.13), (0.08, 0.04, 0.13)]
    res = [api.reach("index", np.array(q)) for q in cands]
    good = [q for q, e in zip(cands, res) if e < 0.002]
    Q = np.array(good[0] if good else cands[int(np.argmin(res))])
    press_tips = {"index": Q}
    th = tips0[0]
    if th[0] > 0.035:  # thumb hangs low: lift it toward the palm face
        tq = np.array([0.02, th[1], th[2]])
        if api.reach("thumb", tq) < 0.004:
            press_tips["thumb"] = tq
    api.note(f"press Q {r3(Q)} reach {[round(e*1000,1) for e in res]} thumb {r3(th)}")

    # ---- phase 1: drag the tile toward the near edge
    target_y = edge + 0.009
    for attempt in range(4):
        c, Rt = api.object_pose("tile")
        D = c[1] - target_y
        if -0.004 < D < 0.004:
            break
        W = c + np.array([0, 0, T_HALF + api.pad_offset + 0.02])
        O = api.palm_for(W, Q, Rd)
        yield from pads_palm(api, press_tips, 1.5, xyz=O + np.array([0, 0, 0.04]), R=Rd)
        yield from api.carry(O, Rd, 0.8)
        sq = 0.006
        r = yield from api.grasp("tile", {"index": np.array([0, 0, sz * T_HALF])}, squeeze=sq, seconds=0.8,
                                 normals={"index": np.array([0, 0, sz])})
        f = api.contact_forces("tile")
        while f[1] < 1.0 and sq < 0.02:
            sq += 0.004
            yield from api.grasp("tile", {"index": np.array([0, 0, sz * T_HALF])}, squeeze=sq, seconds=0.4,
                                 normals={"index": np.array([0, 0, sz])})
            f = api.contact_forces("tile")
        while f[1] > 4.0 and sq > 0.0:
            sq -= 0.003
            yield from api.grasp("tile", {"index": np.array([0, 0, sz * T_HALF])}, squeeze=sq, seconds=0.3,
                                 normals={"index": np.array([0, 0, sz])})
            f = api.contact_forces("tile")
        api.note(f"press#{attempt} sq {sq*1000:.0f}mm forces {np.round(f, 2).tolist()}")
        c, _ = api.object_pose("tile")
        D = c[1] - target_y
        Pc, _ = api.palm_pose()
        yield from api.carry(Pc + np.array([0, -D, 0]), Rd, max(1.0, abs(D) / 0.05))
        c, _ = api.object_pose("tile")
        api.note(f"after drag#{attempt}: tile {r3(c)} y-edge {(c[1]-edge)*1000:.1f}mm forces "
                 f"{np.round(api.contact_forces('tile'), 2).tolist()}")
        # relieve the finger's lateral deflection, then lift straight up (release() drags the tile ~5 mm)
        Pc, _ = api.palm_pose()
        back = 0.003 * np.sign(D)
        yield from api.carry(Pc + np.array([0, back, 0]), Rd, 0.3)
        yield from api.carry(Pc + np.array([0, back, 0.05]), Rd, 0.8)
        c, _ = api.object_pose("tile")
        api.note(f"after lift#{attempt}: y-edge {(c[1]-edge)*1000:.1f}mm")

    # ---- phase 2: droop the hand and pinch the overhang
    Pc, _ = api.palm_pose()
    yield from api.carry(np.array([Pc[0], Pc[1], max(Pc[2], 1.0)]), Rd, 0.8)
    c, Rt = api.object_pose("tile")
    api.note(f"phase2 tile {r3(c)} y-edge {(c[1]-edge)*1000:.1f}mm tilt R22 {Rt[2,2]:.3f}")
    hw0 = 0.02
    yield from api.tuck(["middle", "ring", "little"], 0.6)
    r = yield from pads_palm(api, {"thumb": PC - np.array([0, 0, hw0]), "index": PC + np.array([0, 0, hw0])}, 1.0)
    api.note(f"open pinch ik {r.get('ik_residual_mm')}")
    P = c + np.array([0, -0.033, 0])
    Og = api.palm_for(P, PC, RA)
    Opre = Og + np.array([0, -0.07, 0])
    yield from api.carry(Opre + np.array([0, 0, 0.12]), RA, 2.5)
    yield from api.carry(Opre, RA, 1.2)
    yield from api.carry(Og, RA, 1.5)
    c, Rt = api.object_pose("tile")
    tw = api.tips_world()
    api.note(f"at grasp: tile {r3(c)} P {r3(P)} thumb {r3(tw[0])} index {r3(tw[1])} "
             f"f {np.round(api.contact_forces('tile'), 2).tolist()} book f {np.round(api.contact_forces('book'), 2).tolist()}")
    loc = Rt.T @ (P - c)
    ez = np.array([0, 0, sz])
    pts = {"thumb": loc * np.array([1, 1, 0]) + ez * T_HALF, "index": loc * np.array([1, 1, 0]) - ez * T_HALF}
    nrm = {"thumb": ez, "index": -ez}
    yield from api.approach("tile", pts, standoff=0.006, seconds=0.5, normals=nrm)
    sq = 0.004
    r = yield from api.grasp("tile", pts, squeeze=sq, seconds=0.8, normals=nrm)
    f = api.contact_forces("tile")
    if min(f[0], f[1]) < 0.8:
        sq = 0.007
        r = yield from api.grasp("tile", pts, squeeze=sq, seconds=0.5, normals=nrm)
        f = api.contact_forces("tile")
    api.note(f"pinch sq {sq*1000:.0f} forces {np.round(f, 2).tolist()} ik {r.get('ik_residual_mm')}")
    yield from api.carry(Og + np.array([0, 0, 0.17]), RA, 3.0)
    yield from api.hold(0.3)
    c, Rt = api.object_pose("tile")
    tw = api.tips_world()
    api.note(f"lifted: tile {r3(c)} lift {(c[2]-c0[2])*1000:.0f}mm R22 {Rt[2,2]:.2f} pinch-mid {r3((tw[0]+tw[1])/2)} "
             f"f {np.round(api.contact_forces('tile'), 2).tolist()}")
