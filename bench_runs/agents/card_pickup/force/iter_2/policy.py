"""card_pickup, force level.

Plan
 1. Palm down: index + middle pads press on the tile top under the grip-force servo
    (direction = into the tile, -z), and the palm drags the tile toward the robot until its
    centre sits ~8 mm behind the book's near edge (closed loop on the measured tile pose).
    The tile then overhangs the edge by ~30-45 mm but is still stable on its own.
 2. Release, rotate the hand to fingers-down / palm facing the book (+y), so the measured
    thumb-index pinch axis (palm z, hand.py) is vertical: thumb above, index hooked below.
    Open pinch, come in from the robot side under/over the overhang (guarded against book
    contact), close to near contact, then grip with the force servo along the pinch line.
 3. Pull the tile off the book and lift ~150 mm with the servo holding the pinch force.
"""
import numpy as np

C = np.array([0.080, 0.040, 0.118])   # thumb-index pinch centre, palm frame (from hand.py)
EZ = np.array([0.0, 0.0, 1.0])
BOOK_HALF_Y = 0.11
PAD = 0.008                            # pad centre to fingertip surface


def mm(v):
    return np.round(np.asarray(v, float) * 1000).astype(int).tolist()


def R_hook():
    # palm x (face) -> +y world (toward the book), palm z (fingers) -> down, palm y (thumb) -> +x
    return np.column_stack([[0, 1.0, 0], [1.0, 0, 0], [0, 0, -1.0]])


def guarded(api, gen, stop):
    """Run a motion generator step by step; stop early when stop() is true."""
    for _ in gen:
        yield
        if stop():
            return True
    return False


def run(api):
    pt, Rt = api.object_pose("tile")
    pb, _ = api.object_pose("book")
    y_edge = pb[1] - BOOK_HALF_Y
    top = pt[2] + 0.0015
    yaw = np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0]))
    api.note(f"tile {mm(pt)} yaw {yaw:.1f} book {mm(pb)} edge_y {y_edge * 1000:.0f}")
    api.note(f"start tips palm {mm(api.tips())}")

    # ------------------------------------------------ phase 1: press and drag
    Rd = api.hand_down()
    yield from api.move_joints(q=api.OPEN.copy(), seconds=0.6)
    yield from api.tuck(["ring", "little"], seconds=0.5)
    pads = {"index": np.array([0.080, 0.040, 0.138]), "middle": np.array([0.080, 0.018, 0.138])}
    r = yield from api.move_tips(tips=pads, seconds=0.6)
    tt = api.tip_targets()
    api.note(f"drag shape ik {r['ik_residual_mm']} tips {mm(tt)}")
    if tt[0][0] > 0.05:  # thumb hangs too low: lift it out of the way
        best = None
        for x in (0.0, 0.015, 0.03, 0.045):
            for y in (0.05, 0.07, 0.09):
                for z in (0.03, 0.05, 0.07, 0.09):
                    e = api.reach("thumb", [x, y, z])
                    if best is None or e < best[0]:
                        best = (e, [x, y, z])
        api.note(f"thumb out of the way at {best}")
        yield from api.move_tips(tips={"thumb": best[1]}, seconds=0.5)

    mid = (pads["index"] + pads["middle"]) / 2
    target = np.array([pt[0], pt[1], top + PAD + 0.004])
    P1 = api.palm_for(target, mid, Rd)
    yield from api.carry(P1 + [0, 0, 0.06], Rd, seconds=1.6)
    yield from api.carry(P1, Rd, seconds=0.8)
    g = yield from api.grip("tile", {"index": 1.2, "middle": 1.2},
                            directions={"index": [0, 0, -1.0], "middle": [0, 0, -1.0]},
                            seconds=1.5, max_travel=0.015)
    api.note(f"press grip {g} book {np.round(api.contact_forces('book'), 2).tolist()}")

    y_goal = y_edge + 0.008
    for k in range(4):
        pt, _ = api.object_pose("tile")
        err = pt[1] - y_goal
        api.note(f"drag {k}: tile y {pt[1] * 1000:.1f} goal {y_goal * 1000:.1f} grip {api.grip_state()}")
        if abs(err) < 0.002:
            break
        P, _ = api.palm_pose()
        yield from api.carry(P - [0, err, 0], seconds=max(0.6, abs(err) / 0.04))
        yield from api.hold(0.2)
    yield from api.ungrip()
    P, _ = api.palm_pose()
    yield from api.carry(P + [0, 0, 0.05], seconds=0.6)
    yield from api.hold(0.3)
    pt, Rt = api.object_pose("tile")
    api.note(f"after drag tile {mm(pt)} tilt {np.degrees(np.arccos(min(1, abs(Rt[2, 2])))):.1f} overhang_centre {1000 * (y_edge - pt[1]):.1f}")

    # ------------------------------------------------ phase 2: hook pinch from the near side
    Rp = R_hook()
    G = np.array([pt[0], y_edge - 0.018, pt[2]])
    P_pinch = api.palm_for(G, C, Rp)
    P_pre = P_pinch + [0, -0.08, 0]
    P_air = P_pre + [0, 0, 0.12]
    P, _ = api.palm_pose()
    yield from api.carry([P[0], P[1], max(P[2], 1.06)], seconds=0.6)
    yield from api.move_joints(q=api.OPEN.copy(), seconds=0.5)
    yield from api.tuck(["middle", "ring", "little"], seconds=0.4)
    hw = 0.024
    r = yield from api.move_tips(tips={"thumb": C - hw * EZ, "index": C + hw * EZ}, seconds=0.6)
    api.note(f"open pinch ik {r['ik_residual_mm']} tips {mm(api.tip_targets())}")
    yield from api.carry(P_air, Rp, seconds=3.0)
    yield from api.carry(P_pre, seconds=1.2)

    def hit():
        fb = api.contact_forces("book")
        ft = api.contact_forces("tile")
        return fb[:2].max() > 0.4 or ft[:2].max() > 0.4
    stopped = yield from guarded(api, api.carry(P_pinch, seconds=1.6), hit)
    P, _ = api.palm_pose()
    api.note(f"approach stopped={stopped} palm {mm(P)} short_by {1000 * (P_pinch[1] - P[1]):.1f} "
             f"book {np.round(api.contact_forces('book'), 2).tolist()} tile {np.round(api.contact_forces('tile'), 2).tolist()}")
    if stopped:
        yield from api.carry(P + [0, -0.003, 0], seconds=0.3)

    # re-centre the pinch height on the (possibly nudged) tile, then close to near contact
    pt, _ = api.object_pose("tile")
    hw = 0.0125
    r = yield from api.move_tips(tips={"thumb": C - hw * EZ, "index": C + hw * EZ}, seconds=0.6, straight=True)
    g = yield from api.grip("tile", {"thumb": 2.0, "index": 2.0}, seconds=1.5, max_travel=0.014)
    api.note(f"pinch grip {g} tile {mm(api.object_pose('tile')[0])}")

    # ------------------------------------------------ phase 3: pull off and lift
    P, _ = api.palm_pose()
    yield from api.carry(P + [0, -0.04, 0.008], seconds=1.0)
    yield from api.carry(P + [0, -0.07, 0.15], seconds=2.0)
    yield from api.hold(0.5)
    pt, _ = api.object_pose("tile")
    tw = api.tips_world()
    api.note(f"final tile {mm(pt)} pinch_mid {mm((tw[0] + tw[1]) / 2)} grip {api.grip_state()}")
