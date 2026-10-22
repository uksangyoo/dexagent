"""card_pickup, layered level.

Plan: (1) palm down (default yaw), index pad presses the
tile centre and drags it toward the book's near edge until the tile centre sits
just inside the edge (long overhang).  (2) lift off, rotate the hand to fingers
pointing down / palm facing +y, so the thumb-index pinch axis is vertical
(thumb above, index below).  (3) come in from -y with the index below the tile
level, slide the open pinch over the overhang, close (contact grasp), lift.
"""
import numpy as np
from scipy.spatial.transform import Rotation

C = np.array([0.080, 0.040, 0.118])     # thumb-index pinch centre, palm frame
PAD = 0.008                             # pad centre to pad surface
EDGE_IN = 0.004                         # tile centre ends this far inside the book edge
PINCH_OUT = 0.017                       # pinch point this far outside the book edge
PRESS = 0.005


def _r(v, n=3):
    return np.round(np.asarray(v, float), n).tolist()


def run(api):
    pt, Rt = api.object_pose("tile")
    pb, _ = api.object_pose("book")
    y_e = pb[1] - 0.11
    top = pt[2] + 0.0015
    api.note(f"tile {_r(pt, 4)} book {_r(pb, 4)} y_edge {y_e:.4f}")
    api.note(f"start tips(palm) {_r(api.tips())}")

    Rs = api.hand_down()                                                   # palm down, default yaw (reachable)
    Rp = api.hand_down(-np.pi / 2) @ Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()  # fingers down, palm +y

    # ---------------- phase A: shape the hand high above the tile
    yield from api.tuck(["middle", "ring", "little"], 0.5)
    I_s = np.array([0.08, 0.04, 0.15])
    P = api.palm_for(pt + [0, 0, 0.15], I_s, Rs)
    r = yield from api.move_tips(P, Rs, tips={"index": I_s}, seconds=2.5)
    tt = api.tip_targets()
    api.note(f"slide tips(palm) {_r(tt)} res {r.get('ik_residual_mm')} meas {_r(api.tips())}")
    I = tt[1]

    # ---------------- phase B/C: press the tile centre and drag it to the edge
    y_goal = y_e + EDGE_IN
    for k in range(3):
        pt, _ = api.object_pose("tile")
        dy = y_goal - pt[1]
        if abs(dy) < 0.004:
            break
        pad = np.array([pt[0], pt[1], top + PAD])
        yield from api.carry(api.palm_for(pad + [0, 0, 0.03], I, Rs), Rs, 1.5 if k == 0 else 0.8)
        press = PRESS
        yield from api.carry(api.palm_for(pad - [0, 0, press], I, Rs), Rs, 0.8)
        yield from api.hold(0.2)
        f = api.contact_forces("tile")
        if f[1] < 0.3:
            press += 0.004
            yield from api.carry(api.palm_for(pad - [0, 0, press], I, Rs), Rs, 0.4)
            yield from api.hold(0.2)
            f = api.contact_forces("tile")
        api.note(f"press{k} tile_f {_r(f, 2)} book_f {_r(api.contact_forces('book'), 2)}")
        P0 = api.palm_pose()[0]
        yield from api.carry(P0 + [0, dy, 0], Rs, max(1.2, abs(dy) / 0.03))
        yield from api.hold(0.2)
        pt, _ = api.object_pose("tile")
        api.note(f"slide{k} tile {_r(pt, 4)} rem {1000 * (y_goal - pt[1]):.1f}mm")
        yield from api.carry(api.palm_pose()[0] + [0, 0, 0.03], Rs, 0.5)

    # ---------------- phase D: retract, rotate to the pinch pose high and outside the edge
    pt, Rt = api.object_pose("tile")
    zc = pt[2]
    hw = 0.0015 + PAD + 0.012
    tips_open = {"thumb": C + Rp.T @ [0, 0, hw], "index": C + Rp.T @ [0, 0, -hw]}
    yield from api.carry(api.palm_pose()[0] + [0, -0.04, 0.10], Rs, 1.0)
    wA = np.array([pt[0], y_e - 0.07, zc])
    try:
        r = yield from api.move_tips(api.palm_for(wA + [0, 0, 0.12], C, Rp), Rp, tips=tips_open, seconds=3.0)
    except ValueError:   # arm IK failed on the way: finish the rotation at a comfortable spot, then go
        api.note("rotation IK failed; retrying via (0.45,-0.30,1.0)")
        r = yield from api.move_tips(np.array([0.45, -0.30, 1.0]), Rp, tips=tips_open, seconds=3.0)
        yield from api.carry(api.palm_for(wA + [0, 0, 0.12], C, Rp), Rp, 1.5)
    tt = api.tip_targets()
    mid = (tt[0] + tt[1]) / 2
    api.note(f"pinch tips(palm) {_r(tt[:2])} res {r.get('ik_residual_mm')}")
    yield from api.carry(api.palm_for(wA, mid, Rp), Rp, 1.2)
    wG = np.array([pt[0], y_e - PINCH_OUT, zc])
    yield from api.carry(api.palm_for(wG, mid, Rp), Rp, 1.5)
    yield from api.hold(0.2)
    api.note(f"at grasp: tips_w {_r(api.tips_world()[:2])} tile {_r(api.object_pose('tile')[0], 4)} "
             f"tile_f {_r(api.contact_forces('tile'), 2)} book_f {_r(api.contact_forces('book'), 2)}")

    # ---------------- phase E: close the pinch on the overhang
    pt, Rt = api.object_pose("tile")
    tw = api.tips_world()
    xy = (tw[0][:2] + tw[1][:2]) / 2
    w_top = np.array([xy[0], xy[1], pt[2] + 0.0015])
    w_bot = np.array([xy[0], xy[1], pt[2] - 0.0015])
    pts = {"thumb": Rt.T @ (w_top - pt), "index": Rt.T @ (w_bot - pt)}
    nrm = {"thumb": Rt.T @ np.array([0, 0, 1.0]), "index": Rt.T @ np.array([0, 0, -1.0])}
    r = yield from api.grasp("tile", pts, squeeze=0.005, seconds=1.0, normals=nrm)
    yield from api.hold(0.3)
    api.note(f"grasp f {_r(api.contact_forces('tile'), 2)} res {r.get('ik_residual_mm')}")

    # ---------------- phase F: lift
    yield from api.carry(api.palm_pose()[0] + [0, -0.02, 0.06], Rp, 1.5)
    yield from api.carry(api.palm_pose()[0] + [0, -0.02, 0.07], Rp, 1.5)
    yield from api.hold(0.5)
    pt, Rt = api.object_pose("tile")
    tw = api.tips_world()
    api.note(f"lifted tile {_r(pt, 4)} tilt {np.degrees(np.arccos(min(1, abs(Rt[2, 2])))):.0f} "
             f"d_pinch {1000 * np.linalg.norm(pt - (tw[0] + tw[1]) / 2):.0f}mm f {_r(api.contact_forces('tile'), 2)}")
