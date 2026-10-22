"""Slide a flat tile to the edge of a book, then pinch the overhang and lift it.

A playing card is too thin for robust contact in this solver, so the object
is a 90 x 60 x 3 mm polystyrene tile/coaster of 17 g (nominal). It lies flat
on a 160 x 220 x 45 mm book (a hardcover textbook; the index needs room under the overhang). Flat on the book it cannot be grasped: the hand
first presses two fingertips onto it and drags it over the book's near edge
(extrinsic dexterity: the edge creates a graspable overhang), then turns its
fingers down, slides the index under the overhang and the thumb over it, and
lifts. Success: tile raised > 80 mm and still between thumb and index
(its centre < 60 mm from the pinch midpoint; the pinch is on the overhang,
about 35 mm from the centre). Tile-on-book friction 0.3 (plastic on a paper cover), fingertip
friction 0.9; both nominal, not measured. Only arm and finger joint targets
are commanded.
"""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio.specs import CameraSpec
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, TABLE_Z
from dexagent.hand import Controller, OPEN, CURL, PINCH_CENTRE, PINCH_AXIS
from dexagent.cases import current

HERE = Path(__file__).resolve().parent
# Initial conditions (dexagent.cases; nominal at severity 0, full range at 1):
# book position +-20 mm, book thickness 45 mm +-10 %; tile on the book +-15 mm (x),
# +-20 mm (y, toward/away from the edge), yaw +-25 deg, length and width +-12 %
# (thickness fixed), mass 17 g +-50 %.
CASE = current()
_BDX, _BDY = CASE.u(0.02, "book_dx"), CASE.u(0.02, "book_dy")
_BTHICK = CASE.scale(0.10, "book_thickness")
_TDX, _TDY = CASE.u(0.015, "tile_dx"), CASE.u(0.02, "tile_dy")
_TYAW = CASE.u(np.radians(25), "tile_yaw")
_TSIZE = CASE.scale(0.12, "tile_size")
TILE_MASS = 0.017 * CASE.scale(0.5, "tile_mass")
BOOK_HALF = np.array([0.08, 0.11, 0.0225 * _BTHICK])
BOOK = WORK + np.array([_BDX, _BDY, BOOK_HALF[2]])
EDGE_Y = BOOK[1] - BOOK_HALF[1]                  # the near (robot-side) edge the tile is dragged over
TILE_HALF = np.array([0.03 * _TSIZE, 0.045 * _TSIZE, 0.0015])   # long side along the drag
TILE = np.array([BOOK[0] + _TDX, BOOK[1] - 0.02 + _TDY, BOOK[2] + BOOK_HALF[2] + TILE_HALF[2]])
TILE_QUAT = (0.0, 0.0, np.sin(_TYAW / 2), np.cos(_TYAW / 2))
# Fingers pointing down, palm facing +y (toward the book): the pinch axis is vertical,
# index below (distal), thumb above.
FINGERS_DOWN = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).astype(float)

scene = table_scene("card_pickup")
scene.add(rss.slab(name="book", half=tuple(BOOK_HALF), at=tuple(BOOK), color=(0.55, 0.12, 0.1), friction=0.3))
scene.add(rss.box(name="tile", half=tuple(TILE_HALF), at=tuple(TILE), quat=TILE_QUAT, mass=TILE_MASS, friction=0.3, color=(0.9, 0.85, 0.3)))
scene.add(CameraSpec(name="edge", width=1280, height=720, fov_y=40,
                     eye=(BOOK[0] + 0.32, EDGE_Y - 0.28, BOOK[2] + 0.22), look_at=(BOOK[0], EDGE_Y + 0.02, BOOK[2])))


def overhang(c):
    p, R = c.pose("tile")
    corners = p[1] + np.array([s * TILE_HALF[0] * R[1, 0] + t * TILE_HALF[1] * R[1, 1] for s in (-1, 1) for t in (-1, 1)])
    return float(EDGE_Y - corners.min())


def drag_hand(c, palm, R, contact, press):
    """Index and middle pads on `contact` (world), pressed `press` m into the tile; others tucked."""
    q = OPEN.copy()
    q[0:4] = [0.0, -0.2, 0.2, 0.2]
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    targets = {1: c.palm_frame_target(contact + np.array([-0.012, 0, -press]), xyz=palm, R=R),
               2: c.palm_frame_target(contact + np.array([0.012, 0, -press]), xyz=palm, R=R)}
    return c.fingertips(targets, q)


# The drag stops at `target_overhang`, but stick-slip and the release carry the tile
# about 26 mm further (Newton 1.7). 12 mm ends near 38 mm, centre still 7 mm on the
# book; a 30 mm target ended at 46 mm, past the tile's 45 mm half-length, and the
# tile tipped off the edge before the pinch.
def pick_up(sim, session=None, target_overhang=0.012, press=0.008, max_slide=0.13, lean_deg=0.0):
    c = Controller(sim, session, objects=("tile",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, _ = c.pose("tile")
        top = p0 + np.array([0, 0.025, TILE_HALF[2]])            # where the two pads land
        R = hand_down(np.pi / 2)                                   # fingers toward +y, away from the edge
        tip_home = np.array([0.095, 0.016, 0.12])                 # comfortable pad position, palm frame
        palm = top - R @ tip_home
        hover = palm + np.array([0, 0, 0.04])
        yield from c.move(hover, R, drag_hand(c, palm, R, top, -0.01), 1.5, "approach")
        yield from c.move(palm, R, drag_hand(c, palm, R, top, press), 1.0, "press")
        yield from c.hold(0.3)
        # Drag toward the edge until the measured overhang reaches the target.
        c.phase = "drag"
        travel = 0.0
        while overhang(c) < target_overhang and travel < max_slide:
            # 40 mm/s, easing to 10 mm/s over the last 10 mm against stick-slip overshoot
            travel += 0.0002 + 0.0006 * np.clip((target_overhang - overhang(c)) / 0.01, 0, 1)
            c.xyz = palm + np.array([0, -travel, 0])
            c.arm = c.kin.solve_arm(c.xyz, R, c.arm, tol=0.004)
            c.step()
            yield
        metrics["finger_travel_mm"] = round(1000 * travel, 1)
        # Release by backing the pads away from the edge (+y) as they rise, so the
        # unloading fingertips cannot shove the tile further over.
        here = c.xyz.copy()
        yield from c.move(here + [0, 0.02, 0.012], R, seconds=0.5, phase="release")
        yield from c.move(here + [0, 0.02, 0.06], R, seconds=0.6, phase="clear")
        metrics["overhang_mm"] = round(1000 * overhang(c), 1)
        if overhang(c) < 0.015:
            outcome = "slide_failed"; return
        p1, _ = c.pose("tile")
        mid = np.array([p1[0], EDGE_Y - 0.65 * overhang(c), p1[2]])
        # Optional lean of the fingers toward the book (0 = straight down, which works).
        a = np.radians(lean_deg)
        Rp = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ FINGERS_DOWN
        outside = mid + np.array([0, -0.05, 0])
        # Open asymmetrically: thumb 20 mm over the tile, index pad only 12 mm under it,
        # because the curled index's nail end reaches ~25 mm below its pad and the table is 45 mm down.
        open_pinch = c.pinch(0.016, centre=PINCH_CENTRE + 0.004 * PINCH_AXIS)
        yield from c.move(c.palm_for(outside + [0, 0, 0.06], R=Rp), Rp, open_pinch, 2.0, "turn fingers down")
        yield from c.move(c.palm_for(outside, R=Rp), Rp, seconds=1.0, phase="lower beside the edge")
        yield from c.move(c.palm_for(mid, R=Rp), Rp, seconds=1.2, phase="index under the overhang")
        yield from c.move(hand=c.pinch(-0.004), seconds=0.8, phase="pinch")
        yield from c.hold(0.3)
        yield from c.move(c.palm_for(mid + [0, -0.03, 0.12], R=Rp), Rp, seconds=1.5, phase="lift")
        yield from c.hold(0.6, "hold")
        p2, _ = c.pose("tile")
        metrics["lift_mm"] = round(1000 * (p2[2] - p0[2]), 1)
        tips = c.tips_world()
        metrics["tile_to_pinch_mm"] = round(1000 * float(np.linalg.norm(p2 - (tips[0] + tips[1]) / 2)), 1)
        held = p2[2] - p0[2] > 0.08 and metrics["tile_to_pinch_mm"] < 60
        outcome = "success" if held else "pinch_failed"
    finally:
        c.finish("card_pickup", outcome, metrics, HERE / "results")


routines = {"slide to the edge and pick up": pick_up}
