"""IMG_4776 / IMG_4777: roll a vertical pen about its own axis between thumb and index, then swing it.

Both clips start with the pen already upright in the hand. Here it stands in a
pen cup (an addition the clips do not show) and is taken out of it from
above (pen_cup.py): thumb and index pads on opposite sides of the barrel,
30 mm below its top, and the middle finger on the third side. Then:

* roll (4776): thumb, index and middle each move with the barrel surface for
  a turn about the pen axis (thumb +y, index -y, middle -z in the palm
  frame); rolling without slip turns it by about d / r for a shear d;
* swing (4777): see-saw. The thumb pad slides 3.5 mm down the barrel while
  the index slides up, then the reverse, tilting the pen in the pinch plane;
  it is measured in the palm frame and must come back to rest. The clip's
  swing is wider (about 30 deg); 5 mm slides drop the pen.

Picking the pen off the table, as in IMG_4773, is not used: with these
16 x 20 mm fingertip shells the best synthesized pinch (pen_grasp.py) lifts
it 2-9 cm before the contacts ride over the top of the 7.5 mm barrel.

Assumed: pen 7.5 mm x 140 mm, 5 g, friction 0.8 (dexagent.rss.objects); hand
pads 0.9; cup 60 mm tall with a 20 mm square bore. The capsule is symmetric,
so roll is read off the body's orientation (twist about its own axis).
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.rss.scene import table_scene, WORK
from dexagent.rss.workcell import TABLE_Z
from dexagent.rss.hand import Controller, PINCH_CENTRE
from pen_cup import add_cup_and_pen, pick_from_cup, PINCH_HALF, MIDDLE_HOLD

HERE = Path(__file__).resolve().parent
CLIPS = [HERE.parents[2] / "reference/hand_demos/IMG_4776_pen_finger_roll.mov",
         HERE.parents[2] / "reference/hand_demos/IMG_4777_pen_swing.mov"]
SWING_SHIFT = 0.0035                           # opposite thumb/index slide along the barrel for the swing
SHOW = np.array([-0.04, 0.07, -0.02])          # where the in-hand motions are shown, from the lifted grip

scene = table_scene("pen_finger_roll")
add_cup_and_pen(scene, WORK)


def twist_about_axis(R0, R):
    """Rotation of R relative to R0 about the body's own z (the pen axis), degrees."""
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float(np.degrees(2 * np.arctan2(q[2], q[3])))


def signed_tilt(rest, now):
    """Tilt of the pen about palm y (in the thumb-index pinch plane), degrees."""
    d = np.arctan2(now[2], now[0]) - np.arctan2(rest[2], rest[0])
    return float(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))


def pen_in_palm(c):
    """The pen axis in the measured palm frame."""
    return c.in_palm("pen")[1][:, 2]


def roll_and_swing(sim, session=None):
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        p0, _ = c.pose("pen")
        held = yield from pick_from_cup(c, lift=0.10, show=SHOW)
        p, R_pen = c.pose("pen")
        metrics["lift_mm"] = round(1000 * (p[2] - p0[2]), 1)
        if p[2] - p0[2] < .06:
            outcome = "dropped_on_lift"; return
        metrics["hang_tilt_deg"] = round(float(np.degrees(np.arccos(abs(R_pen[2, 2])))), 1)

        # Roll (4776): every contact moves with the barrel surface for a turn
        # about the pen axis (palm +x): thumb +y, index -y, middle -z.
        R_ref, twists = R_pen, []
        for k in range(3):
            for d in (0.0025, -0.0025):
                q = c.fingertips({0: PINCH_CENTRE + [0, d, -PINCH_HALF], 1: PINCH_CENTRE + [0, -d, PINCH_HALF],
                                  2: PINCH_CENTRE + MIDDLE_HOLD + [0, 0, -d]}, held)
                yield from c.move(hand=q, seconds=0.8, phase=f"roll_{k}")
                yield from c.hold(0.2)
                twists.append(twist_about_axis(R_ref, c.pose("pen")[1]))
        yield from c.move(hand=held, seconds=0.6, phase="roll_centre")
        metrics["roll_twist_deg"] = [round(t, 1) for t in twists]
        metrics["roll_range_deg"] = round(max(twists) - min(twists), 1)
        if c.pose("pen")[0][2] < TABLE_Z + 0.05:
            outcome = "dropped_while_rolling"; return

        # Swing (4777): see-saw the pen in the pinch plane. The thumb pad
        # slides down the barrel while the index pad slides up (and back), a
        # couple that tilts the pen about palm y by about 2 d / 7.5 mm.
        yield from c.hold(0.6, "swing_rest")
        rest = pen_in_palm(c)
        tilts = []
        for k, d in enumerate((SWING_SHIFT, -SWING_SHIFT, SWING_SHIFT, -SWING_SHIFT)):
            q = c.fingertips({0: PINCH_CENTRE + [d, 0, -PINCH_HALF], 1: PINCH_CENTRE + [-d, 0, PINCH_HALF]}, held)
            yield from c.move(hand=q, seconds=0.7, phase=f"swing_{k}")
            yield from c.hold(0.2)
            tilts.append(signed_tilt(rest, pen_in_palm(c)))
        yield from c.move(hand=held, seconds=0.6, phase="swing_home")
        yield from c.hold(0.4)
        tilts.append(signed_tilt(rest, pen_in_palm(c)))
        metrics["swing_tilt_deg"] = [round(t, 1) for t in tilts]
        z = c.pose("pen")[0][2]
        metrics["final_height_mm"] = round(1000 * (z - TABLE_Z), 1)
        held_ok = z > TABLE_Z + 0.05
        rolled = metrics["roll_range_deg"] >= 30
        swing = np.array(tilts[:4])
        swung = swing.max() - swing.min() >= 15 and abs(tilts[4]) < 6
        metrics["swing_range_deg"] = round(float(swing.max() - swing.min()), 1)
        outcome = "success" if held_ok and rolled and swung else (
            "dropped" if not held_ok else f"partial(rolled={rolled},swung={swung})")
    finally:
        c.finish("pen_finger_roll", outcome, metrics, HERE / "results")


routines = {"lift from cup, roll and swing": roll_and_swing}
