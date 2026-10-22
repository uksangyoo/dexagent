"""IMG_4776 / IMG_4777: roll a vertical pen about its own axis between the fingers, and swing it.

Both clips start with the pen already upright in the hand. Here it stands in a
pen cup (an addition the clips do not show) and is taken out of it from
above (pen_cup.py): thumb and index on opposite sides of the barrel, 30 mm
below its top, and the middle finger on the third side. The grip is centred
12 mm toward the middle finger from the hand's widest pinch, so the middle
can press on the barrel (from PINCH_CENTRE it is 8 mm short). Then:

* swing (4777): see-saw. The thumb pad slides 8 mm along the barrel while the
  index slides the other way, then the reverse, tilting the pen about +-25 deg
  in the thumb-index plane (measured in the palm frame). Afterwards the slide
  is trimmed against the measured tilt until the pen is back within 3 deg;
* roll (4776): the thumb strokes 8 mm back and forth along the barrel's
  surface tangent (from the measured contact force) while index and middle
  hold still, so the pen rolls on them. Strokes under ~6 mm stay inside the
  servo compliance and static friction and do not roll it; 8 mm rolls it in
  stick-slip jumps of 30-50 deg.

Picking the pen off the table, as in IMG_4773, is not used: with these
16 x 20 mm fingertip shells the best synthesized pinch (pen_grasp.py) lifts
it 2-9 cm before the contacts ride over the top of the 7.5 mm barrel.

Assumed: pen 7.5 mm x 140 mm, 5 g, friction 0.8 (dexagent.objects); hand
pads 0.9 with torsional friction (condim 4, workcell default); cup 60 mm
tall with a 20 mm square bore. The capsule is symmetric, so roll is read off
the body's orientation (twist about its own axis), unwrapped step by step.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.scene import table_scene, WORK
from dexagent.workcell import TABLE_Z
from dexagent.hand import Controller, PINCH_CENTRE
from dexagent.objects import PEN_RADIUS, PEN_HALF, PEN_MASS
from dexagent.cases import current
from pen_cup import add_cup_and_pen, pick_from_cup

HERE = Path(__file__).resolve().parent
CLIPS = [HERE.parents[2] / "reference/hand_demos/IMG_4776_pen_finger_roll.mov",
         HERE.parents[2] / "reference/hand_demos/IMG_4777_pen_swing.mov"]
GRIP_CENTRE = PINCH_CENTRE + np.array([0, -0.012, 0])   # palm-frame point put on the pen axis
SWING_SHIFT = 0.008                            # opposite thumb/index slide along the barrel for the swing
ROLL_STROKE = 0.008                            # thumb stroke along the barrel tangent for the roll
SHOW = np.array([-0.04, 0.07, -0.02])          # where the in-hand motions are shown, from the lifted grip

# Initial conditions (dexagent.cases; nominal at severity 0, full range at 1):
# cup and pen position +-20 mm, cup yaw +-30 deg, pen diameter 7.5 mm +-12 %,
# pen length 140 mm +-10 %, pen mass 5 g +-50 %.
CASE = current()
_DX, _DY, _DYAW = CASE.u(0.020, "dx"), CASE.u(0.020, "dy"), CASE.u(np.radians(30), "cup_yaw")
_R = PEN_RADIUS * CASE.scale(0.12, "pen_diameter")
_L = (PEN_HALF + PEN_RADIUS) * CASE.scale(0.10, "pen_length")   # centre to tip
_M = PEN_MASS * CASE.scale(0.5, "pen_mass")
PEN_TOP = _L if CASE.severity else PEN_HALF + PEN_RADIUS
CUP_AT = WORK + np.array([_DX, _DY, 0.0])

scene = table_scene("pen_finger_roll")
if CASE.severity:
    add_cup_and_pen(scene, CUP_AT, yaw=_DYAW, radius=_R, half=_L - _R, mass=_M)
else:
    add_cup_and_pen(scene, WORK)


def twist_about_axis(R0, R):
    """Rotation of R relative to R0 about the body's own z (the pen axis), degrees in (-180, 180]."""
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((np.degrees(2 * np.arctan2(q[2], q[3])) + 180) % 360 - 180)


def signed_tilt(rest, now):
    """Tilt of the pen about palm y (in the thumb-index pinch plane), degrees."""
    d = np.arctan2(now[2], now[0]) - np.arctan2(rest[2], rest[0])
    return float(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))


def pen_in_palm(c):
    """The pen axis in the measured palm frame."""
    return c.in_palm("pen")[1][:, 2]


def force_on_pen(c, finger):
    """Contact force [N] of one finger on the pen, palm frame."""
    return c.measured_palm()[1].T @ c.sim.contacts.pairwise_force(c.shapes("pen"), c._finger_shapes()[finger])


def roll_and_swing(sim, session=None):
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        p0, _ = c.pose("pen")
        held = yield from pick_from_cup(c, lift=0.10, show=SHOW, centre=GRIP_CENTRE, top=PEN_TOP)
        p, R_pen = c.pose("pen")
        metrics["lift_mm"] = round(1000 * (p[2] - p0[2]), 1)
        if p[2] - p0[2] < .06:
            outcome = "dropped_on_lift"; return
        metrics["hang_tilt_deg"] = round(float(np.degrees(np.arccos(abs(R_pen[2, 2])))), 1)
        pads = c.kin.tips_in_palm(held)

        # Swing (4777): see-saw the pen in the thumb-index plane. The thumb pad
        # slides along the barrel while the index pad slides the other way, a
        # couple that tilts the pen about palm y.
        def see_saw(d):
            return c.fingertips({0: pads[0] + [d, 0, 0], 1: pads[1] - [d, 0, 0]}, held)
        yield from c.hold(0.4, "swing_rest")
        rest = pen_in_palm(c)
        tilts = []
        for k, d in enumerate((SWING_SHIFT, -SWING_SHIFT, SWING_SHIFT, -SWING_SHIFT)):
            yield from c.move(hand=see_saw(d), seconds=0.7, phase=f"swing_{k}")
            yield from c.hold(0.2)
            tilts.append(signed_tilt(rest, pen_in_palm(c)))
        # Back to rest: trim the slide against the measured tilt.
        d = -SWING_SHIFT
        for k in range(6):
            tilt = signed_tilt(rest, pen_in_palm(c))
            if abs(tilt) < 3:
                break
            d = float(np.clip(d - 0.0002 * tilt, -SWING_SHIFT, SWING_SHIFT))
            yield from c.move(hand=see_saw(d), seconds=0.5, phase="swing_return")
            yield from c.hold(0.2)
        swing = np.array(tilts)
        metrics["swing_tilt_deg"] = [round(t, 1) for t in tilts]
        metrics["swing_range_deg"] = round(float(swing.max() - swing.min()), 1)
        metrics["swing_rest_deg"] = round(signed_tilt(rest, pen_in_palm(c)), 1)
        if c.pose("pen")[0][2] < TABLE_Z + 0.05:
            outcome = "dropped_while_swinging"; return

        # Roll (4776): the thumb strokes along the barrel's surface tangent
        # while index and middle hold, so the pen rolls on them about its axis.
        yield from c.hold(0.3, "roll_rest")
        grip = c.hand.copy()
        thumb = c.kin.tips_in_palm(grip)[0]
        axis = c.in_palm("pen")[1][:, 2]
        n = force_on_pen(c, 0)
        n = n - (n @ axis) * axis
        tangent = np.cross(axis, -n / np.linalg.norm(n))
        R_last, turned, twists = c.pose("pen")[1], 0.0, []
        for k in range(3):
            for s in (1, -1):
                yield from c.move(hand=c.fingertips({0: thumb + s * ROLL_STROKE * tangent}, grip), seconds=0.8, phase=f"roll_{k}")
                yield from c.hold(0.2)
                R_now = c.pose("pen")[1]
                turned += twist_about_axis(R_last, R_now)
                R_last = R_now
                twists.append(turned)
        yield from c.move(hand=grip, seconds=0.6, phase="roll_centre")
        yield from c.hold(0.4)
        metrics["roll_twist_deg"] = [round(t, 1) for t in twists]
        metrics["roll_range_deg"] = round(max(twists + [0.0]) - min(twists + [0.0]), 1)
        z = c.pose("pen")[0][2]
        metrics["final_height_mm"] = round(1000 * (z - TABLE_Z), 1)
        held_ok = z > TABLE_Z + 0.05
        rolled = metrics["roll_range_deg"] >= 30
        swung = metrics["swing_range_deg"] >= 15 and abs(metrics["swing_rest_deg"]) < 6
        outcome = "success" if held_ok and rolled and swung else (
            "dropped_while_rolling" if not held_ok else f"partial(rolled={rolled},swung={swung})")
    finally:
        c.finish("pen_finger_roll", outcome, metrics, HERE / "results")


routines = {"lift from cup, swing and roll": roll_and_swing}
