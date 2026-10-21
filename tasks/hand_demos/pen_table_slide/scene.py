"""IMG_4781: a pen stands on its tip, the hand pinches it and lowers, and the table pushes the pen up through the fingers.

The pen stands on the table inside a through-bored guide block (a 14 mm
bore, 60 mm tall) that keeps it upright before the hand arrives; the bore
carries no vertical load. The hand comes down palm first as in the clip and
pinches the exposed top lightly: thumb on one side, index and middle on the
other, spread 16 mm along the barrel. It then keeps lowering: the table
pushes the pen up through the fingers (extrinsic dexterity), which is the
clip's motion. Finally it lets go, backs off, takes a fresh three-finger
carry grip near the top (thumb and index across the barrel, middle on the
third side), lifts the pen out of the guide and holds it for 2 s.

What differs from the clips, and why:
- IMG_4780 picks the pen up off the flat table. With the DG5F fingertips
  that failed: the thumb lies flat and rides over a 7.5 mm barrel while the
  index tip's low point sits outboard of its pad (24 tilt/height/squeeze
  combinations, none lifted it).
- A thumb-index pinch alone does not carry the pen: the two fingertip
  contacts are not antipodal (the index pushes 37 deg off the thumb's line),
  so the barrel is squeezed out within ~1 s, sooner the firmer the pinch.
  The carry grip adds the middle finger on the third side. The guide is
  60 mm tall so that, as a pen cup does, it holds the lower barrel while
  the carry grip closes; with a 30 mm guide the pen pivots on its tip, the
  squeeze pushes it 12 mm out of the middle finger's reach, and the grip
  creeps out within ~2 s of the lift.

Contact model: hand geoms use condim 4 (torsional friction 0.005 m, the
workcell default), so a two-point pinch is not a free hinge about the
pinch axis.

Assumed numbers (not measured): pen 7.5 mm x 140 mm, 5 g, friction 0.8
(dexagent.objects); fingertip friction 0.9 (workcell default).
Only robot joint targets are written.
"""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, TABLE_Z, YAW
from dexagent.hand import Controller, PINCH_CENTRE
from dexagent.objects import pen, PEN_RADIUS, PEN_HALF

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4781_pen_table_slide.mov"
END = PEN_HALF + PEN_RADIUS
DOWN = hand_down(YAW)        # palm down, fingers forward: the pinch axis is horizontal
HOLDER = WORK + np.array([0.06, 0.13, 0.0])
# A guide block on the table: a 14 mm square through-bore, 60 mm tall, in a 40 mm block.
BORE, BLOCK, HEIGHT = 0.0070, 0.020, 0.060
GRIP_BELOW_TOP = 0.030      # pinch this far below the upper cap centre
SQUEEZE = 0.006             # thumb-to-index pad half-separation commanded (a light pinch)
RISE = 0.016                # middle-finger contact above the index, along the barrel [m]
SLIDE = 0.035               # how far the hand keeps lowering with the tip on the table [m]
LIFT = 0.10                 # lift after the slide [m]
HOLD_S = 2.0                # hold after the lift [s]
# Carry grip for the lift-out (as in pen_finger_roll): thumb and index pads 3 mm either
# side of a centre 12 mm toward the middle finger from PINCH_CENTRE, and the middle pad
# on the third side. A thumb-index pinch alone squeezes the barrel out within ~1 s.
CARRY_CENTRE = PINCH_CENTRE + np.array([0, -0.012, 0])
CARRY_HALF = 0.003
CARRY_MIDDLE = np.array([0.010, -0.006, 0.0])

scene = table_scene("pen_table_slide")
wall = (BLOCK - BORE) / 2
for name, offset, half in [
        ("block_e", (BORE + wall, 0, HEIGHT / 2), (wall, BLOCK, HEIGHT / 2)),
        ("block_w", (-BORE - wall, 0, HEIGHT / 2), (wall, BLOCK, HEIGHT / 2)),
        ("block_n", (0, BORE + wall, HEIGHT / 2), (BORE, wall, HEIGHT / 2)),
        ("block_s", (0, -BORE - wall, HEIGHT / 2), (BORE, wall, HEIGHT / 2))]:
    scene.add(rss.slab(name=name, half=half, at=tuple(HOLDER + offset), color=(0.55, 0.42, 0.28)))
scene.add(pen(at=tuple(HOLDER + [0, 0, PEN_RADIUS + PEN_HALF + 0.0005]), direction=(0, 0, 1)))
from robosimstudio.specs import CameraSpec
# Close side view of the block and the touch point (inspection only).
scene.add(CameraSpec(name="side", width=1280, height=720, fov_y=40,
                     eye=(0.80, 0.28, TABLE_Z + 0.16), look_at=(0.55, 0.0, TABLE_Z + 0.08)))



def grip_q(c, half):
    """Thumb against index and middle, the three contacts spread along the barrel.

    Palm down, the hanging pen runs along palm -x (up). The index presses at
    the pinch centre, the middle RISE higher on the same side and the thumb
    between them on the other side, so the table's push on a slightly tilted
    tip cannot rotate the pen about a single thumb-index contact.
    """
    from dexagent.hand import PINCH_CENTRE as C, PINCH_AXIS as A, CURL
    q = c.hand.copy()
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    up = np.array([-RISE, 0, 0])
    return c.fingertips({0: C + half * A + up / 2, 1: C - half * A, 2: C - half * A + up + [0, -0.012, 0]}, q)


def pen_geometry(c):
    p, R = c.pose("pen")
    a = R[:, 2]
    return p, a, np.array([p + END * a, p - END * a])


def grip_coordinate(c):
    """Pen centre along its axis, measured from the thumb-index pad midpoint [m]."""
    p, a, _ = pen_geometry(c)
    if a[2] < 0:
        a = -a
    return float((p - c.tips_world()[:2].mean(0)) @ a)


def tilt_of(c):
    return float(np.degrees(np.arccos(abs(pen_geometry(c)[1][2]))))


def tip_height(c):
    p, a, _ = pen_geometry(c)
    return float(min(p[2] + PEN_HALF * a[2], p[2] - PEN_HALF * a[2]) - PEN_RADIUS - TABLE_Z)


def slide(sim, session=None):
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.6, "settle")
        p0, a, _ = pen_geometry(c)
        up = a if a[2] > 0 else -a
        grip = p0 + PEN_HALF * up - GRIP_BELOW_TOP * up
        R = DOWN
        yield from c.move(c.palm_for(grip + [0, 0, .06], R=R), R, grip_q(c, 0.024), 2.0, "approach")
        yield from c.move(c.palm_for(grip, R=R), R, seconds=1.0, phase="reach")
        yield from c.move(hand=grip_q(c, SQUEEZE), seconds=0.6, phase="pinch")
        yield from c.hold(0.4)
        s0 = grip_coordinate(c)
        z0 = c.measured_palm()[0][2]
        metrics["grip_force_N"] = round(float(c.contact_forces("pen")[:3].sum()), 2)
        # Keep lowering: the tip is on the table, so the pen slides up through the fingers.
        heights, tilts, forces = [], [], []
        for _ in c.move(c.xyz - [0, 0, SLIDE], seconds=2.5, phase="slide"):
            heights.append(tip_height(c))
            tilts.append(tilt_of(c))
            forces.append(float(c.contact_forces("pen")[:3].sum()))
            yield
        yield from c.hold(0.3)
        s1 = grip_coordinate(c)
        # Positive slip: the pen moved up through the fingers.
        metrics.update(slip_mm=round(1000 * (s1 - s0), 1),
                       palm_lowered_mm=round(1000 * (z0 - c.measured_palm()[0][2]), 1),
                       tip_height_during_slide_mm=[round(1000 * min(heights), 1), round(1000 * max(heights), 1)],
                       max_tilt_during_slide_deg=round(float(max(tilts)), 1),
                       finger_force_during_slide_N=[round(min(forces), 2), round(max(forces), 2)])
        # Let go, back off, and take a fresh pinch where the pen now stands
        # (the slide leaves the fingers low on the barrel, wedged against the
        # guide), then lift it out of the guide and hold it.
        yield from c.move(hand=grip_q(c, 0.024), seconds=0.5, phase="release")
        yield from c.move(c.xyz + [0, 0, 0.10], seconds=1.2, phase="back_off")
        yield from c.move(hand=c.pinch(0.026, CARRY_CENTRE), seconds=0.5)
        p1, a1, _ = pen_geometry(c)
        up1 = a1 if a1[2] > 0 else -a1
        grip1 = p1 + PEN_HALF * up1 - GRIP_BELOW_TOP * up1
        yield from c.move(c.palm_for(grip1 + [0, 0, .08], CARRY_CENTRE, R), R, seconds=1.0, phase="regrasp")
        yield from c.move(c.palm_for(grip1, CARRY_CENTRE, R), R, seconds=1.0)
        carry = c.fingertips({2: CARRY_CENTRE + CARRY_MIDDLE}, c.pinch(CARRY_HALF, CARRY_CENTRE, tuck=(3, 4)))
        yield from c.move(hand=carry, seconds=0.8)
        yield from c.hold(0.4)
        yield from c.move(c.xyz + [0, 0, LIFT], seconds=2.0, phase="lift_out")
        yield from c.hold(HOLD_S, "hold")
        metrics["held_tip_height_mm"] = round(1000 * tip_height(c), 1)
        metrics["held_tilt_deg"] = round(tilt_of(c), 1)
        slid = metrics["slip_mm"] > 15 and max(heights) < 0.003
        held = metrics["held_tip_height_mm"] > 0.6 * 1000 * LIFT
        outcome = "success" if slid and held else ("slid_then_dropped" if slid else "no_slide")
    finally:
        c.finish("pen_table_slide", outcome, metrics, HERE / "results")


routines = {"pinch and slide": slide}
