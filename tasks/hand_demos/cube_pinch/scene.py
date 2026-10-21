"""IMG_4772: pinch a 40 mm foam cube off the table, lift and turn it, set it back down."""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, TABLE_Z, YAW
from dexagent.hand import Controller, OPEN

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4772_cube_pinch_lift.mov"
EDGE = 0.04
START = WORK + np.array([0.0, 0.0, EDGE / 2])
# The cube starts square to the default palm yaw, so the thumb and index close on opposite faces.
QUAT = (0.0, 0.0, np.sin(YAW / 2), np.cos(YAW / 2))
HELPERS, TUCK = (), (2, 3, 4)
# Pads 36 mm apart on the 40 mm cube: about 8 N each side. A harder squeeze lets the
# stronger thumb push the index back, and the offset pads tip the cube about the pinch axis.
SQUEEZE = 0.018

scene = table_scene("cube_pinch")
scene.add(rss.box(name="cube", half=(EDGE / 2,) * 3, at=tuple(START), quat=QUAT, mass=0.012, friction=0.9, color=(0.42, 0.28, 0.72)))


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pick_and_place(sim, session=None):
    c = Controller(sim, session, objects=("cube",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, _ = c.pose("cube")
        R = hand_down()
        above = c.palm_for(p0 + [0, 0, .08], R=R)
        yield from c.move(above, R, c.pinch(0.034, helpers=HELPERS, tuck=TUCK), 1.5, "approach")
        yield from c.move(c.palm_for(p0, R=R), R, seconds=1.0, phase="descend")
        yield from c.move(hand=c.pinch(SQUEEZE, helpers=HELPERS, tuck=TUCK), seconds=0.8, phase="pinch")
        yield from c.hold(0.3)
        lifted = c.palm_for(p0 + [0, 0, .12], R=R)
        yield from c.move(lifted, R, seconds=1.2, phase="lift")
        z = c.pose("cube")[0][2]
        metrics["lift_mm"] = round(1000 * (z - p0[2]), 1)
        if z - p0[2] < .08:
            outcome = "dropped"; return
        yaw0 = yaw_of(c.pose("cube")[1])
        Rt = hand_down(YAW + np.radians(35))
        yield from c.move(c.palm_for(p0 + [0, 0, .12], R=Rt), Rt, seconds=1.5, phase="turn")
        yield from c.hold(0.5, "hold")
        metrics["held_turn_deg"] = round(float(np.degrees(wrap(yaw_of(c.pose("cube")[1]) - yaw0))), 1)
        yield from c.move(c.palm_for(p0 + [0, 0, .001], R=Rt), Rt, seconds=1.2, phase="place")
        yield from c.move(hand=c.pinch(0.032, helpers=HELPERS, tuck=TUCK), seconds=0.6, phase="release")
        yield from c.move(c.palm_for(p0 + [0, 0, .12], R=Rt), Rt, seconds=1.0, phase="retreat")
        yield from c.hold(0.4, "rest")
        p, Rc = c.pose("cube")
        metrics.update(placed_offset_mm=(1000 * (p - p0)).round(1).tolist(),
                       resting_height_mm=round(1000 * (p[2] - TABLE_Z), 1),
                       final_yaw_deg=round(float(np.degrees(yaw_of(Rc))), 1))
        upright = abs(Rc[2, 2]) > .99
        outcome = "success" if upright and np.linalg.norm(metrics["placed_offset_mm"][:2]) < 15 else "placed_off_target"
    finally:
        c.finish("cube_pinch", outcome, metrics, HERE / "results")


routines = {"pick, turn and place": pick_and_place}
