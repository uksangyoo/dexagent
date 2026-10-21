"""IMG_4772: pinch a 40 mm foam cube off the table, lift and turn it, set it back down."""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from dexagent.rss.scene import table_scene, WORK
from dexagent.rss.workcell import hand_down, TABLE_Z, YAW
from dexagent.rss.hand import Controller, OPEN

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4772_cube_pinch_lift.mov"
EDGE = 0.04
START = WORK + np.array([0.0, 0.0, EDGE / 2])

scene = table_scene("cube_pinch")
scene.add(rss.box(name="cube", half=(EDGE / 2,) * 3, at=tuple(START), mass=0.012, friction=0.9, color=(0.42, 0.28, 0.72)))


def pick_and_place(sim, session=None):
    c = Controller(sim, session, objects=("cube",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, _ = c.pose("cube")
        R = hand_down()
        above = c.palm_for(p0 + [0, 0, .08], R=R)
        yield from c.move(above, R, c.pinch(0.034, helpers=(2,), tuck=(3, 4)), 1.5, "approach")
        yield from c.move(c.palm_for(p0, R=R), R, seconds=1.0, phase="descend")
        yield from c.move(hand=c.pinch(0.012, helpers=(2,), tuck=(3, 4)), seconds=0.8, phase="pinch")
        yield from c.hold(0.3)
        lifted = c.palm_for(p0 + [0, 0, .12], R=R)
        yield from c.move(lifted, R, seconds=1.2, phase="lift")
        z = c.pose("cube")[0][2]
        metrics["lift_mm"] = round(1000 * (z - p0[2]), 1)
        if z - p0[2] < .08:
            outcome = "dropped"; return
        Rt = hand_down(YAW + np.radians(35))
        yield from c.move(c.palm_for(p0 + [0, 0, .12], R=Rt), Rt, seconds=1.5, phase="turn")
        yield from c.hold(0.5, "hold")
        _, Rc = c.pose("cube")
        metrics["held_turn_deg"] = round(float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))), 1)
        yield from c.move(c.palm_for(p0 + [0, 0, .001], R=Rt), Rt, seconds=1.2, phase="place")
        yield from c.move(hand=c.pinch(0.032, helpers=(2,), tuck=(3, 4)), seconds=0.6, phase="release")
        yield from c.move(c.palm_for(p0 + [0, 0, .12], R=Rt), Rt, seconds=1.0, phase="retreat")
        yield from c.hold(0.4, "rest")
        p, Rc = c.pose("cube")
        metrics.update(placed_offset_mm=(1000 * (p - p0)).round(1).tolist(),
                       resting_height_mm=round(1000 * (p[2] - TABLE_Z), 1),
                       final_yaw_deg=round(float(np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))), 1))
        upright = abs(Rc[2, 2]) > .99
        outcome = "success" if upright and np.linalg.norm(metrics["placed_offset_mm"][:2]) < 15 else "placed_off_target"
    finally:
        c.finish("cube_pinch", outcome, metrics, HERE / "results")


routines = {"pick, turn and place": pick_and_place}
