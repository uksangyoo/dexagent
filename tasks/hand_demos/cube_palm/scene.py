"""IMG_4782: pick the 40 mm foam cube up with the fingertips, turn the hand palm-up and
bring the cube into the palm, hold it on the open palm, close the hand over it, turn
back and set it on the table.

Cube: 40 mm, 12 g, friction 0.9 (foam on the hand's pads; nominal, not measured).
The person returns the cube to a fingertip pinch before setting it down; this
routine sets it down out of the closed hand instead (see `return_to_table`).
"""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio.specs import CameraSpec
from dexagent.scene import table_scene, WORK
from dexagent.workcell import TABLE_Z, YAW
from dexagent.hand import Controller
from palm_transfer import EDGE, to_palm, open_palm, resting_on_palm, return_to_table, note

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4782_cube_palm_roll.mov"
# Where the clip's cube sits in the phone view. The clip's cube is turned about
# 40 deg; this one is square to the table, which is the start the routine is
# tuned for (turned to the hand's yaw, it lands toward the thumb side and rolls off).
START = WORK + np.array([-0.01, -0.024, EDGE / 2])
YAW0 = float(__import__("os").environ.get("CUBE_YAW", "0"))

scene = table_scene("cube_palm")
# The hand turns over 320 mm above the table, above the phone camera's frame top.
scene.add(CameraSpec(name="side", width=1280, height=720, fov_y=40,
                     eye=(0.42, 0.42, TABLE_Z + 0.52), look_at=(0.55, 0.0, TABLE_Z + 0.30)))
scene.add(rss.box(name="cube", half=(EDGE / 2,) * 3, at=tuple(START), quat=(0, 0, np.sin(YAW0 / 2), np.cos(YAW0 / 2)), mass=0.012, friction=0.9, color=(0.42, 0.28, 0.72)))


def palm_roll(sim, session=None):
    c = Controller(sim, session, objects=("cube",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, _ = c.pose("cube")
        yield from to_palm(c, p0)
        yield from open_palm(c)
        # 1.5 s on the open palm, counting the steps the palm alone carries it.
        c.phase, on_palm = "on palm", 0
        for _ in range(75):
            c.step()
            yield
            on_palm += resting_on_palm(c)
        note(c, "on palm")
        metrics["open_palm_hold_s"] = round(on_palm / 50, 2)
        if on_palm < 40:
            outcome = "not_on_palm"
            return
        yield from return_to_table(c, p0)
        p, R = c.pose("cube")
        metrics.update(placed_offset_mm=(1000 * (p - p0 - [0, 0.05, 0])).round(1).tolist(),
                       resting_height_mm=round(1000 * (p[2] - TABLE_Z), 1))
        flat = np.abs(R[2]).max() > 0.99  # some face down
        on_table = abs(p[2] - TABLE_Z - EDGE / 2) < 0.003
        ok = flat and on_table and np.linalg.norm(metrics["placed_offset_mm"][:2]) < 30
        outcome = "success" if ok else "placed_off_target"
    finally:
        c.finish("cube_palm", outcome, metrics, HERE / "results")


routines = {"palm roll and place": palm_roll}
