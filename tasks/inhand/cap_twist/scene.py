"""Unscrew a jar lid with the fingertips: grip the rim, twist, let go, re-grip, repeat.

The palm stays put; the lid turns because the thumb, index and middle pads
walk around its rim (a fingertip gait). The jar is fixed to the table. The lid
turns on a hinge about the jar axis with a 0.02 N m friction loss standing in
for thread friction; the thread's lead (the lid rising) is not modelled.
Assumed, not measured: lid 70 mm across, 15 mm tall, 18 g, friction 1.0.
Done when the lid has turned 360 deg counter-clockwise seen from above.
"""
from pathlib import Path
import numpy as np
import robosimstudio as rss
from robosimstudio.specs import AssetSpec, CameraSpec, StaticSpec
from dexagent.rss.scene import table_scene, WORK
from dexagent.rss.workcell import hand_down
from dexagent.rss.hand import OPEN
from twist import CapController, CENTRE, PHI, rim_targets

HERE = Path(__file__).resolve().parent
JAR = WORK + np.array([0.0, 0.0, 0.0])
LID_Z = 0.1155            # lid centre above the table
RADIUS = 0.035
GOAL = 360.0              # degrees, counter-clockwise from above
STROKE = 30.0             # each twist runs the pads from -STROKE to +STROKE around the rim

scene = table_scene("cap_twist")
scene.add(StaticSpec(name="jar", asset=AssetSpec(path=str(HERE / "jar.xml")), xform=(*JAR, 0, 0, 0, 1)))
scene.add(CameraSpec(name="jar_view", width=1280, height=720, fov_y=40,
                     eye=(JAR[0] + 0.30, JAR[1] + 0.26, JAR[2] + 0.33), look_at=(JAR[0], JAR[1], JAR[2] + 0.10)))
scene.add(CameraSpec(name="top", width=1280, height=720, fov_y=45,
                     eye=(JAR[0] + 0.12, JAR[1] - 0.02, JAR[2] + 0.45), look_at=(JAR[0], JAR[1], JAR[2] + 0.1)))


def unscrew(sim, session=None):
    c = CapController(sim, session)
    outcome, metrics = "incomplete", {}
    lid = JAR + np.array([0, 0, LID_Z])
    try:
        yield from c.hold(0.4, "settle")
        R = hand_down()
        palm = lid - R @ CENTRE
        yaw0 = c.palm_yaw()
        start = c.lid_angle()
        grip = lambda deg, rad: rim_targets(c, lid, R, palm, np.radians(deg), rad)
        yield from c.move(palm + [0, 0, .08], R, grip(-STROKE, RADIUS + .014), 1.5, "approach")
        yield from c.move(palm, R, seconds=1.0, phase="descend")
        strokes, slips = 0, []
        while strokes < 12 and np.degrees(c.lid_angle() - start) < GOAL:
            before = c.lid_angle()
            yield from c.move(hand=grip(-STROKE, RADIUS - .007), seconds=0.5, phase=f"grip {strokes + 1}")
            yield from c.hold(0.15)
            yield from c.move(hand=lambda u: grip(-STROKE + 2 * STROKE * u, RADIUS - .007), seconds=1.2,
                              phase=f"twist {strokes + 1}")
            yield from c.move(hand=grip(STROKE, RADIUS + .014), seconds=0.4, phase=f"release {strokes + 1}")
            turned = np.degrees(c.lid_angle() - before)
            slips.append(round(turned / (2 * STROKE), 3))
            strokes += 1
            yield from c.move(hand=lambda u: grip(STROKE - 2 * STROKE * u, RADIUS + .014), seconds=0.7,
                              phase=f"return {strokes}")
        yield from c.move(palm + [0, 0, .10], R, OPEN, 1.0, "retreat")
        yield from c.hold(0.5, "rest")
        total = float(np.degrees(c.lid_angle() - start))
        metrics.update(lid_turn_deg=round(total, 1), strokes=strokes,
                       turn_per_stroke_fraction=slips,
                       palm_yaw_change_deg=round(float(np.degrees(c.max_yaw_change(yaw0))), 2),
                       peak_grip_force_N=c.peak_force.round(2).tolist())
        outcome = "success" if total >= GOAL else "partial"
    finally:
        c.finish("cap_twist", outcome, metrics, HERE / "results")


routines = {"unscrew the lid": unscrew}
