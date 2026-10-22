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
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down
from dexagent.hand import OPEN
from dexagent.cases import current
from twist import CapController, CENTRE, PHI, rim_targets

HERE = Path(__file__).resolve().parent
# Initial conditions (dexagent.cases; nominal at severity 0, full range at 1):
# jar position +-20 mm, lid radius 35 mm +-10 % (jar body follows), lid mass 18 g +-50 %,
# thread friction 0.02 N m +-50 %.
CASE = current()
_DX, _DY = CASE.u(0.020, "dx"), CASE.u(0.020, "dy")
RADIUS = 0.035 * CASE.scale(0.10, "lid_radius")
LID_MASS = 0.018 * CASE.scale(0.5, "lid_mass")
THREAD = 0.02 * CASE.scale(0.5, "thread_friction")
JAR = WORK + np.array([_DX, _DY, 0.0])
LID_Z = 0.1155            # lid centre above the table
GOAL = 360.0              # degrees, counter-clockwise from above
CLEAR, LIFT = 0.018, 0.008  # released pads: this far outside the rim and above mid-height
STROKE = 30.0             # each twist runs the pads from -STROKE to +STROKE around the rim



def _jar_xml():
    """jar.xml at nominal; otherwise a copy with the perturbed lid radius, mass and thread friction."""
    if CASE.severity == 0:
        return HERE / "jar.xml"
    import os, tempfile
    text = (HERE / "jar.xml").read_text()
    for old, new in [('size="0.04 0.05"', f'size="{RADIUS + 0.005:.5f} 0.05"'),
                     ('size="0.036 0.0035"', f'size="{RADIUS + 0.001:.5f} 0.0035"'),
                     ('size="0.035 0.0075"', f'size="{RADIUS:.5f} 0.0075"'),
                     ('frictionloss="0.02"', f'frictionloss="{THREAD:.5f}"'),
                     ('mass="0.018"', f'mass="{LID_MASS:.5f}"'),
                     ('size="0.028 0.004 0.0005"', f'size="{0.8 * RADIUS:.5f} 0.004 0.0005"'),
                     ('pos="0.022 0 0.0084"', f'pos="{0.63 * RADIUS:.5f} 0 0.0084"')]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    d = Path(tempfile.gettempdir()) / "dexagent_cap_twist"
    d.mkdir(exist_ok=True)
    path = d / f"jar_{CASE.seed}_{CASE.severity}_{os.getpid()}.xml"
    path.write_text(text)
    return path


scene = table_scene("cap_twist")
scene.add(StaticSpec(name="jar", asset=AssetSpec(path=str(_jar_xml())), xform=(*JAR, 0, 0, 0, 1)))
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
        grip = lambda deg, rad, lift=0.0: rim_targets(c, lid, R, palm, np.radians(deg), rad, lift)
        yield from c.move(palm + [0, 0, .08], R, grip(-STROKE, RADIUS + .014), 1.5, "approach")
        yield from c.move(palm, R, seconds=1.0, phase="descend")
        strokes, slips = 0, []
        while strokes < 12 and np.degrees(c.lid_angle() - start) < GOAL:
            before = c.lid_angle()
            yield from c.move(hand=grip(-STROKE, RADIUS - .007), seconds=0.5, phase=f"grip {strokes + 1}")
            yield from c.hold(0.15)
            yield from c.move(hand=lambda u: grip(-STROKE + 2 * STROKE * u, RADIUS - .007), seconds=1.2,
                              phase=f"twist {strokes + 1}")
            yield from c.move(hand=grip(STROKE, RADIUS + CLEAR, LIFT), seconds=0.4, phase=f"release {strokes + 1}")
            turned = np.degrees(c.lid_angle() - before)
            slips.append(round(turned / (2 * STROKE), 3))
            strokes += 1
            yield from c.move(hand=lambda u: grip(STROKE - 2 * STROKE * u, RADIUS + CLEAR, LIFT), seconds=0.7,
                              phase=f"return {strokes}")
        yield from c.move(palm + [0, 0, .06], R, seconds=0.8, phase="retreat")
        yield from c.move(palm + [0, 0, .10], R, OPEN, 0.6)
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
