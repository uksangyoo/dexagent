"""IMG_4783: take the pen up in a three-finger writing grasp, write a short stroke on the table, lift.

The clip picks the pen off the table. On this hand that pickup is not reliable
(the 16 x 20 mm fingertip shells meet a 7.5 mm barrel above its equator and
ride over it; see ../pen_finger_roll/pen_grasp.py), so the pen starts in a cup
the clip does not show and is taken out in a tripod: thumb and index across
the barrel 30 mm below its top, middle finger on the third side
(../pen_finger_roll/pen_cup.py). The hand then tilts the pen 30 deg, lowers it
until its tip touches the table (contact read from the solver, with a 1.5 mm
press), draws a 40 mm stroke with the tip on the table, and lifts.

Measured: tip-table contact over the stroke (fraction of control steps), the
inked length (path of the tip while in contact), the pen's slip in the grasp,
and that it is still held after the lift.

Assumed: pen 7.5 mm x 140 mm, 5 g, friction 0.8; table friction 1.0 (Newton
default for the slab); hand pads 0.9. The pen has no ink model; "inked" means
the tip was in contact.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.scene import table_scene, WORK
from dexagent.workcell import TABLE_Z
from dexagent.hand import Controller
from dexagent.objects import PEN_RADIUS, PEN_HALF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pen_finger_roll"))
from pen_cup import add_cup_and_pen, pick_from_cup  # noqa: E402

CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4783_pen_writing_grasp.mov"
TILT = np.radians(30)          # pen lean from vertical while writing
PRESS = 0.0008                 # palm lowered this far past first tip contact
STROKE_MM = 40.0
SHOW = np.array([-0.02, -0.12, -0.02])  # write well clear of the cup

scene = table_scene("pen_writing_grasp")
add_cup_and_pen(scene, WORK + [0.0, 0.10, 0.0])


def pen_tip(c):
    """World position of the pen's lower end."""
    p, R = c.pose("pen")
    a = R[:, 2] if R[2, 2] < 0 else -R[:, 2]
    return p + a * (PEN_HALF + PEN_RADIUS)


def touching_table(c):
    lab = list(c.sim.model.shape_label)
    if not hasattr(c, "_table_shapes"):
        c._table_shapes = [i for i, l in enumerate(lab) if l == "table"]
    c.sim.contacts.update(c.sim.step_count)
    return c.sim.contacts.in_contact(c.sim.scene.handles.objects["pen"].shapes, c._table_shapes)


def write(sim, session=None):
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        p0, _ = c.pose("pen")
        yield from pick_from_cup(c, lift=0.10, show=SHOW)
        if c.pose("pen")[0][2] - p0[2] < .06:
            outcome = "dropped_on_lift"; return
        grip0 = c.in_palm("pen")[0]

        # Touch down with the pen hanging, then lean the hand about the tip:
        # the table holds the tip, so the lean does not swing back through the
        # grip under gravity as it does in the air (measured: 30 deg of palm
        # turn gave 12 deg of pen lean).
        gap = pen_tip(c)[2] - TABLE_Z
        yield from c.move(c.xyz - [0, 0, max(0.0, gap - 0.006)], seconds=1.0, phase="lower")
        steps = 0
        while not touching_table(c) and steps < 150:
            c.xyz = c.xyz - [0, 0, 0.0004]
            c.arm = c.kin.solve_arm(c.xyz, c.R, c.arm, tol=0.004)
            c.step(); steps += 1
            yield
        if not touching_table(c):
            outcome = "tip_never_touched"; return
        yield from c.move(c.xyz - [0, 0, PRESS], seconds=0.3, phase="press")
        tip = pen_tip(c)
        R0, x0 = c.R.copy(), c.xyz.copy()
        axis = R0 @ np.array([0, 1.0, 0])     # across the pinch line
        n = 60
        c.phase = "writing_posture"
        for k in range(1, n + 1):
            u = k / n
            Rk = Rotation.from_rotvec(TILT * (u * u * (3 - 2 * u)) * axis).as_matrix()
            c.R = Rk @ R0
            c.xyz = tip + Rk @ (x0 - tip)
            if not touching_table(c):
                c.xyz = c.xyz - [0, 0, 0.0003]
            c.arm = c.kin.solve_arm(c.xyz, c.R, c.arm, tol=0.004)
            c.step()
            yield
        yield from c.hold(0.3)
        _, R_pen = c.pose("pen")
        metrics["pen_lean_deg"] = round(float(np.degrees(np.arccos(abs(R_pen[2, 2])))), 1)

        # Stroke, away from where the tip trails (a pull stroke).
        c.phase = "stroke"
        start = c.xyz.copy()
        trail = pen_tip(c) - (c.xyz + c.R @ np.array([0.086, 0.038, 0.112]))
        trail[2] = 0
        stroke = -STROKE_MM / 1000 * trail / np.linalg.norm(trail)
        n = 125
        tips, contact = [], []
        drop = 0.0
        for k in range(1, n + 1):
            u = k / n
            c.xyz = start + (u * u * (3 - 2 * u)) * stroke - [0, 0, drop]
            if not touching_table(c):
                drop += 0.0003                  # keep the tip on the paper
            c.arm = c.kin.solve_arm(c.xyz, c.R, c.arm, tol=0.004)
            c.step()
            tips.append(pen_tip(c)); contact.append(touching_table(c))
            yield
        tips, contact = np.array(tips), np.array(contact)
        inked = tips[contact]
        metrics["contact_fraction"] = round(float(contact.mean()), 3)
        metrics["inked_mm"] = round(1000 * float(np.sum(np.linalg.norm(np.diff(inked[:, :2], axis=0), axis=1))) if len(inked) > 1 else 0.0, 1)
        metrics["stroke_commanded_mm"] = STROKE_MM
        metrics["grip_slip_mm"] = round(1000 * float(np.linalg.norm(c.in_palm("pen")[0] - grip0)), 1)

        yield from c.move(c.xyz + [0, 0, 0.06], seconds=1.0, phase="lift")
        yield from c.hold(0.5, "hold")
        z = pen_tip(c)[2] - TABLE_Z
        metrics["tip_height_after_lift_mm"] = round(1000 * z, 1)
        held = z > 0.03
        wrote = metrics["contact_fraction"] >= 0.8 and metrics["inked_mm"] >= 25
        outcome = "success" if held and wrote else ("dropped" if not held else "partial(stroke)")
    finally:
        c.finish("pen_writing_grasp", outcome, metrics, HERE / "results")


routines = {"tripod grasp and write": write}
