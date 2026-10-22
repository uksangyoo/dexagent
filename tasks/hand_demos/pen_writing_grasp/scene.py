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
from dexagent.workcell import hand_down
from dexagent.hand import PINCH_CENTRE
from dexagent.objects import quat_from_axis, PEN_RADIUS as _R0, PEN_HALF as _H0, PEN_MASS as _M0, PEN_GREY
from dexagent.cases import current

HERE = Path(__file__).resolve().parent

# Initial conditions (dexagent.cases; nominal at severity 0, full range at 1):
# cup and pen position +-20 mm, pen lean in the cup +-3 deg about x and y
# (the pen then settles against the bore), barrel diameter 7.5 mm +-10 %,
# length 140 mm +-10 %, mass 5 g +-50 %.
CASE = current()
_DX, _DY = CASE.u(0.020, "dx"), CASE.u(0.020, "dy")
_TX, _TY = CASE.u(np.radians(3), "lean_x"), CASE.u(np.radians(3), "lean_y")
PEN_RADIUS = _R0 * CASE.scale(0.10, "diameter")
_SLEN = CASE.scale(0.10, "length")
PEN_HALF = _H0 + ((_H0 + _R0) * (_SLEN - 1) - (PEN_RADIUS - _R0))   # exact at nominal
PEN_MASS = _M0 * CASE.scale(0.5, "mass")
CUP_AT = WORK + np.array([0.0, 0.10, 0.0]) + np.array([_DX, _DY, 0.0])

# The cup and the three-finger pick from ../pen_finger_roll/pen_cup.py, copied
# here so the pen's perturbed size reaches the pick (same logic).
CUP_BORE, CUP_WALL, CUP_HEIGHT = 0.010, 0.003, 0.06   # half bore 10 mm, 60 mm tall (assumed)
GRIP_BELOW_TOP = 0.030
PINCH_HALF = 0.003                                     # pad sites 3 mm either side of the pinch centre: ~2 N each
MIDDLE_HOLD = np.array([0.010, -0.006, 0.0])          # middle pad 10 mm down the barrel, palm -y side (at its abduction limit)


def add_cup_and_pen(scene, at):
    import robosimstudio as rss
    at = np.asarray(at, float)
    for k, (dx, dy, hx, hy) in enumerate([(1, 0, CUP_WALL, CUP_BORE + 2 * CUP_WALL), (-1, 0, CUP_WALL, CUP_BORE + 2 * CUP_WALL),
                                          (0, 1, CUP_BORE, CUP_WALL), (0, -1, CUP_BORE, CUP_WALL)]):
        off = (CUP_BORE + CUP_WALL) * np.array([dx, dy])
        scene.add(rss.slab(name=f"cup_wall_{k}", half=(hx, hy, CUP_HEIGHT / 2),
                           at=(at[0] + off[0], at[1] + off[1], TABLE_Z + CUP_HEIGHT / 2), color=(0.15, 0.32, 0.55)))
    # Lower end 1 mm above the table at the bore centre, leaning by the drawn angles.
    axis = Rotation.from_euler("xy", [_TX, _TY]).apply([0, 0, 1.0])
    centre = at + [0, 0, PEN_HALF + PEN_RADIUS + 0.001] + (axis - [0, 0, 1.0]) * PEN_HALF
    scene.add(rss.capsule(name="pen", radius=PEN_RADIUS, half_height=PEN_HALF, at=tuple(centre),
                          quat=quat_from_axis(axis), mass=PEN_MASS, friction=0.8, color=PEN_GREY))


def pick_from_cup(c, lift=0.10, show=(0, 0, 0), centre=PINCH_CENTRE, middle=MIDDLE_HOLD):
    """Generator: pinch the pen 30 mm below its top and lift it out. Returns the held hand pose."""
    p0, R0 = c.pose("pen")
    grip = p0 + R0[:, 2] * (PEN_HALF + PEN_RADIUS) - np.array([0, 0, GRIP_BELOW_TOP])
    R = hand_down()
    yield from c.move(c.palm_for(grip + [0, 0, .08], centre, R), R, c.pinch(0.026, centre), 1.5, "approach")
    yield from c.move(c.palm_for(grip, centre, R), R, seconds=1.0, phase="descend")
    held = c.fingertips({2: centre + middle}, c.pinch(PINCH_HALF, centre, tuck=(3, 4)))
    yield from c.move(hand=held, seconds=0.8, phase="pinch")
    yield from c.hold(0.3)
    out = grip + [0, 0, lift]
    yield from c.move(c.palm_for(out, centre, R), R, seconds=2.5, phase="lift_out")
    yield from c.move(c.palm_for(out + np.asarray(show), centre, R), R, seconds=1.2, phase="present")
    yield from c.hold(0.5, "steady")
    return held

CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4783_pen_writing_grasp.mov"
TILT = np.radians(30)          # pen lean from vertical while writing
PRESS = 0.0008                 # palm lowered this far past first tip contact
STROKE_MM = 40.0
SHOW = np.array([-0.02, -0.12, -0.02])  # write well clear of the cup

scene = table_scene("pen_writing_grasp")
add_cup_and_pen(scene, CUP_AT)


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
