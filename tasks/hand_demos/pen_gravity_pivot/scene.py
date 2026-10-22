"""IMG_4774 / IMG_4775: hold a pen level near one end, let it swing down under gravity in the pinch, catch it upright.

Prepared start, not a table pickup: the clips begin with the pen already in
the hand. The pen rests on a small block (a pen rest) so the fingertips can
reach its equator; a 7.5 mm pen lying flat on the table cannot be held level
by these 16-20 mm thick fingertips (see pen_pickup_pivot for the table pickup).

Mechanism. The pen's contacts have no torsional friction (`grip.smooth_pen`:
its geom takes priority over the hand's condim 4, as a hard smooth barrel), so
a thumb-index pinch alone cannot stop the pen turning about the pinch axis.
The level hold therefore uses three contacts, as people do: the thumb on one
side, index and middle on the other, spaced 25 mm along the pen, so friction at
the spaced pair resists gravity's turn. Lifting the middle finger off leaves
only the thumb-index pinch, and the pen swings down about it. The middle
finger then comes back beside the index as a lateral stop, which holds the
hanging pen in the thumb-index-middle notch.

Assumed numbers (not measured): pen 5 g, 7.5 x 140 mm, pad friction 0.9 (hand
MJCF). Thumb 45 mm from the pen's centre. Rest: 44 x 20 x 30 mm, wood.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import robosimstudio as rss
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down
from dexagent.hand import Controller
from dexagent.objects import quat_from_axis, PEN_RADIUS, PEN_HALF, PEN_MASS, PEN_GREY
from dexagent.cases import current
import grip as fingers

HERE = Path(__file__).resolve().parent
CLIPS = [HERE.parents[2] / f"reference/hand_demos/IMG_{n}.mov" for n in ("4774_pen_gravity_pivot_a", "4775_pen_gravity_pivot_b")]
GRASP_YAW = 0.15                                      # fingers point almost straight away, as in IMG_4774
ACROSS = np.array([-np.sin(GRASP_YAW), np.cos(GRASP_YAW), 0.0])   # the pen lies across the fingers
OFFSET = 0.045                                        # thumb to pen centre, along the pen [m]
REST_H = 0.03
# Initial conditions (dexagent.cases; nominal at severity 0, full range at 1): pen and
# rest together +-20 mm in x and y and +-25 deg in yaw; pen radius and length +-10 %,
# mass 5 g +-50 %. The hand's grasp yaw stays GRASP_YAW; the routine reads the pen pose.
CASE = current()
_DX, _DY, _DYAW = CASE.u(0.02, "dx"), CASE.u(0.02, "dy"), CASE.u(np.radians(25), "dyaw")
RADIUS = PEN_RADIUS * CASE.scale(0.10, "radius")
HALF = PEN_HALF * CASE.scale(0.10, "length")
MASS = PEN_MASS * CASE.scale(0.5, "mass")
PEN_DIR = np.array([-np.sin(GRASP_YAW + _DYAW), np.cos(GRASP_YAW + _DYAW), 0.0])
BASE = WORK + np.array([_DX, _DY, 0.0])
START = BASE + np.array([0.0, 0.0, REST_H + RADIUS + 0.0005])
REST_AT = BASE + 0.012 * PEN_DIR + np.array([0, 0, REST_H / 2])
PITCH, BETA = 0.5, np.radians(50)
CENTRE = np.array([0.10, 0.02, 0.10])                 # thumb-index-middle grasp centre, palm frame
SPREAD = 0.0125                                       # index and middle either side of the thumb, along the pen
TIGHT, OPEN_W = -0.001, 0.025
STOP_Y, DROP, PARK_BACK = -0.004, 0.015, -0.006        # parked middle fingertip: across, below, ahead of the pinch

scene = table_scene("pen_gravity_pivot")
_yaw = GRASP_YAW + _DYAW + np.pi / 2
scene.add(rss.slab(name="pen_rest", half=(0.022, 0.010, REST_H / 2),
                   at=(*REST_AT, 0.0, 0.0, np.sin(_yaw / 2), np.cos(_yaw / 2)), color=(0.45, 0.33, 0.22)))
# dexagent.objects.pen at the perturbed size and mass. Friction 0.9, the pad friction: the pen pair uses the pen's own.
scene.add(rss.capsule(name="pen", radius=RADIUS, half_height=HALF, at=tuple(START), quat=quat_from_axis(PEN_DIR),
                      mass=MASS, friction=0.9, color=PEN_GREY))
# Built here so the pen's contact material (no torsional friction, grip.smooth_pen) is
# part of the scene for every policy, not only for the expert routine, which sets it again.
sim = scene.build(control_mode="joint_position")
fingers.smooth_pen(sim)


def tilt_from_vertical(axis):
    return float(np.degrees(np.arccos(min(1.0, abs(axis[2])))))


def pivot(sim, session=None, tight=TIGHT, lift_pad=-0.005, spread=SPREAD, beta=BETA, stop_y=STOP_Y, drop=DROP,
          park_back=PARK_BACK):
    fingers.smooth_pen(sim)
    c = Controller(sim, session, objects=("pen",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.4, "settle")
        p0, R0 = c.pose("pen")
        axis0 = R0[:, 2]
        side = np.sign(axis0 @ ACROSS) or 1.0
        grip = p0 - side * OFFSET * axis0
        Ry = Rotation.from_rotvec([0, PITCH, 0]).as_matrix()
        R = hand_down(GRASP_YAW) @ Ry
        axis, down = Ry.T @ np.array([0, 0, -1.0]), Ry.T @ np.array([1.0, 0, 0])
        hand = lambda h, lift=0.0, back=0.0: fingers.tweezer3(c.kin, CENTRE, axis, down, h, spread, beta,
                                                               mid_lift=lift, mid_back=back)[0]
        at = lambda dz: c.palm_for(grip + [0, 0, lift_pad + dz], centre=CENTRE, R=R)
        yield from c.move(at(.08), R, hand(OPEN_W), 1.5, "approach")
        yield from c.hold(0.5)
        yield from c.move(at(0), R, seconds=1.0, phase="descend")
        yield from c.hold(0.4)
        # Correct the few mm of arm sag from the measured pad positions.
        err = p0[2] + lift_pad - (fingers.pads_world(c)[:3, 2].mean() - fingers.TIP_END * np.cos(beta))
        yield from c.move(c.xyz + [0, 0, err], R, seconds=0.3, phase="correct")
        yield from c.hold(0.2)
        yield from c.move(hand=lambda u: hand(OPEN_W + u * (tight - OPEN_W)), seconds=1.0, phase="grasp")
        yield from c.hold(0.3)
        yield from c.move(c.xyz + [0, 0, .11], R, seconds=1.5, phase="lift level")
        yield from c.hold(0.6, "hold level")
        p, Rp = c.pose("pen")
        level = Rp[:, 2].copy()
        metrics["lift_mm"] = round(1000 * (p[2] - p0[2]), 1)
        metrics["level_tilt_from_horizontal_deg"] = round(90 - tilt_from_vertical(level), 1)
        if p[2] - p0[2] < .07:
            outcome = "dropped"; return
        # Release: draw the middle finger back off the pen, leaving only the
        # thumb-index pinch, and park its tip below the pinch just past
        # vertical on the far side. The pen swings down freely under gravity,
        # strikes the parked fingertip on its first pass, and comes to rest
        # against it. (Raising the middle instead lets the pen's short end
        # follow it up: a driven pivot, not a free one.)
        dm = np.cos(beta) * down + np.sin(beta) * axis
        held_q = c.hand.copy()
        def middle(back, y, low):
            q = held_q.copy()
            end = CENTRE - (tight + back) * axis + np.array([0.0, y, 0.0]) + low * down
            q[8:12] = fingers.solve(c.kin, 2, end - fingers.TIP_END * dm, dm, held_q[8:12], 0.02)[0]
            return q
        yield from c.move(hand=lambda u: middle(0.02 * u, -spread, 0.0), seconds=0.08, phase="release middle")
        yield from c.move(hand=lambda u: middle(0.02 * (1 - u) + park_back * u, -spread + u * (spread + stop_y), u * drop),
                          seconds=0.12, phase="park middle")
        swept = 0.0
        for _ in range(60):
            yield from c.hold(0.02, "swing")
            a = c.pose("pen")[1][:, 2]
            swept = max(swept, float(np.degrees(np.arccos(np.clip(abs(a @ level), 0, 1)))))
        metrics["swing_deg"] = round(swept, 1)
        tilts = []
        for _ in range(50):
            yield from c.hold(0.02, "hold upright")
            tilts.append(tilt_from_vertical(c.pose("pen")[1][:, 2]))
        metrics["hold_tilt_range_deg"] = [round(min(tilts), 1), round(max(tilts), 1)]
        p, Rp = c.pose("pen")
        a = Rp[:, 2]
        metrics.update(pivot_swept_deg=round(float(np.degrees(np.arccos(np.clip(abs(a @ level), 0, 1)))), 1),
                       final_tilt_from_vertical_deg=round(tilt_from_vertical(a), 1),
                       final_height_mm=round(1000 * (p[2] - p0[2]), 1))
        held = p[2] - p0[2] > .05
        ok = (held and metrics["level_tilt_from_horizontal_deg"] < 15 and metrics["pivot_swept_deg"] > 60
              and max(tilts) < 20)
        outcome = "success" if ok else ("dropped" if not held else "did_not_pivot")
    finally:
        c.finish("pen_gravity_pivot", outcome, metrics, HERE / "results")


routines = {"hold level, gravity pivot, catch": pivot}
