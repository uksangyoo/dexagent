"""IMG_4778 / IMG_4779: pick up a pair of scissors by the handle loops, open and close the blades, set them down.

The scissors are articulated (scissors.xml): part B turns on a pivot screw
with 0.04 N m friction, and its loop spreads away from part A's as the blades
open. As in the clips, the thumb goes into one loop and a finger into the
other: the thumb into loop A and the index into loop B, along the hand's
thumb-index pinch axis, with the blades pointing left.

Squeezing the thumb and index together closes the blades against their stop
and holds the scissors by friction. That hold only lasts a short lift: the
loops are 9 mm walls and slide off the rounded fingertips within about two
seconds. So, like the person, the hand rolls 90 deg about the finger
direction once the scissors are clear of the table. The fingers are then
horizontal through the loops, the blades hang below, and the loops rest on
the fingers. The blades are opened and closed in the air in that pose: the
thumb holds loop A while the index pushes loop B along its arc about the
pivot. The hand then rolls back, sets the scissors down and lets go.

Measured: the lift height, the height held after the roll, the blade opening
angle (hinge coordinate) through two open/close cycles, whether the scissors
are still in the hand after them, and where they come to rest.

Differences from the clips: one finger (the index) works loop B where the
person uses two, and the hand rolls further than the person's (about 90 deg
against their 45 deg) so the loops hang on the fingers.

Assumed numbers (not measured): 190 mm, 58 g, steel blades friction 0.5,
plastic loops friction 0.7, loop holes 35 x 27 mm.
Only robot joint targets are written.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robosimstudio.specs import AssetSpec, CameraSpec, StaticSpec, ObjectSpec
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, TABLE_Z, YAW
from dexagent.hand import Controller, PINCH_CENTRE, PINCH_AXIS
from dexagent.retarget.replay import make_replay_routine

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4778_scissors_open_close_a.mov"
TRACKS = HERE / "tracks"
HAND = hand_down(YAW)
# Loop A toward the thumb (the wrist side of the pinch), loop B toward the
# index; the blades then point along the palm's +y, to the left as in the clip.
LOCAL_Y = HAND @ PINCH_AXIS
LOCAL_X = np.cross(LOCAL_Y, [0, 0, 1.0])
QUAT = tuple(Rotation.from_matrix(np.column_stack([LOCAL_X, LOCAL_Y, [0, 0, 1.0]])).as_quat())
LOOPS = np.array([-0.065, 0.0, 0.0])        # midpoint between the loop holes (scissors frame)
HOLE_A = np.array([-0.065, 0.024, 0.0])     # loop A's hole centre (scissors frame)
LOOPS_WORLD = WORK + np.array([0.05, 0.0, 0.0])
PIVOT = LOOPS_WORLD - Rotation.from_quat(QUAT).as_matrix() @ LOOPS + [0, 0, 0.0058]
PAD_HEIGHT = 0.013     # pad centres over the table: the finger ends reach about 11 mm below the pads
INTO = 0.024           # thumb and index pads at the two hole centres
SQUEEZE = 0.004        # commanded half-separation: the pads stop on the inner walls with the blades shut
LIFT = 0.05            # friction lift before the roll
HANG = 0.22            # pinch height above the table once rolled: the blades hang about 170 mm below
ROLL = -np.pi / 2      # about the finger direction: the palm turns to face away from the thumb
OPEN_ANGLE = np.radians(30)
TUCK = (2, 3, 4)

scene = table_scene("scissors")
scene.add(StaticSpec(name="scissors_asset", asset=AssetSpec(path=str(HERE / "scissors.xml"))))
scene.add(ObjectSpec(name="scissors", body_name="scissors", joint_name="scissors_free", pose=tuple(PIVOT), quat=QUAT,
                     obb_grasp=False))
# A low camera across the table from the arm, for the fingers in the loops.
scene.add(CameraSpec(name="loops", width=640, height=480, fov_y=45, eye=(0.80, 0.12, TABLE_Z + 0.14),
                     look_at=(0.60, -0.04, TABLE_Z + 0.05)))


def hinge_angle(sim):
    model = sim.model
    if not hasattr(hinge_angle, "index"):
        labels = list(model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("scissors_hinge"))
        hinge_angle.index = int(model.joint_q_start.numpy()[j])
    return float(sim.state.joint_q.numpy()[hinge_angle.index])


def rolled(phi):
    return HAND @ Rotation.from_rotvec([0, 0, phi]).as_matrix()


def hole_b(angle):
    """Loop B's hole centre (scissors frame) with the blades open by `angle`."""
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array([-0.065 * ca + 0.024 * sa, -0.065 * sa - 0.024 * ca, 0.0])


def work_q(c, angle, push):
    """Thumb held in loop A; index on loop B's arc at `angle`, `push` m past the hole centre
    along the arc (positive opens), from the measured scissors pose."""
    p, R = c.pose("scissors")
    a = c.palm_frame_target(p + R @ HOLE_A)
    b_local = hole_b(angle)
    tangent = np.array([-b_local[1], b_local[0], 0.0]) / np.linalg.norm(b_local[:2])   # +angle direction
    b = c.palm_frame_target(p + R @ (b_local - push * tangent))
    a[0] = b[0] = PINCH_CENTRE[0]     # keep the fingertip depth in the loops
    a[1] = b[1] = PINCH_CENTRE[1]
    return c.fingertips({0: a, 1: b}, c.pinch(INTO, tuck=TUCK))


def cut(sim, session=None):
    c = Controller(sim, session, objects=("scissors",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        p, R = c.pose("scissors")
        grip = p + R @ LOOPS
        grip[2] = TABLE_Z + PAD_HEIGHT
        z0 = p[2]
        into = c.pinch(INTO, tuck=TUCK)
        yield from c.move(c.palm_for(grip + [0, 0, .07], R=HAND), HAND, into, 2.0, "approach")
        yield from c.move(c.palm_for(grip, R=HAND), HAND, seconds=1.2, phase="into_loops")
        yield from c.move(hand=c.pinch(SQUEEZE, tuck=TUCK), seconds=0.6, phase="squeeze")
        yield from c.hold(0.3)
        yield from c.move(c.palm_for(grip + [0, 0, LIFT], R=HAND), HAND, seconds=1.0, phase="lift")
        metrics["lift_mm"] = round(1000 * (c.pose("scissors")[0][2] - z0), 1)
        top = grip + [0, 0, HANG - PAD_HEIGHT]
        yield from c.move(c.palm_for(top, R=rolled(ROLL)), rolled(ROLL), seconds=2.0, phase="roll")
        yield from c.hold(0.6, "hang")
        metrics["held_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - z0), 1)
        if metrics["held_height_mm"] < 50:
            outcome = "dropped"
            return
        # Open and close in the air: the index follows loop B's arc a little
        # ahead of where the blades are, so it keeps pushing the wall.
        angles = []
        for cycle in range(2):
            for end, phase, push in ((OPEN_ANGLE, "open", 0.006), (0.0, "close", -0.006)):
                start = hinge_angle(sim)
                for _ in c.move(hand=lambda u: work_q(c, start + u * (end - start), push), seconds=2.0,
                                phase=f"{phase}_{cycle + 1}"):
                    angles.append(hinge_angle(sim))
                    yield
                yield from c.hold(0.3)
                angles.append(hinge_angle(sim))
        metrics.update(blade_angle_deg=[round(float(np.degrees(min(angles))), 1), round(float(np.degrees(max(angles))), 1)],
                       closed_after_deg=round(float(np.degrees(angles[-1])), 1),
                       held_after_cycles_mm=round(1000 * (c.pose("scissors")[0][2] - z0), 1))
        # Squeeze shut, roll back and set them down.
        yield from c.move(hand=c.pinch(SQUEEZE, tuck=TUCK), seconds=0.6, phase="squeeze_again")
        yield from c.move(c.palm_for(grip + [0, 0, LIFT], R=HAND), HAND, seconds=2.0, phase="unroll")
        yield from c.move(c.palm_for(grip + [0, 0, 0.002], R=HAND), HAND, seconds=1.0, phase="set_down")
        yield from c.move(hand=into, seconds=0.5, phase="release")
        yield from c.move(c.palm_for(grip + [0, 0, .09], R=HAND), HAND, seconds=1.0, phase="retreat")
        yield from c.hold(0.4, "rest")
        metrics["rest_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - TABLE_Z), 1)
        opened = np.degrees(max(angles)) > 10
        closed = np.degrees(angles[-1]) < 3
        held = metrics["held_height_mm"] > 100 and metrics["held_after_cycles_mm"] > 100
        set_down = metrics["rest_height_mm"] < 15
        if opened and closed and held and set_down:
            outcome = "success"
        else:
            outcome = "_".join(["opened" if opened else "not_opened", "closed" if closed else "not_closed",
                                "held" if held else "dropped", "set_down" if set_down else "not_set_down"])
    finally:
        c.finish("scissors", outcome, metrics, HERE / "results")


routines = {
    "pick up, open and close": cut,
    "retarget replay": make_replay_routine(
        (TRACKS / "from_video.json", TRACKS / "synthetic.json"),
        objects=("scissors",),
        results_dir=HERE / "results",
    ),
}
