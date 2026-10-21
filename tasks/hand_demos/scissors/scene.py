"""IMG_4778 / IMG_4779: pick up a pair of scissors by the handle loops, open and close the blades, set them down.

The scissors are articulated (scissors.xml): part B turns on a pivot screw
with 0.04 N m friction, and its loop spreads away from part A's as the blades
open. The index goes into loop A and the middle into loop B, curled straight
down. Squeezing them together (adduction) closes the blades and holds the
scissors; spreading them (abduction) pushes the loops' outer walls apart and
opens the blades. A thumb-index pinch was tried first: the DG5F thumb meets a
9 mm loop wall with its rounded lower cap and slides out when lifting, and
pitching the palm to fix that drives the palm into the table.

Measured: the lift and hold height, the blade opening angle (hinge
coordinate) through two open/close cycles, and a second lift. The open/close
cycles run with the scissors resting on the table: spreading the fingers
relaxes the squeeze that holds them up, and in the air they dropped.

Difference from the clips: the person uses thumb and fingers in the loops
with the hand rolled; here two fingers do the opening and closing.

Assumed numbers (not measured): 190 mm, 58 g, steel blades friction 0.5,
plastic loops friction 0.7, loop holes 32 x 24 mm.
Only robot joint targets are written.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robosimstudio.specs import AssetSpec, StaticSpec, ObjectSpec
from dexagent.rss.scene import table_scene, WORK
from dexagent.rss.workcell import hand_down, TABLE_Z, YAW
from dexagent.rss.hand import Controller, CURL, OPEN

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4778_scissors_open_close_a.mov"
# Index in loop A, middle in loop B: both fingertips curl straight down into
# the loops, so their sides (not their rounded ends) meet the loop walls, and
# finger abduction spreads or squeezes them along the palm's y axis.
HAND = hand_down(YAW)
SPREAD_AXIS = np.array([0.0, 1.0, 0.0])            # palm frame: middle -> index
GRIP_CENTRE = np.array([0.100, 0.016, 0.125])      # palm frame; both reach 16-34 mm either side (CPU IK search)
LOCAL_Y = HAND @ SPREAD_AXIS                       # loop A (index) side
LOCAL_X = np.cross(LOCAL_Y, [0, 0, 1.0])           # blades: far-left, as in the clip
QUAT = tuple(Rotation.from_matrix(np.column_stack([LOCAL_X, LOCAL_Y, [0, 0, 1.0]])).as_quat())
PIVOT = WORK + np.array([0.07, -0.05, 0.0058])
LOOPS = np.array([-0.065, 0.0, 0.0])                  # midpoint between the loop holes (local)
LOOP_HALF = 0.024                                   # loop-hole centres either side of LOOPS, closed
SITE_HEIGHT = 0.003                                 # pad-centre height over the table inside the loops
HOLD_HALF = 0.004                                   # fingertip half-separation commanded to squeeze shut
OPEN_ANGLE = np.radians(35)                         # commanded blade opening

scene = table_scene("scissors")
scene.add(StaticSpec(name="scissors_asset", asset=AssetSpec(path=str(HERE / "scissors.xml"))))
scene.add(ObjectSpec(name="scissors", body_name="scissors", joint_name="scissors_free", pose=tuple(PIVOT), quat=QUAT,
                     obb_grasp=False))


def hinge_angle(sim):
    model = sim.model
    if not hasattr(hinge_angle, "index"):
        labels = list(model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("scissors_hinge"))
        hinge_angle.index = int(model.joint_q_start.numpy()[j])
    return float(sim.state.joint_q.numpy()[hinge_angle.index])


def grip_q(c, half):
    """Index and middle fingertips `half` either side of GRIP_CENTRE along SPREAD_AXIS; thumb, ring and little clear."""
    q = OPEN.copy()
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    return c.fingertips({1: GRIP_CENTRE + half * SPREAD_AXIS, 2: GRIP_CENTRE - half * SPREAD_AXIS}, q)


def holes_palm(c, angle):
    """Palm-frame centres of loop A's hole and of loop B's hole with the blades open by `angle`."""
    p, R = c.pose("scissors")
    a = p + R @ np.array([-0.065, 0.024, 0.0])
    ca, sa = np.cos(angle), np.sin(angle)
    b_local = np.array([-0.065 * ca + 0.024 * sa, -0.065 * sa - 0.024 * ca, 0.0])
    return c.palm_frame_target(a), c.palm_frame_target(p + R @ b_local)


def work_q(c, angle):
    """Index held in loop A; middle tracking loop B's hole along its arc to `angle`."""
    a, b = holes_palm(c, angle)
    a[0] = b[0] = GRIP_CENTRE[0]   # keep the fingertip depth
    q = OPEN.copy()
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    return c.fingertips({1: a, 2: b}, q)


def loops_world(c):
    p, R = c.pose("scissors")
    return p + R @ LOOPS


def cut(sim, session=None):
    c = Controller(sim, session, objects=("scissors",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        mid = loops_world(c)
        R = HAND
        target = mid.copy()
        target[2] = TABLE_Z + SITE_HEIGHT
        into = grip_q(c, LOOP_HALF)
        yield from c.move(c.palm_for(target + [0, 0, .07], centre=GRIP_CENTRE, R=R), R, into, 2.0, "approach")
        yield from c.move(c.palm_for(target, centre=GRIP_CENTRE, R=R), R, seconds=1.2, phase="into_loops")
        # Squeeze the loops together: the blades close and the scissors are held.
        yield from c.move(hand=grip_q(c, HOLD_HALF), seconds=0.6, phase="squeeze")
        yield from c.hold(0.3)
        z0 = c.pose("scissors")[0][2]
        yield from c.move(c.xyz + [0, 0, .09], seconds=1.8, phase="lift")
        lift = c.pose("scissors")[0][2] - z0
        metrics["lift_mm"] = round(1000 * lift, 1)
        yield from c.hold(0.5, "hold")
        metrics["held_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - z0), 1)
        # Spreading the fingers relaxes the squeeze that holds the scissors up,
        # so the blades are worked with the scissors resting on the table.
        yield from c.move(c.xyz - [0, 0, .09], seconds=1.5, phase="set_down")
        if c.tips_world()[1, 2] > TABLE_Z + 0.02:
            outcome = "fingers_out_of_loops"
            return
        # The middle fingertip follows loop B's hole around the pivot (computed
        # from the measured scissors pose), a few degrees past the commanded
        # angle so it keeps pressing the wall; the index holds loop A.
        angles = []
        for cycle in range(2):
            for end, phase, lead in ((OPEN_ANGLE, "open", np.radians(4)), (0.0, "close", -np.radians(8))):
                start = hinge_angle(sim)   # each stroke starts from where the blades are
                for _ in c.move(hand=lambda u: work_q(c, start + u * (end - start) + lead), seconds=2.0,
                                phase=f"{phase}_{cycle + 1}"):
                    angles.append(hinge_angle(sim))
                    yield
                yield from c.hold(0.3)
                angles.append(hinge_angle(sim))
        metrics.update(blade_angle_deg=[round(float(np.degrees(min(angles))), 1), round(float(np.degrees(max(angles))), 1)],
                       closed_after_deg=round(float(np.degrees(angles[-1])), 1))
        # Pick them up closed once more, then set them down and let go.
        yield from c.move(c.xyz + [0, 0, .06], seconds=1.2, phase="lift_again")
        metrics["second_lift_mm"] = round(1000 * (c.pose("scissors")[0][2] - z0), 1)
        yield from c.move(c.xyz - [0, 0, .06], seconds=1.2, phase="set_down_again")
        yield from c.move(hand=grip_q(c, LOOP_HALF), seconds=0.5, phase="release")
        yield from c.move(c.xyz + [0, 0, .08], seconds=1.0, phase="retreat")
        yield from c.hold(0.4, "rest")
        metrics["rest_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - TABLE_Z), 1)
        opened = np.degrees(max(angles)) > 10
        closed = np.degrees(angles[-1]) < 3
        lifted = metrics["held_height_mm"] > 50 and metrics["second_lift_mm"] > 40
        if opened and closed and lifted:
            outcome = "success"
        else:
            outcome = "_".join(["opened" if opened else "not_opened", "closed" if closed else "not_closed",
                                "lifted" if lifted else "not_lifted"])
    finally:
        c.finish("scissors", outcome, metrics, HERE / "results")


routines = {"pick up, open and close": cut}
