"""IMG_4778 / IMG_4779: pick up a pair of scissors by the handle loops, open and close the blades, set them down.

The scissors are articulated (scissors.xml): part B turns on a pivot screw
with 0.04 N m friction, and its loop spreads away from part A's as the blades
open. The index goes into loop A and the middle into loop B, curled straight
down. Squeezing them together (abduction joints only) closes the blades
against their stop and grips the loops; spreading them pushes the loops'
outer walls apart and opens the blades.

A straight friction lift fails: with the finger ends clear of the table only
about 9 mm of finger is in the loops, and the loops slide off the domed
fingertips within a couple of seconds. So the scissors are stood up the way
one lifts a ladder by one end: the blade tips stay on the table while the
loops rise on an arc around them and the hand pitches with the loops, fingers
down, to 100 deg. Leaning just past vertical, the scissors rest on the
fingers like a stick pinned at its top; the squeeze is relaxed, the fingers
are pushed 14 mm further into the loops, and the hand lifts the blade tips
off the table. The loops then hang on the curled fingers, and the blades are
opened and closed twice in the air. The hand lowers the tips back to the
table, backs the fingers out to their first depth and lays the scissors
down about the tips.

The loops still creep on the fingers while standing up (about 2 mm/s, loop B
off the middle finger), which sets the stand-up time: at 4 s or more the
middle finger leaves loop B before vertical; at 3 s or less the palm trails
its path and the fingers lever the blade tips off the table.

A thumb-in-a-loop grasp, as in the clips, was tried: the DG5F thumb's pinch
pose lies nearly flat over the table (its middle phalanx reaches 50 mm along
the pad at pad depth), and pointing its tip down into a 27 mm hole leaves
12-30 mm of fingertip IK residual, so it lands on the loop wall.

Measured: the height held once lifted, the blade opening angle (hinge
coordinate) through two open/close cycles and the peak of each, the height
still held after them, and the rest height after release.

Differences from the clips: the person works the loops with the thumb against
the fingers and lifts the scissors flat; here the index and middle work them
against each other, the scissors are stood up on their tips first and hang
blades-down, the blades open about 14 deg against the person's 30, and the
scissors start 0.23 m nearer the arm (from the clip's spot the stand-up runs
out of reach).

Assumed numbers (not measured): 190 mm, 58 g, steel blades friction 0.5,
plastic loops friction 0.7, loop holes 35 x 27 mm with 11 mm tall walls.
Only robot joint targets are written.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robosimstudio.specs import AssetSpec, CameraSpec, StaticSpec, ObjectSpec
from dexagent.scene import table_scene, WORK
from dexagent.workcell import hand_down, TABLE_Z, YAW
from dexagent.hand import Controller, CURL, OPEN, smooth

HERE = Path(__file__).resolve().parent
CLIP = HERE.parents[2] / "reference/hand_demos/IMG_4778_scissors_open_close_a.mov"
HAND = hand_down(YAW)
SPREAD_AXIS = np.array([0.0, 1.0, 0.0])            # palm frame: middle -> index
# Index and middle pads either side of this palm-frame point along SPREAD_AXIS,
# 4-38 mm either side with the distal links within 15 deg of straight down
# (CPU IK search on the fingertip pad sites).
GRIP_CENTRE = np.array([0.100, 0.016, 0.130])
LOCAL_Y = HAND @ SPREAD_AXIS                       # loop A (index) side
LOCAL_X = np.cross(LOCAL_Y, [0, 0, 1.0])           # blades: along the fingers
QUAT = tuple(Rotation.from_matrix(np.column_stack([LOCAL_X, LOCAL_Y, [0, 0, 1.0]])).as_quat())
# Nearer the arm than the clip's scissors: standing them up carries the palm about
# 0.3 m further out and up, past the arm's reach from the clip's position.
PIVOT = WORK + np.array([-0.10, -0.20, 0.0068])
LOOPS = np.array([-0.065, 0.0, 0.0])               # midpoint between the loop holes (scissors frame)
LOOP_HALF = 0.024                                  # loop-hole centres either side of LOOPS, closed
PAD_HEIGHT = 0.0105    # pad centres over the table: the finger ends reach 9.4 mm below the pads
HOLD_HALF = 0.004      # commanded half-separation to squeeze shut: the walls stop the pads at about 20 mm
OPEN_HALF = 0.050      # spread: the middle abduction joint at its 42 deg limit, the index at -39 deg
BLADE_TIP = np.array([0.096, 0.0, 0.0])           # scissors frame
PITCH = np.radians(100)  # stand-up angle: fingers down, the scissors leaning 10 deg past vertical
STAND_S = 3.0          # stand-up time: the loops creep off the middle finger if slower; faster, the palm trails
                       # its path and the fingers lever the blade tips off the table
DEEPER = 0.014         # how much further into the loops the fingers go once stood up
OVERSHOOT = np.radians(10)
RAISE = 0.06           # straight lift once vertical: the blade tips leave the table

scene = table_scene("scissors")
scene.add(StaticSpec(name="scissors_asset", asset=AssetSpec(path=str(HERE / "scissors.xml"))))
scene.add(ObjectSpec(name="scissors", body_name="scissors", joint_name="scissors_free", pose=tuple(PIVOT), quat=QUAT,
                     obb_grasp=False))
# A low camera across the table from the arm, for the fingers in the loops.
scene.add(CameraSpec(name="loops", width=640, height=480, fov_y=50, eye=(0.80, 0.05, TABLE_Z + 0.20),
                     look_at=(0.48, -0.22, TABLE_Z + 0.12)))


def hinge_angle(sim):
    model = sim.model
    if not hasattr(hinge_angle, "index"):
        labels = list(model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("scissors_hinge"))
        hinge_angle.index = int(model.joint_q_start.numpy()[j])
    return float(sim.state.joint_q.numpy()[hinge_angle.index])


def pitched(theta):
    """The palm turned `theta` about its y axis: at 90 deg the fingers point down and the curled tips horizontal."""
    return HAND @ Rotation.from_rotvec([0, theta, 0]).as_matrix()


def grip_q(c, half):
    """Index and middle pads `half` either side of GRIP_CENTRE along SPREAD_AXIS; thumb, ring and little clear."""
    q = OPEN.copy()
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    return c.fingertips({1: GRIP_CENTRE + half * SPREAD_AXIS, 2: GRIP_CENTRE - half * SPREAD_AXIS}, q)


def side_q(into, q):
    """`into` with only the index and middle abduction joints taken from `q`, so a squeeze or spread
    stopped by a loop wall pushes the fingers sideways and leaves their flexion at the loop depth."""
    out = into.copy()
    out[[4, 8]] = q[[4, 8]]
    return out


def cut(sim, session=None):
    c = Controller(sim, session, objects=("scissors",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.5, "settle")
        p, R = c.pose("scissors")
        grip = p + R @ LOOPS
        grip[2] = TABLE_Z + PAD_HEIGHT
        z0 = p[2]
        into = grip_q(c, LOOP_HALF)
        hold, spread = side_q(into, grip_q(c, HOLD_HALF)), side_q(into, grip_q(c, OPEN_HALF))
        yield from c.move(c.palm_for(grip + [0, 0, .07], GRIP_CENTRE, HAND), HAND, into, 2.0, "approach")
        yield from c.move(c.palm_for(grip, GRIP_CENTRE, HAND), HAND, seconds=1.2, phase="into_loops")
        # Squeeze the loops together: the blades close against their stop.
        yield from c.move(hand=hold, seconds=0.6, phase="squeeze")
        yield from c.hold(0.2)
        # Stand the scissors up about their blade tips, which stay on the table:
        # the loops rise on an arc around the tips and the hand pitches with them,
        # so the fingers keep their place in the loops while the table carries
        # most of the weight. At 90 deg the curled fingertips are horizontal
        # through the loops and the loops hang on them.
        tip = p + R @ BLADE_TIP
        tip[2] = TABLE_Z

        def loop_point(theta, deeper):
            """Where the grip centre goes at stand-up angle `theta`, `deeper` m further into the loops."""
            Rk = pitched(theta)
            return tip + Rk @ HAND.T @ (grip - tip) + deeper * (Rk @ [1.0, 0, 0]), Rk

        def stand(theta0, theta1, deeper, seconds, phase):
            """Pitch from theta0 to theta1 along the arc, one smooth profile over the whole sweep."""
            n = max(1, round(seconds * 50))
            for i in range(1, n + 1):
                g, Rk = loop_point(theta0 + (theta1 - theta0) * smooth(i / n), deeper)
                yield from c.move(c.palm_for(g, GRIP_CENTRE, Rk), Rk, seconds=0.02, phase=phase)
        # The fingers start only 9 mm into the loops (their ends would hit the
        # table otherwise), which is too little to hang the scissors on. Stood
        # a little past vertical, with the blade tips still on the table, the
        # scissors lean on the fingers like a stick pinned at its top, and the
        # fingertips tilt up: the squeeze is relaxed and the fingers pushed
        # further into the loops, up to the curl of the distal joints.
        yield from stand(0.0, PITCH, 0.0, STAND_S, "stand_up")
        yield from c.move(hand=into, seconds=0.3, phase="deepen")
        g, Rk = loop_point(PITCH, DEEPER)
        yield from c.move(c.palm_for(g, GRIP_CENTRE, Rk), Rk, seconds=0.8, phase="deepen")
        yield from c.move(hand=hold, seconds=0.4, phase="deepen")
        # Lift the blade tips off the table, pitching further so the loops
        # settle into the curled fingers.
        Rk = pitched(PITCH + OVERSHOOT)
        yield from c.move(c.palm_for(g + [0, 0, RAISE], GRIP_CENTRE, Rk), Rk, seconds=1.2, phase="lift")
        yield from c.hold(0.6, "hang")
        metrics["held_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - z0), 1)
        if metrics["held_height_mm"] < 100:
            outcome = "dropped"
            return
        angles, peaks = [], []
        for cycle in range(2):
            for q, phase in ((spread, "open"), (hold, "close")):
                for _ in c.move(hand=q, seconds=1.5, phase=f"{phase}_{cycle + 1}"):
                    angles.append(hinge_angle(sim))
                    yield
                yield from c.hold(0.4)
                angles.append(hinge_angle(sim))
                if phase == "open":
                    peaks.append(round(float(np.degrees(max(angles[-96:]))), 1))
        metrics.update(blade_angle_deg=[round(float(np.degrees(min(angles))), 1), round(float(np.degrees(max(angles))), 1)],
                       opened_per_cycle_deg=peaks,
                       closed_after_deg=round(float(np.degrees(angles[-1])), 1),
                       held_after_cycles_mm=round(1000 * (c.pose("scissors")[0][2] - z0), 1))
        # Squeeze shut, lower the blade tips to the table and lay the scissors back down about them.
        yield from c.move(hand=hold, seconds=0.4, phase="squeeze_again")
        g, Rk = loop_point(PITCH, DEEPER)
        yield from c.move(c.palm_for(g, GRIP_CENTRE, Rk), Rk, seconds=1.2, phase="lower")
        # Back out to the starting depth, then lay them down about the blade tips.
        yield from c.move(hand=into, seconds=0.4, phase="shallow")
        g, Rk = loop_point(PITCH, 0.0)
        yield from c.move(c.palm_for(g, GRIP_CENTRE, Rk), Rk, seconds=0.8, phase="shallow")
        yield from c.move(hand=hold, seconds=0.4, phase="shallow")
        yield from stand(PITCH, 0.0, 0.0, STAND_S, "lay_down")
        yield from c.move(hand=into, seconds=0.5, phase="release")
        yield from c.move(c.palm_for(grip + [0, 0, .09], GRIP_CENTRE, HAND), HAND, seconds=1.0, phase="retreat")
        yield from c.hold(0.4, "rest")
        metrics["rest_height_mm"] = round(1000 * (c.pose("scissors")[0][2] - TABLE_Z), 1)
        opened = min(peaks) > 10
        closed = np.degrees(angles[-1]) < 3
        held = metrics["held_after_cycles_mm"] > 100
        set_down = metrics["rest_height_mm"] < 15
        if opened and closed and held and set_down:
            outcome = "success"
        else:
            outcome = "_".join(["opened" if opened else "not_opened", "closed" if closed else "not_closed",
                                "held" if held else "dropped", "set_down" if set_down else "not_set_down"])
    finally:
        c.finish("scissors", outcome, metrics, HERE / "results")


routines = {"pick up, open and close": cut}
