"""A pen standing in a cup, and the three-finger pick that takes it out.

Shared by pen_finger_roll and pen_writing_grasp. The clips pick the pen off
the table; on this hand that is not reliable (see pen_grasp.py), so both
scenes start from a cup the clips do not show.
"""
import numpy as np
from dexagent.rss.workcell import hand_down, TABLE_Z
from dexagent.rss.hand import PINCH_CENTRE
from dexagent.rss.objects import pen, PEN_RADIUS, PEN_HALF

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
    scene.add(pen(at=tuple(at + [0, 0, PEN_HALF + PEN_RADIUS + 0.001]), direction=(0, 0, 1)))


def pick_from_cup(c, lift=0.10, show=(0, 0, 0)):
    """Generator: pinch the pen 30 mm below its top and lift it out. Returns the held hand pose.

    Thumb and index pinch across the barrel; the middle finger closes the third
    side. Two rounded pads alone squeeze a 7.5 mm barrel out sideways (toward
    palm -y) as soon as the cup stops holding it.
    """
    p0, R0 = c.pose("pen")
    grip = p0 + R0[:, 2] * (PEN_HALF + PEN_RADIUS) - np.array([0, 0, GRIP_BELOW_TOP])
    R = hand_down()
    yield from c.move(c.palm_for(grip + [0, 0, .08], R=R), R, c.pinch(0.026), 1.5, "approach")
    yield from c.move(c.palm_for(grip, R=R), R, seconds=1.0, phase="descend")
    held = c.fingertips({2: PINCH_CENTRE + MIDDLE_HOLD}, c.pinch(PINCH_HALF, tuck=(3, 4)))
    yield from c.move(hand=held, seconds=0.8, phase="pinch")
    yield from c.hold(0.3)
    out = grip + [0, 0, lift]
    yield from c.move(c.palm_for(out, R=R), R, seconds=2.5, phase="lift_out")
    yield from c.move(c.palm_for(out + np.asarray(show), R=R), R, seconds=1.2, phase="present")
    yield from c.hold(0.5, "steady")
    return held
