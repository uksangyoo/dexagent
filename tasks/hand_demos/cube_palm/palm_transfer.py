"""Fingertips-to-palm transfer of the 40 mm cube on the upturned hand, and back.

Shared by tasks/hand_demos/cube_palm and tasks/inhand/cube_reorient. Only
robot joint targets are written (through `dexagent.rss.hand.Controller`).
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.rss.workcell import hand_down, TABLE_Z

EDGE = 0.04
FLIP = np.diag([-1.0, -1.0, 1.0])  # 180 deg about the finger axis: palm down <-> palm up
LIFT = 0.32                         # the hand turns over this far above the table
TILT = np.radians(15)               # palm-up, fingers raised: the palm slopes toward the wrist

# Grip (palm frame): thumb on the wrist-side face, middle and ring side by side on the far face.
PICK = np.array([0.08, -0.004, 0.114])
AXIS = np.array([0.0, 0.0, -1.0])
SQUEEZE = 0.006                     # pads commanded 14 mm inside the 20 mm half-width
FIST = np.array([0, 2.0, 1.5, 1.2])
CRADLE = np.array([0, 0.35, 0.3, 0.2])
LITTLE = np.zeros(4)                # little finger straight throughout
THUMB_WALL = np.array([0.07, 0.065, 0.06])  # thumb pad beside the palm, a wall on its side


def Ry(a):
    return Rotation.from_rotvec([0, a, 0]).as_matrix()


def orbit(c, point, R1, seconds, centre, phase, hand=None, lead=+1):
    """Turn the palm to R1 about a palm-frame point `centre` held at world `point`.

    `hand` is a 20-joint target reached linearly, or a function of progress
    re-solved every fifth step. `lead` picks the way round for a half turn.
    """
    c.phase = phase
    R0, h0 = c.R.copy(), c.hand.copy()
    rel = Rotation.from_matrix(R0.T @ R1).as_rotvec()
    if np.linalg.norm(rel) > np.pi - 1e-3:
        rel = lead * np.array([0, 0, np.pi])
    n = max(1, round(seconds * 50))
    for k in range(1, n + 1):
        u = k / n
        u = u * u * (3 - 2 * u)
        c.R = R0 @ Rotation.from_rotvec(u * rel).as_matrix()
        c.xyz = np.asarray(point) - c.R @ centre
        c.arm = c.kin.solve_arm(c.xyz, c.R, c.arm, tol=0.004)
        if callable(hand):
            if k % 5 == 0 or k == n:
                c.hand = hand(u)
        elif hand is not None:
            c.hand = (1 - u) * h0 + u * np.asarray(hand, float)
        c.step()
        yield


def grip(c, centre, h):
    """Thumb on the near (wrist-side) face; middle and ring side by side on the far face.

    Three contacts, two of them 22 mm apart on one face, so the cube cannot spin
    out about a two-point pinch axis while the hand turns over (the hand's
    contacts have no torsional friction: MuJoCo condim 3). Index and little lie
    straight: palm-down they are above the cube and clear of the table,
    palm-up they are the flat hand the cube lands on.
    """
    q = c.hand.copy()  # seeds the finger IK from the current pose
    q[4:8] = 0
    q[16:20] = LITTLE
    far = centre - h * AXIS
    return c.fingertips({0: centre + h * AXIS, 2: far + [0, 0.011, 0], 3: far + [0, -0.011, 0]}, q)


def note(c, tag):
    pos, _ = c.in_palm("cube")
    f = c.contact_forces("cube")
    c.events.append({"t": round(float(c.sim.time), 2), "phase": tag, "cube_in_palm_mm": (pos * 1000).round(1).tolist(),
                     "force_N": f.round(2).tolist()})
    print(f"[{tag}] t={c.sim.time:.2f} cube in palm {np.round(pos * 1000, 1)} mm, "
          f"force thumb..little,palm {np.round(f, 2)} N", flush=True)


def Rz(a):
    return Rotation.from_rotvec([0, 0, a]).as_matrix()


def to_palm(c, p0, roll=0.0):
    """Pick from the table, lift, turn palm-up, and let the cube into the palm.

    `roll` lowers the thumb side of the palm by that angle while the cube is
    let go, which moves where it lands toward the thumb side.
    """
    R = hand_down()
    yield from c.move(c.palm_for(p0 + [0, 0, .08], PICK, R=R), R, grip(c, PICK, 0.032), 1.5, "approach")
    yield from c.move(c.palm_for(p0, PICK, R=R), R, seconds=1.0, phase="descend")
    yield from c.move(hand=grip(c, PICK, SQUEEZE), seconds=0.8, phase="pinch")
    yield from c.hold(0.3)
    held = p0 + [0, 0, LIFT]
    yield from c.move(c.palm_for(held, PICK, R=R), R, seconds=1.2, phase="lift")
    note(c, "lifted")
    Rup = R @ FLIP
    # Turning thumb-side down keeps gravity's pull toward the two far-face fingers.
    yield from orbit(c, held, Rup, 2.0, PICK, "turn palm-up", lambda u: grip(c, PICK, SQUEEZE), lead=-1)
    yield from orbit(c, held, Rup @ Ry(TILT) @ Rz(roll), 1.0, PICK, "tilt")
    yield from c.hold(0.3)
    note(c, "palm-up")
    # Thumb straight back off the near face, up, and down beside the palm as a wall.
    thumb = c.kin.tips_in_palm(c.hand)[0]
    back = c.fingertips({0: thumb + [0.0, 0.0, -0.015]})
    yield from c.move(hand=back, seconds=0.8, phase="release")
    up = c.fingertips({0: thumb + [0.04, 0.0, -0.025]}, back)
    yield from c.move(hand=up, seconds=0.6)
    wall = c.fingertips({0: THUMB_WALL}, up)
    yield from c.move(hand=wall, seconds=0.5)
    note(c, "released")
    # Curl past a right angle: the knuckles roll the cube over into the palm.
    fist = wall.copy()
    fist[4:16] = np.tile(FIST, 3)
    c.phase, h0 = "curl", wall.copy()
    for k in range(1, 101):
        c.hand = h0 + (k / 100) * (fist - h0)
        c.step()
        yield
        pos = c.in_palm("cube")[0]
        if pos[2] < 0.06 and pos[0] < 0.06:
            break
    note(c, "curled")


def return_to_table(c, p0):
    """Close the hand over the cube, turn palm-down, and set it on the table."""
    fist = c.hand.copy()
    fist[4:20] = np.tile(FIST, 4)
    fist[16:20] = [0.5, 1.4, 1.3, 1.1]
    yield from c.move(hand=fist, seconds=1.0, phase="close")
    note(c, "closed")
    p, _ = c.pose("cube")
    centre = c.in_palm("cube")[0]
    yield from orbit(c, p, hand_down(), 2.0, centre, "turn palm-down")
    note(c, "palm-down")
    # Lower until the cube's bottom is 15 mm off the table, then open.
    target = p0 + [0, 0.05, 0.015]
    p, _ = c.pose("cube")
    yield from c.move(c.xyz + (target - p), seconds=1.5, phase="lower")
    open_ = c.hand.copy()
    open_[4:20] = np.tile(CRADLE, 4)
    open_[0:4] = [0.3, -0.6, 0.2, 0.2]
    yield from c.move(hand=open_, seconds=0.8, phase="release")
    yield from c.move(c.xyz + [0, 0, 0.12], seconds=1.0, phase="retreat")
    yield from c.hold(0.5, "rest")


def open_palm(c, seconds=1.2):
    """Uncurl to a cradle: the cube lies on the open palm."""
    cradle = c.hand.copy()
    cradle[4:16] = np.tile(CRADLE, 3)
    cradle[16:20] = LITTLE
    yield from c.move(hand=cradle, seconds=seconds, phase="open")


def resting_on_palm(c):
    """The palm carries the cube and the fingers do not grip it."""
    f = c.contact_forces("cube")
    pos = c.in_palm("cube")[0]
    return bool(f[5] > 0.02 and f[:5].sum() < 0.5 and pos[0] < 0.06 and abs(pos[1]) < 0.045)
