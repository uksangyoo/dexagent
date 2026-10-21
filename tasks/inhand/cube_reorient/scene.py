"""New task: palm-up in-hand reorientation. The arm starts palm-up, 230 mm above
the table, with the 40 mm cube resting in the middle of the palm. The fingers
turn it 90 deg about the palm normal by gaiting: each gait the thumb (reaching
across the palm) and the index push opposite side faces at opposite ends, a
couple about the palm normal (see `gait`), then back off, lift, and reset on
the faces the cube now presents. Gravity keeps the cube on the palm between gaits.

Cube: 40 mm, 12 g, friction 0.9 (nominal, not measured).
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import robosimstudio as rss
from robosimstudio.specs import CameraSpec
from dexagent.scene import table_scene
from dexagent.workcell import TABLE_Z
from dexagent.hand import Controller, smooth
from palm_up import register_palm_up

HERE = Path(__file__).resolve().parent
EDGE = 0.04
GOAL = np.radians(90)
TIP = 0.008               # fingertip pad radius, roughly
RAISED = 0.05             # how far the tips lift off between gaits
THUMB_SEED = np.array([1.15, -0.92, 1.29, 1.19])  # thumb opposing across the palm
CLEAR = 0.10              # palm-x height a fingertip rises to before it crosses over the cube
SQUEEZE = 0.001           # pads commanded this far inside the faces: about 1 N of pinch
LOW = -0.004              # pinch a little below the cube's mid-height (palm x)
# Pinch axis range (atan2(y, z) of the index contact, palm frame) that thumb and
# index both reach around a cube centred at TURN_AT: START at the start of a gait,
# up to about START + 95 deg at the end.
START = np.radians(90)
TURN_AT = np.array([0.041, 0.0, 0.065])
CENTRE = np.array([0.041, 0.0, 0.065])  # cube centre resting on the flat of the palm (palm face at x = .020 for z = 45-85 mm)
PALM = np.array([0.48, -0.06, TABLE_Z + 0.23])
TILT = np.radians(5)

ROBOT, R_PALM = register_palm_up(PALM, TILT)
scene = table_scene("cube_reorient", robot=ROBOT)
CUBE_AT = PALM + R_PALM @ (CENTRE + [0.0005, 0, 0])
# Looking down into the upturned palm, fingers pointing up the frame.
scene.add(CameraSpec(name="palm", width=1280, height=720, fov_y=38,
                     eye=(0.40, -0.16, TABLE_Z + 0.62), look_at=(0.52, -0.03, TABLE_Z + 0.26)))
scene.add(rss.box(name="cube", half=(EDGE / 2,) * 3, at=tuple(CUBE_AT), quat=tuple(Rotation.from_matrix(R_PALM).as_quat()),
                  mass=0.012, friction=0.9, color=(0.42, 0.28, 0.72)))


def yaw_of(R):
    """Cube yaw about the palm normal (palm x), modulo 90 deg, and how flat it lies."""
    flat = np.abs(R[0]).max()           # some cube axis along the palm normal
    k = int(np.argmax(np.abs(R[0])))
    cols = [R[:, j] for j in range(3) if j != k]
    a = cols[0]
    return np.arctan2(a[1], a[2]) % (np.pi / 2), flat


class Tracker:
    """Unwrapped yaw: the cube is 4-fold symmetric about the palm normal."""
    def __init__(self, R):
        self.last, _ = yaw_of(R)
        self.total = 0.0

    def update(self, R):
        y, flat = yaw_of(R)
        d = (y - self.last + np.pi / 4) % (np.pi / 2) - np.pi / 4
        self.total += d
        self.last = y
        return self.total, flat


DEBUG = True


def dbg(c, tag):
    pos, R = c.in_palm("cube")
    _, flat = yaw_of(R)
    tot = np.degrees(yaw_of(R)[0])
    tips = c.kin.tips_in_palm(c.sim.state.joint_q.numpy()[c.hand_idx])
    q = c.sim.state.joint_q.numpy()[c.hand_idx]
    print(f"      thumb q {np.round(q[:4], 2)} cmd {np.round(c.hand[:4], 2)}  index q {np.round(q[4:8], 2)} cmd {np.round(c.hand[4:8], 2)}")
    print(f"   {tag}: yaw(mod90) {tot:.1f} flat {flat:.3f} pos {np.round(pos*1000,1)} F {np.round(c.contact_forces('cube'),2)} thumb {np.round(tips[0]*1000)} index {np.round(tips[1]*1000)}", flush=True)


def tips_now(c, fingers):
    tips = c.kin.tips_in_palm(c.hand)
    return {i: tips[i].copy() for i in fingers}


def tip_line(c, goal, seconds, phase, track=None):
    """Fingertips along straight lines (palm frame) to `goal` {finger: xyz}, IK every other step.

    Interpolating joint angles instead swings a finger's links through the cube.
    """
    c.phase = phase
    start = tips_now(c, goal)
    n = max(2, round(seconds * 50))
    for k in range(1, n + 1):
        if k % 2 == 0 or k == n:
            u = k / n
            c.hand = c.fingertips({i: (1 - u) * start[i] + u * np.asarray(goal[i]) for i in goal})
        c.step()
        if track is not None:
            track.update(c.in_palm("cube")[1])
        yield
    if DEBUG:
        dbg(c, phase)


def face_angle(R, near):
    """Angle in the palm's y-z plane (atan2(y, z)) of the cube side-face normal nearest `near`."""
    k = int(np.argmax(np.abs(R[0])))
    angles = [np.arctan2(s * R[1, j], s * R[2, j]) for j in range(3) if j != k for s in (1.0, -1.0)]
    return min(angles, key=lambda a: abs((a - near + np.pi) % (2 * np.pi) - np.pi))


def contacts(p, phi, gap):
    """Thumb and index pad centres either side of `p` along the palm-plane direction `phi`."""
    n = np.array([0.0, np.sin(phi), np.cos(phi)])
    return p - gap * n, p + gap * n


def pinch_turn(c, track, turn, seconds=2.4):
    """One gait: pinch two opposite side faces, swing the pinch `turn` about the cube centre, release.

    The index takes the face whose normal is nearest START (the thumb the
    opposite one); the pinch axis then sweeps toward END, the range both
    fingers reach around a cube at TURN_AT. The cube stays on the palm and
    is carried round by friction at the two pads; its centre slides to
    TURN_AT during the swing.
    """
    p, R = c.in_palm("cube")
    phi = face_angle(R, START)
    low = np.array([LOW, 0, 0])
    gap = EDGE / 2 + TIP
    up = np.array([RAISED, 0, 0])
    t, i = contacts(p + low, phi, gap + 0.010)
    # Preshape in joint space with the fingertips high over the start contacts.
    # The thumb is seeded on the opposition branch that tracks the whole swing
    # smoothly; the other branch (joint_1_2 near its -2.67 limit) jams.
    pre = c.hand.copy()
    pre[:4], _ = c.kin.solve_finger(0, np.r_[CLEAR, t[1:]], THUMB_SEED)
    pre[4:8], _ = c.kin.solve_finger(1, np.r_[CLEAR, i[1:]], c.hand[4:8])
    yield from c.move(hand=pre, seconds=0.8, phase="gait: preshape")
    yield from tip_line(c, {0: t + up, 1: i + up}, 0.4, "gait: above", track)
    yield from tip_line(c, {0: t, 1: i}, 0.5, "gait: down", track)
    t, i = contacts(p + low, phi, gap - SQUEEZE)
    yield from tip_line(c, {0: t, 1: i}, 0.5, "gait: pinch", track)
    c.phase = "gait: turn"
    c0, c1 = p + low, np.array([p[0], TURN_AT[1], TURN_AT[2]]) + low
    n = round(seconds * 50)
    for k in range(1, n + 1):
        u = smooth(k / n)
        t, i = contacts((1 - u) * c0 + u * c1, phi + u * turn, gap - SQUEEZE)
        c.hand = c.fingertips({0: t, 1: i})
        c.step()
        track.update(c.in_palm("cube")[1])
        if DEBUG and k % 15 == 0:
            dbg(c, f"turn {np.degrees(u * turn):.0f} cmd thumb {np.round(t * 1000)} index {np.round(i * 1000)}")
        yield
    # Open along the pinch axis about where the cube actually is, then lift clear.
    pc, _ = c.in_palm("cube")
    t, i = contacts(pc + low, phi + turn, gap + 0.008)
    yield from tip_line(c, {0: t + [0.010, 0, 0], 1: i + [0.010, 0, 0]}, 0.4, "gait: release", track)
    yield from tip_line(c, {0: t + up, 1: i + up}, 0.5, "gait: lift", track)


def resting_on_palm(c):
    """The palm carries the cube and the fingers do not grip it."""
    f = c.contact_forces("cube")
    pos = c.in_palm("cube")[0]
    return bool(f[5] > 0.02 and f[:5].sum() < 0.5 and pos[0] < 0.06 and abs(pos[1]) < 0.045)


def reorient(sim, session=None):
    c = Controller(sim, session, objects=("cube",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.6, "settle")
        pos, R = c.in_palm("cube")
        if not resting_on_palm(c):
            outcome = "not_on_palm"
            return
        pos, R = c.in_palm("cube")
        track = Tracker(R)
        gaits = 0
        while gaits < 3:
            total, flat = track.update(c.in_palm("cube")[1])
            left = GOAL - total
            if abs(left) < np.radians(5):
                break
            yield from pinch_turn(c, track, float(np.clip(left, -np.radians(95), np.radians(95))))
            gaits += 1
            yield from c.hold(0.3, "gait: settle")
            total, flat = track.update(c.in_palm("cube")[1])
            pos = c.in_palm("cube")[0]
            print(f"[gait {gaits}] yaw {np.degrees(total):.1f} deg, flat {flat:.3f}, cube in palm {np.round(pos * 1000, 1)} mm", flush=True)
            if flat < 0.95 or pos[0] > 0.08 or abs(pos[1]) > 0.06:
                outcome = "dropped_or_tumbled"
                break
        yield from c.hold(0.6, "hold")
        total, flat = track.update(c.in_palm("cube")[1])
        on_palm = resting_on_palm(c)
        metrics.update(gaits=gaits, yaw_deg=round(float(np.degrees(total)), 1),
                       yaw_error_deg=round(float(np.degrees(abs(total) - GOAL)), 1),
                       on_palm=bool(on_palm), flat=round(float(flat), 3))
        if outcome == "incomplete":
            outcome = "success" if on_palm and abs(abs(total) - GOAL) < np.radians(10) else "short_of_goal"
    finally:
        c.finish("cube_reorient", outcome, metrics, HERE / "results")


routines = {"palm-up 90 deg reorientation": reorient}
