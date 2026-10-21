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
from dexagent.rss.scene import table_scene
from dexagent.rss.workcell import TABLE_Z
from dexagent.rss.hand import Controller
from palm_up import register_palm_up

HERE = Path(__file__).resolve().parent
EDGE = 0.04
GOAL = np.radians(90)
TURN = np.radians(30)     # rotation commanded per gait
TIP = 0.008               # fingertip pad radius, roughly
RAISED = 0.035            # how far the tips lift off between gaits
PUSH_LIMIT = 8.0          # N, measured fingertip force at which a push stops
CENTRE = np.array([0.046, 0.0, 0.055])  # middle of the palm (palm frame), the cube resting
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


def side_faces(R):
    """Outward normals, in the palm's y-z plane, of the cube faces nearest -y and +y."""
    k = int(np.argmax(np.abs(R[0])))
    normals = []
    for j in range(3):
        if j != k:
            for n in (R[:, j], -R[:, j]):
                n = n - n[0] * np.array([1.0, 0, 0])
                normals.append(n / np.linalg.norm(n))
    return min(normals, key=lambda n: n[1]), max(normals, key=lambda n: n[1])


def rot_x(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


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


DEBUG = False


def dbg(c, tag):
    pos, R = c.in_palm("cube")
    _, flat = yaw_of(R)
    tot = np.degrees(yaw_of(R)[0])
    tips = c.kin.tips_in_palm(c.sim.state.joint_q.numpy()[c.hand_idx])
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


def gait(c, track):
    """One gait: a couple about the palm normal, then back off and lift.

    The thumb reaches across the palm to the cube's -y (little-finger side)
    face and pushes it toward +y near the knuckle end; the index pushes the
    +y face toward -y near the wrist end. Both contact points swing TURN
    about the cube centre, a millimetre inside the faces, while the centre is
    carried back toward the middle of the palm; each push stops early when a
    measured fingertip force passes PUSH_LIMIT. The next gait starts from the
    faces the cube then presents.
    """
    p, R = c.in_palm("cube")
    left, right = side_faces(R)

    def along(n):  # in-face direction toward +z (the knuckles)
        t = np.cross([1.0, 0, 0], n)
        return t if t[2] > 0 else -t
    low = np.array([-0.006, 0, 0])  # below mid-height: push without tipping the cube
    thb = p + low + (EDGE / 2 + TIP) * left + 0.012 * along(left)
    idx = p + low + (EDGE / 2 + TIP) * right - 0.012 * along(right)
    up = np.array([RAISED, 0, 0])
    yield from tip_line(c, {0: thb + up + 0.006 * left, 1: idx + up + 0.006 * right}, 0.5, "gait: above")
    yield from tip_line(c, {0: thb + 0.006 * left, 1: idx + 0.006 * right}, 0.4, "gait: down")
    home = np.array([p[0], CENTRE[1], CENTRE[2]])
    shift = np.clip(home - p, -0.008, 0.008)
    c.phase = "gait: push"
    n = 40
    for k in range(1, n + 1):
        if k % 2 == 0:
            u = k / n
            c.hand = c.fingertips({0: p + u * shift + rot_x(thb - p - 0.001 * left, -u * TURN),
                                   1: p + u * shift + rot_x(idx - p - 0.001 * right, -u * TURN)})
        c.step()
        track.update(c.in_palm("cube")[1])
        yield
        f = c.contact_forces("cube")
        if k > 10 and max(f[0], f[1]) > PUSH_LIMIT:
            break
    # Back off away from where the cube actually is, then lift.
    pc, _ = c.in_palm("cube")
    now = tips_now(c, (0, 1))
    away = {}
    for i, t in now.items():
        d = t - pc
        d[0] = 0
        away[i] = t + 0.005 * d / np.linalg.norm(d) + [0.012, 0, 0]
    yield from tip_line(c, away, 0.3, "gait: release", track)
    yield from tip_line(c, {i: a + up for i, a in away.items()}, 0.4, "gait: lift", track)


def centre(c):
    """Thumb and index close on the side faces symmetrically about the palm's midline."""
    p, R = c.in_palm("cube")
    left, right = side_faces(R)
    low = np.array([-0.006, 0, 0])
    z = np.clip(p[2], 0.045, 0.06)
    up = np.array([RAISED, 0, 0])
    at = np.array([p[0], p[1], z]) + low
    mid = np.array([p[0], CENTRE[1], z]) + low
    out = EDGE / 2 + TIP + 0.006
    yield from tip_line(c, {0: at + out * left + up, 1: at + out * right + up}, 0.5, "centre: above")
    yield from tip_line(c, {0: at + out * left, 1: at + out * right}, 0.4, "centre: down")
    yield from tip_line(c, {0: mid + (EDGE / 2 + TIP - 0.002) * left, 1: mid + (EDGE / 2 + TIP - 0.002) * right}, 0.8, "centre: squeeze")
    yield from tip_line(c, {0: mid + out * left + [0.012, 0, 0], 1: mid + out * right + [0.012, 0, 0]}, 0.3, "centre: release")
    yield from tip_line(c, {0: mid + out * left + up, 1: mid + out * right + up}, 0.4, "centre: lift")


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
        yield from centre(c)
        pos, R = c.in_palm("cube")
        track = Tracker(R)
        gaits = 0
        while gaits < 8:
            total, flat = track.update(c.in_palm("cube")[1])
            if abs(total) >= GOAL - np.radians(5):
                break
            yield from gait(c, track)
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
