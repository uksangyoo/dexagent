"""New task: palm-up in-hand reorientation. The arm starts palm-up, 230 mm above
the table, with the 40 mm cube resting on the flat of the palm. The thumb and
little finger turn it 90 deg about the palm normal by gaiting. Each gait is a
couple: the thumb pushes the thumb-side face at its fingertip end, the little
finger pushes the opposite face at its wrist end, and both contacts swing TURN
about the cube centre (see `gait`). Then they back off, lift and reset on the
faces the cube now presents. The palm carries the cube throughout.

Why these fingers: on this hand the only collision-free contacts on a cube
resting mid-palm are the thumb on the thumb-side and fingertip-side faces,
middle/ring/little on the fingertip-side face, and the little finger on the
little-side face (CPU clearance check of each finger's links against the
cube). No thumb-index pair opposes across the cube except at one angle, so
the thumb-index pinch-and-turn jams.

Cube: 40 mm, 12 g, friction 0.9 (nominal, not measured; Newton takes the
larger of the two friction coefficients, so the hand's 0.9 applies anyway).
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import robosimstudio as rss
from robosimstudio.specs import CameraSpec
from dexagent.scene import table_scene
from dexagent.workcell import TABLE_Z
from dexagent.hand import Controller
from palm_up import register_palm_up

HERE = Path(__file__).resolve().parent
EDGE = 0.04
GOAL = np.radians(90)
TURN = np.radians(20)     # contact swing per gait
PRESS = 0.002             # contacts commanded this far inside the faces during the swing
OUT = 0.007               # pre-contact standoff
RELEASE = 0.022           # how far the tips back off along the cube-to-tip direction before lifting
RAISED = 0.05             # how far the tips lift between gaits (palm x)
# Contacts, found with a CPU clearance check of every finger link against the
# cube: (palm-frame height offset from the cube centre, pad-site distance from
# the centre along the face normal, offset along the face toward the fingertips,
# a joint seed on the right IK branch).
THUMB = dict(dx=-0.011, gap=0.028, along=0.014, seed=np.array([0.16, -0.13, 1.14, 1.55]))
LITTLE = dict(dx=0.004, gap=0.034, along=-0.007, seed=np.array([0.0, 0.6, 1.57, 1.57]))
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
    a = [R[:, j] for j in range(3) if j != k][0]
    return np.arctan2(a[1], a[2]) % (np.pi / 2), flat


class Tracker:
    """Unwrapped yaw: the cube is 4-fold symmetric about the palm normal."""
    def __init__(self, R):
        self.last, _ = yaw_of(R)
        self.total = 0.0

    def update(self, R):
        y, flat = yaw_of(R)
        self.total += (y - self.last + np.pi / 4) % (np.pi / 2) - np.pi / 4
        self.last = y
        return self.total, flat


def face(R, toward):
    """Outward normal (palm y-z plane) of the cube side face nearest the direction `toward`."""
    k = int(np.argmax(np.abs(R[0])))
    normals = []
    for j in range(3):
        if j != k:
            for n in (R[:, j], -R[:, j]):
                n = np.array([0.0, n[1], n[2]])
                normals.append(n / np.linalg.norm(n))
    return max(normals, key=lambda n: n @ toward)


def along(n):
    """In-face direction of a side face, toward the fingertips (+z)."""
    t = np.cross([1.0, 0, 0], n)
    return t if t[2] > 0 else -t


def rot_x(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


DEBUG = False


def dbg(c, tag):
    pos, R = c.in_palm("cube")
    print(f"   {tag}: yaw(mod90) {np.degrees(yaw_of(R)[0]):.1f} flat {yaw_of(R)[1]:.3f} pos {np.round(pos * 1000, 1)} "
          f"F {np.round(c.contact_forces('cube'), 2)}", flush=True)


def tip_line(c, goal, seconds, phase, track=None):
    """Fingertips along straight lines (palm frame) to `goal` {finger: xyz}, IK every other step."""
    c.phase = phase
    tips = c.kin.tips_in_palm(c.hand)
    start = {i: tips[i].copy() for i in goal}
    n = max(2, round(seconds * 50))
    for k in range(1, n + 1):
        if k % 2 == 0 or k == n:
            u = k / n
            c.hand = c.fingertips({i: (1 - u) * start[i] + u * np.asarray(goal[i]) for i in goal})
        c.step()
        if track is not None:
            track.update(c.in_palm("cube")[1])
        if DEBUG and phase in ("gait: release", "gait: lift") and k % 4 == 0:
            dbg(c, f"{phase} {k}")
        yield
    if DEBUG:
        dbg(c, phase)


def contact(p, n, spec, extra):
    """Pad-site target for a finger contact on the face with outward normal n."""
    q = p + (spec["gap"] + extra) * n + spec["along"] * along(n)
    q[0] = p[0] + spec["dx"]
    return q


def gait(c, track, sense, turn=TURN):
    """One gait: thumb and little finger push opposite faces at opposite ends and swing `turn`."""
    p, R = c.in_palm("cube")
    # Thumb-side face; between two candidates (+-45 deg) take the one the swing starts from.
    nt = face(R, rot_x(np.array([0, 1.0, 0]), -sense * np.radians(15)))
    nl = -nt                                # little-side face
    up = np.array([RAISED, 0, 0])
    pre = {0: contact(p, nt, THUMB, OUT), 4: contact(p, nl, LITTLE, OUT)}
    # Preshape in joint space on the IK branches found offline, tips high over the contacts.
    shape = c.hand.copy()
    for i, spec in ((0, THUMB), (4, LITTLE)):
        shape[4 * i:4 * i + 4], _ = c.kin.solve_finger(i, pre[i] + up, spec["seed"])
    yield from c.move(hand=shape, seconds=0.6, phase="gait: preshape")
    yield from tip_line(c, pre, 0.5, "gait: down", track)
    press = {0: contact(p, nt, THUMB, -PRESS), 4: contact(p, nl, LITTLE, -PRESS)}
    yield from tip_line(c, press, 0.3, "gait: touch", track)
    c.phase = "gait: swing"
    n = 50
    for k in range(1, n + 1):
        if k % 2 == 0:
            a = sense * turn * k / n
            c.hand = c.fingertips({i: p + rot_x(press[i] - p, a) for i in press})
        c.step()
        track.update(c.in_palm("cube")[1])
        yield
    if DEBUG:
        dbg(c, "gait: swing")
    # Back off along the faces' measured normals, then lift.
    pc, Rc = c.in_palm("cube")
    nt = face(Rc, nt)
    tips = c.kin.tips_in_palm(c.hand)
    away = {0: tips[0] + RELEASE * nt, 4: tips[4] - RELEASE * nt}
    yield from tip_line(c, away, 0.3, "gait: release", track)
    yield from tip_line(c, {i: a + up for i, a in away.items()}, 0.4, "gait: lift", track)


def resting_on_palm(c):
    """The palm carries the cube and the fingers do not grip it."""
    f = c.contact_forces("cube")
    pos = c.in_palm("cube")[0]
    return bool(f[5] > 0.02 and f[:5].sum() < 0.5 and pos[0] < 0.06 and abs(pos[1]) < 0.045)


def reorient(sim, session=None, sense=1.0, max_gaits=8):
    c = Controller(sim, session, objects=("cube",))
    outcome, metrics = "incomplete", {}
    try:
        yield from c.hold(0.6, "settle")
        if not resting_on_palm(c):
            outcome = "not_on_palm"
            return
        track = Tracker(c.in_palm("cube")[1])
        gaits = 0
        while gaits < max_gaits and abs(track.total) < GOAL - np.radians(5):
            yield from gait(c, track, sense, min(TURN, GOAL - abs(track.total)))
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
                       on_palm=bool(on_palm), flat=round(float(flat), 3),
                       cube_in_palm_mm=np.round(c.in_palm("cube")[0] * 1000, 1).tolist())
        if outcome == "incomplete":
            outcome = "success" if on_palm and abs(abs(total) - GOAL) < np.radians(10) else "short_of_goal"
    finally:
        c.finish("cube_reorient", outcome, metrics, HERE / "results")


routines = {"palm-up 90 deg reorientation": reorient}
