"""Finger-coordination abstraction levels: what a policy may command.

Every level gets the same observations and the same arm interface; only the
finger command differs. A policy is `run(api)`, a generator; every motion
call is a generator too (`yield from api.move(...)`) and may `return` a dict.

  joints      20 finger joint targets (no coordination)
  fingertips  palm-frame pad positions per finger, solved by IK (intra-finger coordination)
  synergies   k whole-hand posture coefficients (PCA of other tasks' expert grasps; inter-finger coordination)
  contacts    object-centric contact primitives with finger roles: place pads on the object,
              squeeze, rotate/translate the held contacts with the object, release, relocate
  layered     all of the above at once (higher layers offered as a prior, not imposed); usage is counted

Frames. World: z up, table top at z = 0.75. Palm frame: +z along the fingers,
+x out of the palm face (fingers flex toward +x), +y toward the thumb.
Fingers: 0 thumb, 1 index, 2 middle, 3 ring, 4 little. `contact_forces`
returns 6 values: the five fingers, then the palm.

Common API (all levels)
  api.t                                   simulated time [s]
  api.objects                             scored object names
  api.object_pose(name)                   (xyz, R) world
  api.object_in_palm(name)                (xyz, R) in the measured palm frame
  api.object_velocity(name)               (linear, angular) world, if available
  api.object_joint(joint)                 coordinate of an object's own joint (hinge angle etc.), see TASK.md
  api.palm_pose()                         commanded (xyz, R) world
  api.measured_palm()                     measured (xyz, R) world
  api.tips()                              (5, 3) measured pad centres, palm frame
  api.tips_world()                        (5, 3) measured pad centres, world
  api.contact_forces(name)                (6,) contact force magnitude [N] per finger + palm
  api.hand_down(yaw=0.7) / api.hand_up(yaw=0.7, tilt=0)   palm rotations
  api.palm_for(point_world, palm_point, R)   palm position putting palm-frame `palm_point` at `point_world`
  api.hold(seconds)                       keep all targets
  api.move(xyz=None, R=None, seconds=1.0, <level finger command>)   palm and fingers together
  api.note(text)                          add a line to the run log
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation
from .hand import Controller, OPEN, CURL, smooth
from .workcell import hand_down, hand_up, YAW

LEVELS = ("joints", "fingertips", "synergies", "contacts", "layered")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _finger_index(f):
    return FINGER_NAMES.index(f) if isinstance(f, str) else int(f)


class _Base:
    level = None

    def __init__(self, sim, objects=()):
        self._c = Controller(sim, None, objects)
        self._sim = sim
        self.objects = tuple(objects)
        self._log = []
        lim = self._c.kin.model.jnt_range
        self._limits = np.array([lim[self._c.kin.model.joint(f"joint_{i}_{j}").id]
                                 for i in range(1, 6) for j in range(1, 5)])

    # ---------------------------------------------------------- observing
    @property
    def t(self):
        return float(self._sim.time)

    def object_pose(self, name):
        ref = self._sim[name]
        try:
            p = np.array(ref.pose, dtype=float)
        except Exception:  # welded (static) items own no body; they sit at their xform
            p = np.asarray(ref.handle.xform, dtype=float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def object_in_palm(self, name):
        P, Rp = self._c.measured_palm()
        p, R = self.object_pose(name)
        return Rp.T @ (p - P), Rp.T @ R

    def object_velocity(self, name):
        try:
            v = np.array(self._sim[name].velocity, dtype=float)
            return v[:3], v[3:]
        except Exception:
            return None

    def object_joint(self, joint):
        """Coordinate [rad or m] of an object's own joint (e.g. "scissors_hinge", "lid_twist")."""
        m = self._sim.model
        labels = list(m.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith(joint))
        return float(self._sim.state.joint_q.numpy()[int(m.joint_q_start.numpy()[j])])

    def palm_pose(self):
        return self._c.xyz.copy(), self._c.R.copy()

    def measured_palm(self):
        return self._c.measured_palm()

    def _q(self):
        return self._sim.state.joint_q.numpy()[self._c.hand_idx].copy()

    def tips(self):
        return self._c.kin.tips_in_palm(self._q())

    def tips_world(self):
        return self._c.tips_world()

    def contact_forces(self, name):
        self._sim.contacts.update(self._sim.step_count)
        return self._c.contact_forces(name)

    hand_down = staticmethod(hand_down)
    hand_up = staticmethod(hand_up)
    DEFAULT_YAW = YAW

    def palm_for(self, point_world, palm_point, R=None):
        return self._c.palm_for(point_world, np.asarray(palm_point, float), R)

    def note(self, text):
        self._log.append(f"{self.t:6.2f} note {text}")

    # ---------------------------------------------------------- acting
    def hold(self, seconds):
        yield from self._c.hold(seconds)

    def _go(self, xyz, R, hand, seconds, what):
        self._log.append(f"{self.t:6.2f} {what}")
        try:
            yield from self._c.move(xyz, R, hand, seconds)
        except ValueError as e:  # arm IK out of reach
            self._log.append(f"{self.t:6.2f} ARM IK FAILED: {e}")
            raise

    def _clip(self, q):
        return np.clip(np.asarray(q, float), self._limits[:, 0], self._limits[:, 1])


class Joints(_Base):
    """api.move(xyz, R, q=None, seconds): q is 20 finger joint targets [rad]
    (finger i owns q[4i:4i+4]: abduction, then three flexion joints, base to tip;
    thumb joint order is the hand's own) or a dict {finger: 4 angles}; joint-space
    interpolation. Also: api.joint_limits (20, 2), api.q() measured, api.q_target(),
    api.OPEN (an open-hand posture)."""
    level = "joints"
    OPEN = OPEN.copy()

    @property
    def joint_limits(self):
        return self._limits.copy()

    def q(self):
        return self._q()

    def q_target(self):
        return self._c.hand.copy()

    def move(self, xyz=None, R=None, q=None, seconds=1.0):
        hand = None
        if q is not None:
            hand = self._c.hand.copy()
            if isinstance(q, dict):
                for f, v in q.items():
                    i = _finger_index(f)
                    hand[4 * i:4 * i + 4] = v
            else:
                hand = np.asarray(q, float)
            hand = self._clip(hand)
        yield from self._go(xyz, R, hand, seconds, f"move palm={xyz is not None} q={'set' if q is not None else '-'} {seconds}s")


class Fingertips(_Base):
    """api.move(xyz, R, tips=None, seconds, straight=False, directions=None):
    tips {finger: palm-frame pad-centre xyz}; fingers not named keep their
    targets. IK per finger (residual > 4 mm is logged and returned). With
    straight=True pads move along straight palm-frame lines (IK every step);
    otherwise joint interpolation between IK endpoints. `directions` optionally
    points a finger's distal link {finger: unit palm-frame vector}.
    Also: api.tip_targets() (5,3) commanded pads; api.reach(finger, xyz) -> IK residual [m];
    api.open(seconds) returns all fingers to an open posture."""
    level = "fingertips"

    def tip_targets(self):
        return self._c.kin.tips_in_palm(self._c.hand)

    def _solve(self, targets, base, directions=None):
        q = base.copy()
        errs = {}
        for f, xyz in targets.items():
            i = _finger_index(f)
            d = None if not directions or f not in directions else np.asarray(directions[f], float)
            q[4 * i:4 * i + 4], err = self._c.kin.solve_finger(i, np.asarray(xyz, float), q[4 * i:4 * i + 4], direction=d)
            errs[FINGER_NAMES[i]] = round(err * 1000, 1)
        return q, errs

    def reach(self, finger, xyz):
        _, errs = self._solve({finger: xyz}, self._c.hand.copy())
        return list(errs.values())[0] / 1000

    def move(self, xyz=None, R=None, tips=None, seconds=1.0, straight=False, directions=None):
        hand, errs = None, {}
        if tips:
            if straight:
                start = self.tip_targets()
                h0 = self._c.hand.copy()
                goal = {f: np.asarray(v, float) for f, v in tips.items()}
                _, errs = self._solve(goal, h0, directions)
                state = {"q": h0}

                def hand(u):
                    tgt = {f: (1 - u) * start[_finger_index(f)] + u * g for f, g in goal.items()}
                    state["q"], _ = self._solve(tgt, state["q"], directions)
                    return state["q"]
            else:
                hand, errs = self._solve(tips, self._c.hand.copy(), directions)
        bad = {f: e for f, e in errs.items() if e > 4}
        yield from self._go(xyz, R, hand, seconds, f"move tips={sorted(tips) if tips else '-'} {seconds}s" + (f" IK_RESIDUAL_MM={bad}" if bad else ""))
        return {"ik_residual_mm": errs}

    def open(self, seconds=0.8):
        yield from self._go(None, None, OPEN.copy(), seconds, "open")


class Synergies(_Base):
    """api.move(xyz, R, s=None, seconds): s is k synergy coefficients; the finger
    targets are mean + basis @ s (clipped to joint limits), joint-space
    interpolated. The basis is a PCA of finger postures from the expert
    routines of the *other* tasks (this task's own data is excluded), so
    coefficients are in units of one standard deviation of that data.
    api.k, api.describe() -> per-synergy text: which fingers flex/spread and
    how the pads move at +-1 sd; api.coeffs() -> current measured posture
    projected onto the synergies; api.open_coeffs (coefficients of an open hand)."""
    level = "synergies"

    def __init__(self, sim, objects=()):
        super().__init__(sim, objects)
        task = os.environ.get("DEXAGENT_TASK", "")
        d = np.load(os.path.join(os.path.dirname(__file__), "assets", "synergies", f"{task}.npz"))
        self.mean, self.basis, self.sd = d["mean"], d["basis"], d["sd"]  # basis (20, k) scaled by sd
        self.k = self.basis.shape[1]
        self.open_coeffs = self._project(OPEN)

    def _project(self, q):
        return np.linalg.lstsq(self.basis, np.asarray(q, float) - self.mean, rcond=None)[0]

    def coeffs(self):
        return self._project(self._q())

    def posture(self, s):
        return self._clip(self.mean + self.basis @ np.asarray(s, float))

    def describe(self):
        lines = []
        t0 = self._c.kin.tips_in_palm(self.mean)
        for j in range(self.k):
            e = np.zeros(self.k); e[j] = 1
            dp = (self._c.kin.tips_in_palm(self.posture(e)) - t0) * 1000
            moves = ", ".join(f"{FINGER_NAMES[i]} {np.round(dp[i]).astype(int).tolist()}mm" for i in range(5)
                              if np.linalg.norm(dp[i]) > 3)
            lines.append(f"s[{j}] +1 sd: pads move (palm-frame dx,dy,dz) {moves or 'little'}")
        return "\n".join(lines)

    def move(self, xyz=None, R=None, s=None, seconds=1.0):
        hand = None if s is None else self.posture(s)
        yield from self._go(xyz, R, hand, seconds, f"move s={None if s is None else np.round(s, 2).tolist()} {seconds}s")


class Contacts(Fingertips):
    """Object-centric contact primitives. Points are given in the OBJECT frame
    (metres, object origin = its pose position) and mapped through the object's
    current measured pose into the palm frame. Fingers not named in a primitive
    keep their targets (they maintain their contacts). All return dicts with
    the contact forces [N] reached and IK residuals.

      api.approach(obj, {finger: point}, standoff=0.01, seconds=1.0, xyz=None, R=None)
          pads to `standoff` outside each point (away from the object origin, or along `normals`)
      api.grasp(obj, {finger: point}, squeeze=0.002, seconds=0.8, normals=None)
          pads onto the points, pressed `squeeze` m into the surface: an opposition grasp of any finger set
      api.squeeze(fingers, obj, delta, seconds=0.5)    press held pads `delta` m further in (negative loosens)
      api.rotate(obj, axis, angle, fingers, seconds=1.0, frame="palm", centre=None)
          move the named fingers' pads rigidly as if fixed to the object turning by `angle`
          rad about `axis` through `centre` (default the object origin); the others hold
      api.translate(obj, delta, fingers, seconds=1.0, frame="palm")   same, translating by `delta` m
      api.release(fingers, obj, clear=0.015, lift=None, seconds=0.5)
          back pads off `clear` m away from the object origin, then optionally by `lift` (palm-frame vector)
      api.relocate(finger, obj, point, clear=0.015, lift=(0.03, 0, 0), squeeze=0.002, seconds=1.2)
          release one finger, carry it over, and place it on a new point while the others hold
      api.tuck(fingers, seconds=0.6)     curl fingers out of the way
      api.open(seconds=0.8)              open hand
      api.carry(xyz, R, seconds)         move the palm, fingers keep targets
      api.pad_offset                     pad centre to finger surface distance used for placement [m]
    """
    level = "contacts"
    pad_offset = 0.008  # pad centre sits about this far inside the fingertip surface

    # The fingertip IK seed matters: start from a half-flexed posture on the right branch.
    def _obj_to_palm(self, obj):
        p, R = self.object_in_palm(obj)
        return p, R

    def _surface_targets(self, obj, points, inset, normals=None):
        p, R = self._obj_to_palm(obj)
        out = {}
        for f, pt in points.items():
            x = p + R @ np.asarray(pt, float)
            if normals and f in normals:
                n = R @ np.asarray(normals[f], float)  # outward normal, object frame -> palm
            else:
                n = x - p
            n = n / (np.linalg.norm(n) + 1e-9)
            out[f] = x + n * (self.pad_offset - inset)
        return out

    def _forces(self, obj):
        try:
            return np.round(self.contact_forces(obj), 2).tolist()
        except Exception:
            return None

    def approach(self, obj, points, standoff=0.01, seconds=1.0, normals=None, xyz=None, R=None):
        tgt = self._surface_targets(obj, points, -standoff, normals)
        r = yield from Fingertips.move(self, xyz, R, tips=tgt, seconds=seconds)
        return dict(r, forces=self._forces(obj))

    def grasp(self, obj, points, squeeze=0.002, seconds=0.8, normals=None):
        tgt = self._surface_targets(obj, points, squeeze, normals)
        r = yield from Fingertips.move(self, tips=tgt, seconds=seconds, straight=True)
        self._held = getattr(self, "_held", {})
        self._held.update({_finger_index(f): (obj, np.asarray(points[f], float)) for f in points})
        return dict(r, forces=self._forces(obj))

    def squeeze(self, fingers, obj, delta, seconds=0.5):
        p, _ = self._obj_to_palm(obj)
        cur = self.tip_targets()
        tgt = {}
        for f in fingers:
            i = _finger_index(f)
            n = cur[i] - p
            tgt[f] = cur[i] - delta * n / (np.linalg.norm(n) + 1e-9)
        r = yield from Fingertips.move(self, tips=tgt, seconds=seconds, straight=True)
        return dict(r, forces=self._forces(obj))

    def _rigid(self, obj, fingers, Rm, d, centre, frame, seconds, what):
        p, Ro = self._obj_to_palm(obj)
        c = p if centre is None else (p + Ro @ np.asarray(centre, float))
        cur = self.tip_targets()
        if frame == "world":
            _, Rp = self.measured_palm()
            Rm = Rp.T @ Rm @ Rp
            d = Rp.T @ d
        goal = {f: c + Rm @ (cur[_finger_index(f)] - c) + d for f in fingers}
        start = {f: cur[_finger_index(f)] for f in fingers}
        rot = Rotation.from_matrix(Rm).as_rotvec()
        state = {"q": self._c.hand.copy()}

        def hand(u):
            Ru = Rotation.from_rotvec(u * rot).as_matrix()
            tgt = {f: c + Ru @ (start[f] - c) + u * d for f in fingers}
            state["q"], _ = self._solve(tgt, state["q"])
            return state["q"]
        _, errs = self._solve(goal, self._c.hand.copy())
        bad = {f: e for f, e in errs.items() if e > 4}
        yield from self._go(None, None, hand, seconds, what + (f" IK_RESIDUAL_MM={bad}" if bad else ""))
        return {"ik_residual_mm": errs, "forces": self._forces(obj)}

    def rotate(self, obj, axis, angle, fingers, seconds=1.0, frame="palm", centre=None):
        a = np.asarray(axis, float)
        Rm = Rotation.from_rotvec(a / np.linalg.norm(a) * angle).as_matrix()
        return (yield from self._rigid(obj, fingers, Rm, np.zeros(3), centre, frame, seconds,
                                       f"rotate {obj} {np.degrees(angle):.0f}deg about {np.round(a, 2).tolist()} with {list(fingers)}"))

    def translate(self, obj, delta, fingers, seconds=1.0, frame="palm"):
        return (yield from self._rigid(obj, fingers, np.eye(3), np.asarray(delta, float), None, frame, seconds,
                                       f"translate {obj} {np.round(np.asarray(delta) * 1000).tolist()}mm with {list(fingers)}"))

    def release(self, fingers, obj, clear=0.015, lift=None, seconds=0.5):
        p, _ = self._obj_to_palm(obj)
        cur = self.tip_targets()
        tgt = {}
        for f in fingers:
            i = _finger_index(f)
            n = cur[i] - p
            tgt[f] = cur[i] + clear * n / (np.linalg.norm(n) + 1e-9)
        yield from Fingertips.move(self, tips=tgt, seconds=seconds * (0.5 if lift is not None else 1), straight=True)
        if lift is not None:
            tgt = {f: v + np.asarray(lift, float) for f, v in tgt.items()}
            yield from Fingertips.move(self, tips=tgt, seconds=seconds * 0.5, straight=True)
        for f in fingers:
            getattr(self, "_held", {}).pop(_finger_index(f), None)
        return {"forces": self._forces(obj)}

    def relocate(self, finger, obj, point, clear=0.015, lift=(0.03, 0, 0), squeeze=0.002, seconds=1.2, normal=None):
        yield from self.release([finger], obj, clear, lift, seconds * 0.3)
        normals = None if normal is None else {finger: normal}
        above = self._surface_targets(obj, {finger: point}, -clear, normals)
        above = {finger: above[finger] + np.asarray(lift, float)}
        yield from Fingertips.move(self, tips=above, seconds=seconds * 0.3)
        yield from self.approach(obj, {finger: point}, standoff=clear, seconds=seconds * 0.2, normals=normals)
        return (yield from self.grasp(obj, {finger: point}, squeeze, seconds * 0.2, normals))

    def tuck(self, fingers, seconds=0.6):
        hand = self._c.hand.copy()
        for f in fingers:
            i = _finger_index(f)
            hand[4 * i:4 * i + 4] = CURL if i != 4 else [0.3, CURL[1] - .2, CURL[2], CURL[3]]
        yield from self._go(None, None, self._clip(hand), seconds, f"tuck {list(fingers)}")

    def carry(self, xyz=None, R=None, seconds=1.0):
        yield from self._go(xyz, R, None, seconds, "carry")

    def move(self, *a, **k):
        raise AttributeError("the contacts level has no direct finger move; use carry() for the palm and the contact primitives for fingers")


def make_api(level, sim, objects=()):
    cls = {"joints": Joints, "fingertips": Fingertips, "synergies": Synergies, "contacts": Contacts}[level]
    return cls(sim, objects)


class Layered(Contacts):
    """Every layer at once: the agentic analogue of EigenDExplore's joint actions
    with a synergy prior. Nothing is taken away; the higher layers are offered,
    not imposed, and api usage is counted per layer.

      api.move_joints(xyz, R, q, seconds)                       joints level `move`
      api.move_tips(xyz, R, tips, seconds, straight, directions) fingertips level `move`
      api.move_synergy(xyz, R, s, seconds)                      synergies level `move`
      api.synergy_posture(s) -> 20 joint targets; api.k, api.describe(), api.coeffs(), api.open_coeffs
      contact primitives: approach, grasp, squeeze, rotate, translate, release, relocate, tuck, open, carry
      api.joint_limits, api.q(), api.q_target(), api.OPEN, api.tip_targets(), api.reach(finger, xyz)
    """
    level = "layered"
    OPEN = OPEN.copy()
    _LAYER = {"move_joints": "joints", "move_tips": "fingertips", "move_synergy": "synergies",
              "approach": "contacts", "grasp": "contacts", "squeeze": "contacts", "rotate": "contacts",
              "translate": "contacts", "release": "contacts", "relocate": "contacts", "tuck": "contacts",
              "open": "contacts", "carry": "arm"}

    def __init__(self, sim, objects=()):
        super().__init__(sim, objects)
        task = os.environ.get("DEXAGENT_TASK", "")
        d = np.load(os.path.join(os.path.dirname(__file__), "assets", "synergies", f"{task}.npz"))
        self.mean, self.basis, self.sd = d["mean"], d["basis"], d["sd"]
        self.k = self.basis.shape[1]
        self.open_coeffs = Synergies._project(self, OPEN)
        self._usage = {}
        self._depth = 0

    def _count(self, name):
        if self._depth == 0:  # count the policy's own calls, not the primitives' internal ones
            u = self._usage.setdefault(name, {"calls": 0, "layer": self._LAYER.get(name, "?")})
            u["calls"] += 1

    def _wrap(name, fn):
        def call(self, *a, **k):
            self._count(name)
            self._depth += 1
            try:
                return (yield from fn(self, *a, **k))
            finally:
                self._depth -= 1
        call.__name__, call.__doc__ = name, fn.__doc__
        return call

    move_joints = _wrap("move_joints", Joints.move)
    move_tips = _wrap("move_tips", Fingertips.move)
    move_synergy = _wrap("move_synergy", Synergies.move)
    approach = _wrap("approach", Contacts.approach)
    grasp = _wrap("grasp", Contacts.grasp)
    squeeze = _wrap("squeeze", Contacts.squeeze)
    rotate = _wrap("rotate", Contacts.rotate)
    translate = _wrap("translate", Contacts.translate)
    release = _wrap("release", Contacts.release)
    relocate = _wrap("relocate", Contacts.relocate)
    tuck = _wrap("tuck", Contacts.tuck)
    open = _wrap("open", Contacts.open)
    carry = _wrap("carry", Contacts.carry)
    del _wrap

    joint_limits = Joints.joint_limits
    q = Joints.q
    q_target = Joints.q_target
    synergy_posture = Synergies.posture
    _project = Synergies._project
    coeffs = Synergies.coeffs
    describe = Synergies.describe
    posture = Synergies.posture


class Force(Layered):
    """The layered level plus a closed-loop grip-force servo (phase 2 of the study:
    the layer that phase-1 agents most often said was missing).

      api.grip(obj, {finger: newtons}, directions=None, seconds=1.0, max_travel=0.015)
          engage a per-step force servo on the named fingers. Each 20 ms step, every
          gripped pad moves along its closing direction by gain * (target - measured
          contact force with `obj`), up to `max_travel` past its commanded target
          (or 6 mm back). The default closing direction points at the centroid of the
          other gripped pads (an opposition pinch, right for thin or long objects);
          a lone gripped finger closes toward the object origin. `directions`
          {finger: unit vector in the OBJECT frame, pointing into the object} overrides.
          The servo keeps running through every later motion (carry, move_tips,
          rotate, hold, ...) until ungrip. grip() itself waits up to `seconds` for the
          forces to settle and returns {"forces": per-finger measured N, "travel_mm": ...}.
      api.ungrip(fingers=None)      stop the servo (all fingers if None); the pads stay where
          the servo had them (the offset is kept as the new target), so nothing jumps
      api.grip_state()              {finger: {"target_N", "force_N", "travel_mm"}}
      release/relocate/grasp on a gripped finger ungrip it first.
    """
    level = "force"
    GAIN = 0.008      # m of pad travel per N of force error per second (time constant ~0.1 s)
    _LAYER = dict(Layered._LAYER, grip="force", ungrip="force")

    def __init__(self, sim, objects=()):
        super().__init__(sim, objects)
        self._grip = {}
        self._offset = np.zeros(20)
        orig = self._c.step

        def step():
            if not self._grip:
                return orig()
            self._servo()
            nominal = self._c.hand.copy()
            self._c.hand = self._clip(nominal + self._offset)
            try:
                orig()
            finally:
                self._c.hand = nominal
        self._c.step = step

    def _dir(self, i, g, tips_n, p_obj, R_obj):
        if g["dir"] is not None:
            n = R_obj @ g["dir"]
        else:
            others = [tips_n[j] + self._travel_vec.get(j, 0) for j in self._grip if j != i]
            n = (np.mean(others, axis=0) if others else p_obj) - tips_n[i]
        return n / (np.linalg.norm(n) + 1e-9)

    def _servo(self):
        nominal = self._c.hand
        tips_n = self._c.kin.tips_in_palm(nominal)
        forces, poses = {}, {}
        self._travel_vec = {}
        for i, g in self._grip.items():
            if g["obj"] not in forces:
                forces[g["obj"]] = self.contact_forces(g["obj"])
                poses[g["obj"]] = self.object_in_palm(g["obj"])
        for i, g in self._grip.items():
            p, R = poses[g["obj"]]
            n = self._dir(i, g, tips_n, p, R)
            f = float(forces[g["obj"]][i])
            g["force"] = f
            g["d"] = float(np.clip(g["d"] + self.GAIN * (g["target"] - f) * 0.02, -0.006, g["max"]))
            self._travel_vec[i] = g["d"] * n
            qn = nominal[4 * i:4 * i + 4]
            seed = qn + self._offset[4 * i:4 * i + 4]
            q, _ = self._c.kin.solve_finger(i, tips_n[i] + g["d"] * n, seed)
            self._offset[4 * i:4 * i + 4] = q - qn
        self._c.kin.set(hand=self._c.hand)

    def _bake(self, fingers):
        for f in fingers:
            i = _finger_index(f)
            if i in self._grip:
                self._c.hand[4 * i:4 * i + 4] = self._clip(self._c.hand + self._offset)[4 * i:4 * i + 4]
                self._offset[4 * i:4 * i + 4] = 0
                self._grip.pop(i)

    def _grip_impl(self, obj, forces, directions=None, seconds=1.0, max_travel=0.015):
        for f, F in forces.items():
            i = _finger_index(f)
            d = None if not directions or f not in directions else np.asarray(directions[f], float)
            prev = self._grip.get(i, {})
            self._grip[i] = {"obj": obj, "target": float(F), "dir": d, "d": prev.get("d", 0.0),
                             "max": float(max_travel), "force": 0.0}
        self._log.append(f"{self.t:6.2f} grip {obj} {dict((FINGER_NAMES[_finger_index(f)], F) for f, F in forces.items())}")
        ok = 0
        for _ in range(max(1, round(seconds * 50))):
            yield from self._c.hold(0.02)
            good = all(abs(g["force"] - g["target"]) <= max(0.3, 0.2 * g["target"]) for g in self._grip.values())
            ok = ok + 1 if good else 0
            if ok >= 5:
                break
        return {"forces": self._forces(obj), "travel_mm": {FINGER_NAMES[i]: round(g["d"] * 1000, 1) for i, g in self._grip.items()}}

    def _ungrip_impl(self, fingers=None):
        fingers = list(self._grip) if fingers is None else fingers
        self._bake(fingers)
        self._log.append(f"{self.t:6.2f} ungrip {[FINGER_NAMES[_finger_index(f)] for f in fingers]}")
        return {}
        yield  # noqa: unreachable; keeps this a generator

    def grip_state(self):
        return {FINGER_NAMES[i]: {"target_N": g["target"], "force_N": round(g["force"], 2),
                                  "travel_mm": round(g["d"] * 1000, 1)} for i, g in self._grip.items()}

    def _release_impl(self, fingers, obj, *a, **k):
        self._bake(fingers)
        return (yield from Contacts.release(self, fingers, obj, *a, **k))

    def _relocate_impl(self, finger, obj, *a, **k):
        self._bake([finger])
        return (yield from Contacts.relocate(self, finger, obj, *a, **k))

    def _grasp_impl(self, obj, points, *a, **k):
        self._bake(list(points))
        return (yield from Contacts.grasp(self, obj, points, *a, **k))

    def _w(name, fn):
        def call(self, *a, **k):
            self._count(name)
            self._depth += 1
            try:
                return (yield from fn(self, *a, **k))
            finally:
                self._depth -= 1
        call.__name__, call.__doc__ = name, fn.__doc__
        return call

    grip = _w("grip", _grip_impl)
    ungrip = _w("ungrip", _ungrip_impl)
    release = _w("release", _release_impl)
    relocate = _w("relocate", _relocate_impl)
    grasp = _w("grasp", _grasp_impl)
    del _w


LEVELS = LEVELS + ("force",)


def make_api(level, sim, objects=()):  # noqa: F811
    cls = {"joints": Joints, "fingertips": Fingertips, "synergies": Synergies, "contacts": Contacts,
           "layered": Layered, "force": Force}[level]
    return cls(sim, objects)
