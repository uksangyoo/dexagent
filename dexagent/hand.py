"""Scripted control of the single-arm hand workcell on a RoboSimStudio sim.

Only robot joint targets are written: 6 arm position servos and 20 finger
position servos (kp 3, |tau| <= .65 N m). Object poses, velocities and forces
are read for planning and scoring and never written.

Routines are generators that yield once per 20 ms control step, which is the
live page's routine contract.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .workcell import Kinematics, hand_down

ROOT = Path(__file__).resolve().parents[2]
# Thumb-index pinch measured on this hand (palm frame): the widest aperture
# (8-70 mm) opposes the thumb toward the wrist and the index toward the
# fingertips, so the pinch axis runs along the palm's finger direction.
PINCH_CENTRE = np.array([0.080, 0.040, 0.118])
PINCH_AXIS = np.array([0.0, 0.0, -1.0])  # from the centre toward the thumb
CURL = np.array([0, 1.6, 1.3, 1.1])       # a finger tucked out of the way
OPEN = np.array([0.0, -0.3, 0.2, 0.2,  0, .25, .25, .25,  0, .25, .25, .25,  0, .25, .25, .25,  0.2, 0, .25, .25])


def smooth(u):
    return u * u * (3 - 2 * u)


class Controller:
    def __init__(self, sim, session=None, objects=()):
        self.sim = sim
        self.kin = Kinematics()
        self.handle = sim.scene.handles.arms["right"]
        self.arm_idx = list(self.handle.arm_coords)
        self.hand_idx = list(self.handle.gripper_coords)
        self.target = sim.pipeline.control.joint_target_q.numpy().copy()
        q = sim.state.joint_q.numpy()
        self.arm = q[self.arm_idx].copy()
        self.hand = q[self.hand_idx].copy()
        self.integral = np.zeros(6)
        self.xyz, self.R = self.kin.palm_pose(self.arm)
        self.objects = tuple(objects)
        self.trace, self.events = [], []
        self.phase = "start"
        self.wall = time.time()
        self.session = session
        self._suspended = {}
        if session is not None:
            # The page's scalar gripper slider cannot hold a 20-joint pose.
            for name, panel in session.arms.items():
                self._suspended[name] = panel.suspended
                panel.suspended = True

    # ------------------------------------------------------------ reading
    def measured_palm(self):
        return self.kin.palm_pose(self.sim.state.joint_q.numpy()[self.arm_idx])

    def pose(self, name):
        p = np.array(self.sim[name].pose, dtype=float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def in_palm(self, name):
        """Object position and rotation in the measured palm frame."""
        P, Rp = self.measured_palm()
        p, R = self.pose(name)
        return Rp.T @ (p - P), Rp.T @ R

    def tips_world(self):
        q = self.sim.state.joint_q.numpy()
        self.kin.set(arm=q[self.arm_idx], hand=q[self.hand_idx])
        return np.array([self.kin.data.site_xpos[s].copy() for s in self.kin.tip_site])

    def shapes(self, name):
        """Collider shape indices of a scene item: an object's own, plus every
        collider whose label has `name` as a path segment (parts of a static or
        articulated asset, e.g. a lid or a scissors blade)."""
        cache = self.__dict__.setdefault("_shape_cache", {})
        if name not in cache:
            labels = list(self.sim.model.shape_label)
            found = set(self.sim.scene.handles.objects[name].shapes) if name in self.sim.scene.handles.objects else set()
            found |= {i for i, label in enumerate(labels)
                      if not label.endswith("_visual") and (label == name or f"/{name}" in label or label.startswith(name + "/"))}
            cache[name] = sorted(found)
        return cache[name]

    def contact_forces(self, name):
        """Contact-force magnitude [N] between each finger (thumb..little, palm) and a scene item."""
        c = self.sim.contacts
        obj = self.shapes(name) if isinstance(name, str) else list(name)
        return np.array([float(np.linalg.norm(c.pairwise_force(g, obj))) if len(g) else 0. for g in self._finger_shapes()])

    def _finger_shapes(self):
        if not hasattr(self, "_groups"):
            model = self.sim.model
            labels = list(model.shape_label) if hasattr(model, "shape_label") else []
            shapes = list(self.handle.shapes)
            groups = []
            for key in [f"/link_{i}_" for i in range(1, 6)] + ["/palm/palm_"]:
                groups.append([s for s in shapes if s < len(labels) and key in labels[s] and not labels[s].endswith("_visual")])
            self._groups = groups
        return self._groups

    # ------------------------------------------------------------ acting
    def step(self):
        actual = self.sim.state.joint_q.numpy()
        # Joint-space integral action through the bounded arm servos offsets
        # steady gravity sag. No body-level gravity compensation.
        self.integral = np.clip(self.integral + .08 * (self.arm - actual[self.arm_idx]), -.15, .15)
        self.target[self.arm_idx] = self.arm + self.integral
        self.target[self.hand_idx] = self.hand
        self.sim.pipeline.control.joint_target_q.assign(self.target)
        self.sim.run(1).sync()
        self.record()

    def record(self):
        row = {"t": round(float(self.sim.time), 4), "phase": self.phase,
               "palm_cmd": self.xyz.round(5).tolist()}
        if self.objects:
            self.sim.contacts.update(self.sim.step_count)
            for name in self.objects:
                row[name] = np.round(np.array(self.sim[name].pose, dtype=float), 5).tolist()
                row[name + "_force"] = self.contact_forces(name).round(3).tolist()
        self.trace.append(row)

    def hold(self, seconds, phase=None):
        if phase:
            self.phase = phase
        for _ in range(max(1, round(seconds * 50))):
            self.step()
            yield

    def move(self, xyz=None, R=None, hand=None, seconds=1.0, phase=None, profile=smooth):
        """Palm to (xyz, R) and fingers to `hand` (20 joints, or a function of u) together."""
        if phase:
            self.phase = phase
        x0, R0, h0 = self.xyz.copy(), self.R.copy(), self.hand.copy()
        x1 = x0 if xyz is None else np.asarray(xyz, float)
        R1 = R0 if R is None else np.asarray(R, float)
        slerp = Slerp([0, 1], Rotation.from_matrix([R0, R1]))
        n = max(1, round(seconds * 50))
        for k in range(1, n + 1):
            u = profile(k / n)
            self.xyz = (1 - u) * x0 + u * x1
            self.R = slerp(u).as_matrix()
            if xyz is not None or R is not None:
                self.arm = self.kin.solve_arm(self.xyz, self.R, self.arm, tol=0.004)
            if callable(hand):
                self.hand = np.asarray(hand(u), float)
            elif hand is not None:
                self.hand = (1 - u) * h0 + u * np.asarray(hand, float)
            self.step()
            yield

    def go_home(self, xyz, R=None, seconds=1.5):
        yield from self.move(xyz, hand_down() if R is None else R, OPEN, seconds, phase="home")

    # ------------------------------------------------------------ grasps
    def fingertips(self, targets, base=None):
        """20 joint angles placing finger pads at palm-frame `targets` {finger: xyz}."""
        q = (self.hand if base is None else np.asarray(base, float)).copy()
        for i, xyz in targets.items():
            q[4 * i:4 * i + 4], err = self.kin.solve_finger(i, np.asarray(xyz, float), q[4 * i:4 * i + 4])
            if err > .004:
                self.events.append({"t": float(self.sim.time), "warning": f"finger {i} IK residual {err * 1000:.1f} mm"})
        return q

    def pinch(self, half_width, centre=PINCH_CENTRE, axis=PINCH_AXIS, tuck=(2, 3, 4), helpers=(), base=None):
        """Thumb and index pads `half_width` either side of `centre` along `axis` (palm frame).

        `helpers` lists extra fingers (2 = middle) that back the index with a
        lateral offset; `tuck` curls the remaining fingers clear of the table.
        """
        q = (OPEN if base is None else np.asarray(base, float)).copy()
        for i in tuck:
            q[4 * i:4 * i + 4] = CURL if i != 4 else [0.3, CURL[1] - .2, CURL[2], CURL[3]]
        targets = {0: centre + half_width * axis, 1: centre - half_width * axis}
        for n, i in enumerate(helpers):
            targets[i] = centre - half_width * axis + np.array([0, -0.022 * (n + 1), 0])
        return self.fingertips(targets, q)

    def palm_for(self, point_world, centre=PINCH_CENTRE, R=None):
        """Palm position that puts the palm-frame point `centre` at `point_world`."""
        R = self.R if R is None else R
        return np.asarray(point_world, float) - R @ centre

    def palm_frame_target(self, point_world, xyz=None, R=None):
        """A world point expressed in a commanded palm frame."""
        xyz = self.xyz if xyz is None else xyz
        R = self.R if R is None else R
        return R.T @ (np.asarray(point_world) - xyz)

    # ------------------------------------------------------------ bookkeeping
    def finish(self, task, outcome, metrics, directory):
        if self.session is not None:
            for name in self._suspended:
                self.session.arms[name].suspended = True
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        report = {"task": task, "outcome": outcome, "metrics": metrics, "simulated_s": float(self.sim.time),
                  "wall_s": time.time() - self.wall, "events": self.events, "trace": self.trace}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out / f"{task}-{stamp}.json"
        path.write_text(json.dumps(report))
        (out / f"{task}-latest.json").write_text(json.dumps(report))
        print(f"HAND_RESULT {task} {outcome} {json.dumps(metrics)} {path}", flush=True)
        return report
