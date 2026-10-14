"""Single-world Newton adapter for contact controllers, independent of PPO jobs.

Collision inputs and solved-force observations have distinct buffers. Snapshots
refer to the state at the beginning of the last solved physics step; forces,
geometry, and relative velocities therefore share the same timestamp.
"""

import threading
from contextlib import contextmanager
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from dexagent.grammar.finger_roles import (
    FINGER_NUMBERS,
    JOINT_NAMES,
    LINK_NAMES,
    Finger,
    FingerRole,
)
from dexagent.grammar.schema import FingerState, HandState, Twist

from .contact_observer import ContactObserver
from .object_state import origin_velocity, pose, relative_object_state
from .surface_geometry import project_object_surface

ROOT = Path(__file__).resolve().parents[2]


class UR5eTesolloEnv:
    def __init__(
        self,
        scene=ROOT / "runs/assets/scene.xml",
        *,
        device="cuda:0",
        object_body="nut",
        object_name="hex_nut",
        substeps=10,
        contact_stiffness_N_m=None,
        contact_damping_N_s_m=50.0,
    ):
        self.scene = Path(scene)
        self.lock = threading.RLock()
        self.device = device
        self.object_name = object_name
        self.object_body_name = object_body
        if contact_stiffness_N_m is not None and (
            not np.isfinite(contact_stiffness_N_m)
            or contact_stiffness_N_m <= 0
            or not np.isfinite(contact_damping_N_s_m)
            or contact_damping_N_s_m <= 0
        ):
            raise ValueError("Contact stiffness and damping must be finite and positive")
        self.contact_stiffness_N_m = contact_stiffness_N_m
        self.contact_profile = {
            "mode": "authored_mjcf" if contact_stiffness_N_m is None else "force_space",
            "stiffness_N_m": contact_stiffness_N_m,
            "damping_N_s_m": contact_damping_N_s_m if contact_stiffness_N_m else None,
        }
        self.physics_dt = 0.002
        self.substeps = substeps
        self.dt = self.physics_dt * substeps
        self.time_s = 0.0
        self.contact_time_s = 0.0
        self.roles = {f: FingerRole.FREE for f in Finger}
        self.observer = ContactObserver()
        self.contact_events = []
        self.observation_listeners = []
        self._last_modes = {}
        self._event_time = -1.0
        self.kin_model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.kin_data = mujoco.MjData(self.kin_model)
        self.qhome = (
            self.kin_model.key_qpos[0].copy()
            if self.kin_model.nkey
            else self.kin_model.qpos0.copy()
        )
        self.kin_data.qpos[:] = self.qhome
        mujoco.mj_forward(self.kin_model, self.kin_data)
        wp.init()
        newton.use_coord_layout_targets = True
        with wp.ScopedDevice(device):
            robot = newton.ModelBuilder()
            robot.add_mjcf(
                str(self.scene),
                parse_visuals=False,
                collapse_fixed_joints=True,
                enable_self_collisions=True,
            )
            robot.approximate_meshes(method="convex_hull", keep_visual_shapes=False)
            robot.shape_gap[:] = [0.0005] * robot.shape_count
            for a in range(robot.shape_count):
                for b in range(a + 1, robot.shape_count):
                    if robot.shape_body[a] == robot.shape_body[b]:
                        robot.shape_collision_filter_pairs.append((a, b))
            self.model = robot.finalize(device=device)
            if contact_stiffness_N_m is not None:
                from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE

                names = [n.split("/")[-1] for n in self.model.body_label]
                hand_links = {n for links in LINK_NAMES.values() for n in links}
                selected = [
                    i
                    for i, body in enumerate(robot.shape_body)
                    if body >= 0 and (names[body] in hand_links or names[body] == object_body)
                ]
                mode = self.model.mujoco.solref_mode.numpy()
                ke = self.model.shape_material_ke.numpy()
                kd = self.model.shape_material_kd.numpy()
                mode[selected] = SOLREF_MODE_FORCE_SPACE
                ke[selected] = contact_stiffness_N_m
                kd[selected] = contact_damping_N_s_m
                self.model.mujoco.solref_mode.assign(mode)
                self.model.shape_material_ke.assign(ke)
                self.model.shape_material_kd.assign(kd)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=30,
                ls_iterations=10,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=768,
                nconmax=256,
                use_mujoco_contacts=False,
            )
            self.state, self.other = self.model.state(), self.model.state()
            self.sample = self.model.state()
            self.control = self.model.control()
            self.collisions = self.model.contacts()
            self.solved = newton.Contacts(
                max(self.solver.mjw_data.naconmax, self.collisions.rigid_contact_max),
                0,
                device=device,
                requested_attributes={"force"},
            )
        self.body_names = [n.split("/")[-1] for n in self.model.body_label]
        self.object_body = self.body_names.index(object_body)
        self.body_fingers = {
            i: f for i, n in enumerate(self.body_names) for f in Finger if n in LINK_NAMES[f]
        }
        self.shape_body = self.model.shape_body.numpy()
        self.body_com = self.model.body_com.numpy()
        self.qmap = []
        self.vmap = []
        # Named mapping supports revolute and free task objects without assuming
        # that the Newton and MuJoCo generalized-coordinate layouts coincide.
        for i, label in enumerate(self.model.joint_label):
            name = label.split("/")[-1]
            native = mujoco.mj_name2id(self.kin_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if native < 0:
                continue
            nq0, nv0 = (
                int(self.model.joint_q_start.numpy()[i]),
                int(self.model.joint_qd_start.numpy()[i]),
            )
            mq0, mv0 = (
                int(self.kin_model.jnt_qposadr[native]),
                int(self.kin_model.jnt_dofadr[native]),
            )
            typ = int(self.kin_model.jnt_type[native])
            if typ == mujoco.mjtJoint.mjJNT_FREE:
                self.qmap.extend(zip(range(mq0, mq0 + 3), range(nq0, nq0 + 3), strict=True))
                self.qmap.extend(
                    [(mq0 + 3, nq0 + 6), (mq0 + 4, nq0 + 3), (mq0 + 5, nq0 + 4), (mq0 + 6, nq0 + 5)]
                )
                self.vmap.extend(zip(range(mv0, mv0 + 6), range(nv0, nv0 + 6), strict=True))
            elif typ == mujoco.mjtJoint.mjJNT_BALL:
                self.qmap.extend(
                    [(mq0, nq0 + 3), (mq0 + 1, nq0), (mq0 + 2, nq0 + 1), (mq0 + 3, nq0 + 2)]
                )
                self.vmap.extend(zip(range(mv0, mv0 + 3), range(nv0, nv0 + 3), strict=True))
            else:
                self.qmap.append((mq0, nq0))
                self.vmap.append((mv0, nv0))
        self.actuated_joint_ids = [int(j) for j in self.kin_model.actuator_trnid[:, 0]]
        self.actuated_q = np.array([self.kin_model.jnt_qposadr[j] for j in self.actuated_joint_ids])
        self.finger_q = {
            f: np.array([self.kin_model.joint(n).qposadr[0] for n in JOINT_NAMES[f]])
            for f in Finger
        }
        self.finger_v = {
            f: np.array([self.kin_model.joint(n).dofadr[0] for n in JOINT_NAMES[f]]) for f in Finger
        }
        # Palm is welded into wrist_3_link by Newton's fixed-joint collapse.
        self.palm_anchor, self.palm_local = self._anchor("dg5fs_mount")
        self.tip_anchors = {}
        for f, i in FINGER_NUMBERS.items():
            link = LINK_NAMES[f][-1]
            body = self.body_names.index(link)
            site = self.kin_model.site(f"tip_{i}").id
            T = np.eye(4)
            T[:3, :3] = self.kin_data.site_xmat[site].reshape(3, 3)
            T[:3, 3] = self.kin_data.site_xpos[site]
            self.tip_anchors[f] = body, np.linalg.inv(self._native_body_transform(link)) @ T
        oid = self.kin_model.body(object_body).id
        jstart = self.kin_model.body_jntadr[oid]
        self.free_body = bool(
            jstart >= 0 and self.kin_model.jnt_type[jstart] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.externally_supported = not self.free_body
        self.fixture_pose_world = pose(self._native_body_transform(object_body))
        self.last_targets = self.qhome.copy()
        self.reset()

    def _native_body_transform(self, name):
        i = self.kin_model.body(name).id
        T = np.eye(4)
        T[:3, :3] = self.kin_data.xmat[i].reshape(3, 3)
        T[:3, 3] = self.kin_data.xpos[i]
        return T

    def _anchor(self, name):
        original = self._native_body_transform(name)
        i = self.kin_model.body(name).id
        while self.kin_model.body(i).name not in self.body_names:
            i = self.kin_model.body_parentid[i]
            if i == 0:
                raise ValueError(f"No surviving Newton body for {name}")
        anchor = self.body_names.index(self.kin_model.body(i).name)
        return anchor, np.linalg.inv(
            self._native_body_transform(self.body_names[anchor])
        ) @ original

    def native_q(self, state=None):
        state = self.state if state is None else state
        result = self.qhome.copy()
        q = state.joint_q.numpy()
        for m, n in self.qmap:
            result[m] = q[n]
        return result

    def reset(self, qpos=None, *, actuator_targets=None):
        with self.lock, wp.ScopedDevice(self.device):
            q = self.qhome.copy() if qpos is None else np.asarray(qpos).copy()
            if q.shape != self.qhome.shape or not np.isfinite(q).all():
                raise ValueError("Invalid reset state")
            mapped = self.model.joint_q.numpy()
            for m, n in self.qmap:
                mapped[n] = q[m]
            self.state.joint_q.assign(mapped)
            self.state.joint_qd.zero_()
            self.other.joint_q.assign(mapped)
            self.other.joint_qd.zero_()
            self.control.joint_target_q.assign(mapped)
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            newton.eval_fk(self.model, self.other.joint_q, self.other.joint_qd, self.other)
            for name in ("qacc_warmstart", "qacc", "qfrc_applied"):
                getattr(self.solver.mjw_data, name).zero_()
            self.time_s = 0.0
            self.observer.reset()
            self.roles = {f: FingerRole.FREE for f in Finger}
            self.contact_events = []
            self._last_modes = {}
            self._event_time = -1.0
            self.last_targets = (
                q.copy() if actuator_targets is None else np.asarray(actuator_targets).copy()
            )
            self.step_targets(
                q if actuator_targets is None else actuator_targets
            )  # Obtain physically solved contacts, not unpopulated force buffers.

    def step_targets(
        self, qpos, *, allow_wrist_motion=False, constraints=None, require_support=False
    ):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != self.qhome.shape or not np.isfinite(qpos).all():
            raise ValueError("Joint target shape/nonfinite value")
        with self.lock, wp.ScopedDevice(self.device):
            targets = self.control.joint_target_q.numpy()
            hand_q = set(np.concatenate(list(self.finger_q.values())))
            qpos = qpos.copy()
            if not allow_wrist_motion:
                for q in self.actuated_q:
                    if q not in hand_q:
                        qpos[q] = self.last_targets[q]
            for m, n in self.qmap:
                if m in self.actuated_q:
                    targets[n] = qpos[m]
            self.control.joint_target_q.assign(targets)
            self.last_targets = qpos.copy()
            self.last_constraint_violation = None
            for step in range(self.substeps):
                if (
                    constraints is not None
                    or self.observation_listeners
                    or step == self.substeps - 1
                ):
                    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                        wp.copy(getattr(self.sample, name), getattr(self.state, name))
                    self.contact_time_s = self.time_s
                self.state.clear_forces()
                self.model.collide(self.state, self.collisions)
                self.solver.step(
                    self.state, self.other, self.control, self.collisions, self.physics_dt
                )
                self.state, self.other = self.other, self.state
                self.time_s += self.physics_dt
                if constraints is not None or self.observation_listeners:
                    self.solver.update_contacts(self.solved)
                    observed = self.observe()
                    for listener in tuple(self.observation_listeners):
                        listener(observed)
                if constraints is not None:
                    from dexagent.controllers.guards import violation

                    self.last_constraint_violation = violation(
                        observed, constraints, require_support=require_support
                    )
                    if self.last_constraint_violation:
                        break
            # Solved forces and geometry come from the same constraint rows.
            self.solver.update_contacts(self.solved)
            wp.synchronize_device(self.device)
            if not np.isfinite(self.state.joint_q.numpy()).all():
                raise RuntimeError("Nonfinite Newton state")
            if int(self.collisions.rigid_contact_count.numpy()[0]) >= self.solver.mjw_data.naconmax:
                raise RuntimeError("Newton contact buffer overflow")

    def observe(self):
        with self.lock:
            transforms = []
            for q in self.sample.body_q.numpy():
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(q[3:]).as_matrix()
                T[:3, 3] = q[:3]
                transforms.append(T)
            velocities = self.sample.body_qd.numpy()
            palm = transforms[self.palm_anchor] @ self.palm_local
            obj = transforms[self.object_body]
            pv = origin_velocity(
                transforms[self.palm_anchor],
                self.body_com[self.palm_anchor],
                velocities[self.palm_anchor],
            )
            pv += np.cross(
                velocities[self.palm_anchor, 3:], palm[:3, 3] - transforms[self.palm_anchor][:3, 3]
            )
            ov = origin_velocity(obj, self.body_com[self.object_body], velocities[self.object_body])
            c = self.solved
            n = min(int(c.rigid_contact_count.numpy()[0]), c.rigid_contact_max)
            contact_args = {
                name: getattr(c, attr).numpy()[:n]
                for name, attr in {
                    "points0": "rigid_contact_point0",
                    "points1": "rigid_contact_point1",
                    "normals": "rigid_contact_normal",
                    "shapes0": "rigid_contact_shape0",
                    "shapes1": "rigid_contact_shape1",
                    "forces": "force",
                }.items()
            }
            # A free joint does not imply unsupported: table/fixture contacts
            # must be visible to task gates independently of hand contacts.
            external_force = 0.0
            palm_force = 0.0
            for a, b, force, normal in zip(
                contact_args["shapes0"],
                contact_args["shapes1"],
                contact_args["forces"],
                contact_args["normals"],
                strict=True,
            ):
                if a < 0 or b < 0:
                    continue
                ba, bb = int(self.shape_body[a]), int(self.shape_body[b])
                if self.object_body not in (ba, bb):
                    continue
                other = bb if ba == self.object_body else ba
                if other == self.palm_anchor:
                    palm_force += abs(float(np.dot(force[:3], normal)))
                elif other < 0 or (other != self.object_body and other not in self.body_fingers):
                    external_force += abs(float(np.dot(force[:3], normal)))
            contacts, penetration = self.observer.from_arrays(
                **contact_args,
                shape_body=self.shape_body,
                body_transforms=transforms,
                body_velocities=velocities,
                body_com=self.body_com,
                body_fingers=self.body_fingers,
                body_names=self.body_names,
                object_body=self.object_body,
                robot_bodies=set(self.body_fingers)
                | {
                    i
                    for i, n in enumerate(self.body_names)
                    if n
                    in {
                        "shoulder_link",
                        "upper_arm_link",
                        "forearm_link",
                        "wrist_1_link",
                        "wrist_2_link",
                        "wrist_3_link",
                    }
                },
                object_name=self.object_name,
                palm=palm,
                sample_time_s=self.contact_time_s,
                force_source="mujoco_warp_constraint",
            )
            q = self.native_q(self.sample)
            v = np.zeros(self.kin_model.nv)
            nv = self.sample.joint_qd.numpy()
            for m, n in self.vmap:
                v[m] = nv[n]
            fingers = {}
            for f in Finger:
                b, local = self.tip_anchors[f]
                mode, confidence, evidence = self.observer.classify(
                    f, contacts[f], self.contact_time_s
                )
                fingers[f] = FingerState(
                    joint_positions=tuple(q[self.finger_q[f]]),
                    joint_velocities=tuple(v[self.finger_v[f]]),
                    fingertip_pose_palm=pose(np.linalg.inv(palm) @ transforms[b] @ local),
                    contacts=contacts[f],
                    contact_state=mode,
                    classification_confidence=confidence,
                    classification_evidence=evidence,
                )
                if (
                    self.contact_time_s > self._event_time
                    and f in self._last_modes
                    and self._last_modes[f] != mode
                ):
                    self.contact_events.append(
                        {
                            "time_s": self.contact_time_s,
                            "finger": f.value,
                            "from": self._last_modes[f].value,
                            "to": mode.value,
                        }
                    )
                self._last_modes[f] = mode
            self._event_time = self.contact_time_s
            object_state = relative_object_state(
                self.object_name,
                palm,
                obj,
                Twist(linear=tuple(pv), angular=tuple(velocities[self.palm_anchor, 3:])),
                Twist(linear=tuple(ov), angular=tuple(velocities[self.object_body, 3:])),
                externally_supported=self.externally_supported or external_force > 0.02,
                free_body=self.free_body,
            )
            object_state.external_contact_force_N = external_force
            object_state.palm_contact_force_N = palm_force
            return HandState(
                time_s=self.contact_time_s,
                contact_time_s=self.contact_time_s,
                palm_pose_world=pose(palm),
                fingers=fingers,
                finger_roles=dict(self.roles),
                object_state=object_state,
                force_estimates_available=True,
                self_collision_penetration_m=penetration,
            )

    def close(self):
        pass

    def support_certificate(self, *, exclude=(), max_force_per_finger_N=5, friction=None):
        """Simulation-only static support check using authored Coulomb friction."""
        from dexagent.controllers.wrench_feasibility import certify_support

        from .object_state import matrix

        state = self.observe()
        transform = matrix(state.object_state.pose_world)
        if friction is None:
            model = self.kin_model
            bodies = {model.body(self.object_body_name).id}
            bodies.update(model.body(n).id for links in LINK_NAMES.values() for n in links)
            geoms = [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] in bodies
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            ]
            friction = float(np.min(model.geom_friction[geoms, 0]))
        return certify_support(
            {
                f: s.contacts
                for f, s in state.fingers.items()
                if f not in exclude and state.finger_roles[f] == FingerRole.SUPPORT
            },
            center_of_mass_world=transform[:3, 3]
            + transform[:3, :3] @ self.body_com[self.object_body],
            mass_kg=float(self.model.body_mass.numpy()[self.object_body]),
            friction=friction,
            max_force_per_finger_N=max_force_per_finger_N,
        )

    def contact_target(self, finger, object_point=None):
        """Nearest mesh surface point and outward approach normal.

        Queries actual MuJoCo collision geometry; requested object points must
        lie within 3 mm of the target body's surface. A small probe sphere is
        unnecessary: the named primitive object's surface is evaluated directly.
        """
        with self.lock:
            d = mujoco.MjData(self.kin_model)
            d.qpos[:] = self.native_q()
            mujoco.mj_forward(self.kin_model, d)
            body = self.kin_model.body(LINK_NAMES[finger][-1]).id
            obj = self.kin_model.body(self.object_body_name).id
            collision = (self.kin_model.geom_contype != 0) | (self.kin_model.geom_conaffinity != 0)
            fingers = np.flatnonzero((self.kin_model.geom_bodyid == body) & collision)
            objects = np.flatnonzero((self.kin_model.geom_bodyid == obj) & collision)
            closest = None
            for fgeom in fingers:
                for ogeom in objects:
                    line = np.zeros(6)
                    distance = mujoco.mj_geomDistance(
                        self.kin_model, d, int(fgeom), int(ogeom), 1.0, line
                    )
                    if closest is None or distance < closest[0]:
                        closest = (distance, line.copy())
            if closest is None:
                raise ValueError("Finger or object has no collision geometry")
            _, line = closest
            obj_R = d.xmat[obj].reshape(3, 3)
            proposal = line[3:] if object_point is None else d.xpos[obj] + obj_R @ object_point
            point, normal = project_object_surface(
                self.kin_model, d, obj, proposal, approach_from=line[:3]
            )
            if object_point is not None and np.linalg.norm(point - proposal) > 0.003:
                raise ValueError("Requested point is not on a supported object collision surface")
            object_point = obj_R.T @ (point - d.xpos[obj])
            finger_R = d.xmat[body].reshape(3, 3)
            local = finger_R.T @ (line[:3] - d.xpos[body])
            return np.asarray(object_point), normal, local

    @contextmanager
    def observe_each_step(self, listener):
        """Synchronized task measurements after every solved physics step.

        A listener may raise an episode-end exception to stop execution at the
        first terminal measurement, including inside a long grammar action.
        """
        with self.lock:
            self.observation_listeners.append(listener)
        try:
            yield
        finally:
            with self.lock:
                self.observation_listeners.remove(listener)

    @contextmanager
    def transaction(self):
        # GaP may evaluate observation/checkpoint callbacks concurrently.
        with self.lock:
            yield self
