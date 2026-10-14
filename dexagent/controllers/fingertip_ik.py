"""Bounded hand/wrist IK with material-point targets and collision rejection."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize

from dexagent.grammar.finger_roles import JOINT_NAMES, Finger


@dataclass(frozen=True)
class PointTarget:
    body: str
    local_point: tuple
    world_point: tuple
    weight: float = 1.0


@dataclass
class IKResult:
    feasible: bool
    qpos: np.ndarray
    max_error_m: float
    min_distance_m: float
    reason: str


class FingertipIK:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.hand_q = np.array([model.joint(n).qposadr[0] for f in Finger for n in JOINT_NAMES[f]])
        self.actuated = np.array([model.jnt_qposadr[j] for j in model.actuator_trnid[:, 0]])
        self.q_to_joint = {int(model.jnt_qposadr[j]): j for j in model.actuator_trnid[:, 0]}
        self.robot_bodies = {
            i
            for i in range(model.nbody)
            if model.body(i).name.startswith("link_")
            or model.body(i).name
            in {
                "dg5fs_mount",
                "right_hand",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
                "upper_arm_link",
                "forearm_link",
                "shoulder_link",
            }
        }

    def min_self_distance(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        distances = [
            float(c.dist)
            for c in self.data.contact
            if self.model.geom_bodyid[c.geom1] in self.robot_bodies
            and self.model.geom_bodyid[c.geom2] in self.robot_bodies
        ]
        return min(distances, default=0.01)

    def solve(
        self,
        qpos,
        targets,
        *,
        allow_wrist=False,
        max_delta_rad=0.02,
        max_penetration_m=0.001,
        tolerance_m=0.001,
        iterations=35,
    ):
        qpos = np.asarray(qpos, dtype=float).copy()
        active = self.actuated if allow_wrist else self.hand_q
        if not targets:
            return IKResult(True, qpos, 0.0, self.min_self_distance(qpos), "no motion")
        ids = [self.model.body(t.body).id for t in targets]
        bounds = []
        for q in active:
            j = self.q_to_joint[int(q)]
            lo, hi = self.model.jnt_range[j]
            if not self.model.jnt_limited[j]:
                lo, hi = -np.inf, np.inf
            bounds.append((max(lo, qpos[q] - max_delta_rad), min(hi, qpos[q] + max_delta_rad)))
        if any(lo > hi for lo, hi in bounds):
            return IKResult(False, qpos, float("inf"), 0.0, "starting joint outside limits")

        def evaluate(x):
            q = qpos.copy()
            q[active] = x
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            points = [
                self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ t.local_point
                for b, t in zip(ids, targets, strict=True)
            ]
            return q, np.array(
                [p - np.asarray(t.world_point) for p, t in zip(points, targets, strict=True)]
            )

        def objective(x):
            _, err = evaluate(x)
            return sum(
                t.weight * np.dot(e, e) * 1e6 for t, e in zip(targets, err, strict=True)
            ) + 0.02 * np.sum((x - qpos[active]) ** 2)

        def collision_constraint(x):
            q, _ = evaluate(x)
            return self.min_self_distance(q) + max_penetration_m

        fit = minimize(
            objective,
            np.clip(qpos[active], np.array(bounds)[:, 0], np.array(bounds)[:, 1]),
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": collision_constraint}],
            options={"maxiter": iterations, "ftol": 1e-5},
        )
        q, err = evaluate(fit.x)
        maximum = float(np.max(np.linalg.norm(err, axis=1)))
        distance = min(self.min_self_distance(qpos + a * (q - qpos)) for a in np.linspace(0, 1, 5))
        feasible = bool(
            np.isfinite(q).all()
            and maximum <= tolerance_m
            and distance >= -max_penetration_m - 1e-7
        )
        return IKResult(
            feasible,
            q,
            maximum,
            distance,
            "ok"
            if feasible
            else f"{fit.message}; point error={maximum:.5g} m, self distance={distance:.5g} m",
        )


def integrate_ik_target(env, solution, current, *, allow_wrist=False):
    """Integrate resolved joint increments into the actuator's target state.

    Replacing a position target with each measured q cancels the servo's
    gravity/contact load error. Keep that error, bounded to avoid windup.
    """
    command = env.last_targets.copy()
    active = env.actuated_q if allow_wrist else np.concatenate(list(env.finger_q.values()))
    command[active] += solution[active] - current[active]
    command[active] = np.clip(command[active], current[active] - 0.15, current[active] + 0.15)
    for joint in env.actuated_joint_ids:
        index = env.kin_model.jnt_qposadr[joint]
        if env.kin_model.jnt_limited[joint]:
            command[index] = np.clip(command[index], *env.kin_model.jnt_range[joint])
    return command
