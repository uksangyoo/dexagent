"""Bounded material-point servo for contact feasibility experiments.

Targets are finger-local surface patches transformed by an object pose. No
object state is assigned, welded, or actuated. This controller is experimental;
its caller must apply measured force, penetration, and slip gates.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from dexagent.grammar.finger_roles import Finger


@dataclass
class MaterialPatch:
    finger: Finger
    body: str
    finger_local: np.ndarray
    object_local: np.ndarray
    normal_object: np.ndarray
    separation: float


class MaterialServo:
    def __init__(self, env, max_speed=0.4):
        self.env = env
        self.data = mujoco.MjData(env.kin_model)
        self.max_speed = max_speed

    def command(self, targets):
        """targets: iterable of (finger, body name, finger-local point, world target)."""
        env = self.env
        model = env.kin_model
        q = env.native_q()
        self.data.qpos[:] = q
        mujoco.mj_forward(model, self.data)
        command = env.last_targets.copy()
        for finger, body, local, target in targets:
            b = model.body(body).id
            world = self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ local
            error = np.asarray(target) - world
            error *= min(1.0, 0.0003 / max(np.linalg.norm(error), 1e-12))
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, self.data, jac, None, world, b)
            ids = env.finger_q[finger]
            J = jac[:, env.finger_v[finger]]
            delta = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), error)
            command[ids] += np.clip(delta, -self.max_speed * env.dt, self.max_speed * env.dt)
            command[ids] = np.clip(command[ids], q[ids] - 0.1, q[ids] + 0.1)
            for j, idx in zip(
                [
                    model.joint(f"joint_{list(Finger).index(finger) + 1}_{i}").id
                    for i in range(1, 5)
                ],
                ids,
                strict=True,
            ):
                command[idx] = np.clip(command[idx], *model.jnt_range[j])
        return command
