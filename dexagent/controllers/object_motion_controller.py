"""Receding-horizon contact-point IK with measured support/force guards."""

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.gap.feedback import report
from dexagent.grammar.action_compiler import InfeasibleAction, compile_action
from dexagent.grammar.finger_roles import FingerRole
from dexagent.grammar.schema import FailureCode
from dexagent.sim.object_state import matrix, pose_error

from .fingertip_ik import FingertipIK, integrate_ik_target
from .guards import violation


class ObjectMotionController:
    def __init__(self, env, ik=None):
        self.env = env
        self.ik = ik or FingertipIK(env.kin_model)

    def execute(self, action):
        with self.env.transaction():
            before = self.env.observe()
            intended = action.model_dump(mode="json")
            try:
                compiled = compile_action(
                    action, before, fixture_pose_world=self.env.fixture_pose_world
                )
            except InfeasibleAction as exc:
                return report(
                    before,
                    before,
                    success=False,
                    feasible=False,
                    code=exc.code,
                    message=str(exc),
                    intended=intended,
                )
            # The fixture curriculum permits only rotation about its passive axis.
            if before.object_state.externally_supported and not before.object_state.free_body:
                current = matrix(before.object_state.pose_world)
                target = matrix(compiled.target_world)
                rot = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
                axis = matrix(self.env.fixture_pose_world)[:3, 2]
                if (
                    np.linalg.norm(target[:3, 3] - current[:3, 3]) > 1e-5
                    or np.linalg.norm(np.cross(rot, axis)) > 1e-4
                ):
                    return report(
                        before,
                        before,
                        success=False,
                        feasible=False,
                        code=FailureCode.INFEASIBLE,
                        message="The passive fixture blocks this translation/rotation axis",
                        intended=intended,
                    )
            self.env.roles = dict(action.finger_roles)
            stable = 0
            max_steps = int(np.floor(action.termination.timeout_s / self.env.dt))
            for step in range(max_steps + 1):
                state = self.env.observe()
                pe, re = pose_error(state.object_state.pose_world, compiled.target_world)
                bad = violation(state, action.constraints)
                if bad:
                    return report(
                        before,
                        state,
                        success=False,
                        code=bad[0],
                        message=bad[1],
                        intended=intended,
                        steps=step,
                        position_error=pe,
                        rotation_error=re,
                    )
                for f, role in self.env.roles.items():
                    if (
                        role in (FingerRole.SUPPORT, FingerRole.DRIVE)
                        and not state.fingers[f].contacts
                    ):
                        return report(
                            before,
                            state,
                            success=False,
                            code=FailureCode.CONTACT_LOST,
                            message=f"{f} lost its contact",
                            intended=intended,
                            steps=step,
                            position_error=pe,
                            rotation_error=re,
                        )
                    if role in (FingerRole.SUPPORT, FingerRole.DRIVE) and any(
                        c.tangential_speed_m_s > action.constraints.max_slip_m_s
                        for c in state.fingers[f].contacts
                    ):
                        return report(
                            before,
                            state,
                            success=False,
                            code=FailureCode.SLIP,
                            message=f"{f} slipped",
                            intended=intended,
                            steps=step,
                            position_error=pe,
                            rotation_error=re,
                        )
                stable = (
                    stable + 1
                    if pe <= action.termination.position_error_m
                    and re <= action.termination.rotation_error_deg
                    else 0
                )
                if stable >= action.termination.stable_steps:
                    return report(
                        before,
                        state,
                        success=True,
                        intended=intended,
                        steps=step,
                        position_error=pe,
                        rotation_error=re,
                    )
                if step == max_steps:
                    break
                current = matrix(state.object_state.pose_world)
                target = matrix(compiled.target_world)
                error = target[:3, 3] - current[:3, 3]
                desired = current.copy()
                desired[:3, 3] += error * min(1.0, 0.0004 / max(np.linalg.norm(error), 1e-12))
                rot = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
                rot *= min(1.0, np.deg2rad(0.7) / max(np.linalg.norm(rot), 1e-12))
                desired[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ current[:3, :3]
                q = self.env.native_q()
                ik = self.ik.solve(
                    q,
                    compiled.point_targets(desired),
                    allow_wrist=action.constraints.allow_wrist_motion,
                    max_delta_rad=action.constraints.max_joint_speed_rad_s * self.env.dt,
                    max_penetration_m=action.constraints.max_self_penetration_m,
                    tolerance_m=0.0008,
                )
                if not ik.feasible:
                    return report(
                        before,
                        state,
                        success=False,
                        code=FailureCode.IK_FAILURE,
                        message=ik.reason,
                        intended=intended,
                        steps=step,
                        position_error=pe,
                        rotation_error=re,
                    )
                command = integrate_ik_target(
                    self.env, ik.qpos, q, allow_wrist=action.constraints.allow_wrist_motion
                )
                self.env.step_targets(
                    command,
                    allow_wrist_motion=action.constraints.allow_wrist_motion,
                    constraints=action.constraints,
                    require_support=True,
                )
                if self.env.last_constraint_violation:
                    code, message = self.env.last_constraint_violation
                    return report(
                        before,
                        self.env.observe(),
                        success=False,
                        code=code,
                        message=message,
                        intended=intended,
                        steps=step + 1,
                    )
            return report(
                before,
                self.env.observe(),
                success=False,
                code=FailureCode.TIMEOUT,
                message="Object pose target not reached",
                intended=intended,
                steps=max_steps,
                position_error=pe,
                rotation_error=re,
            )
