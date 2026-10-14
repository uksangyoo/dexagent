"""Guarded contact-transition state machine, with receiver-first support transfer."""

import numpy as np

from dexagent.gap.feedback import report
from dexagent.grammar.contact_transitions import CONTRACTS
from dexagent.grammar.finger_roles import LINK_NAMES, FingerRole
from dexagent.grammar.schema import FailureCode, TransitionRequest
from dexagent.sim.object_state import matrix, transform_point

from .fingertip_ik import FingertipIK, PointTarget, integrate_ik_target
from .guards import supporting_fingers, violation


class ContactController:
    def __init__(self, env, ik=None):
        self.env = env
        self.ik = ik or FingertipIK(env.kin_model)
        self.last_trace = []
        stiffness = getattr(env, "contact_stiffness_N_m", None)
        self.force_gain = min(0.0005, 0.25 / stiffness) if stiffness else 0.0005

    def execute(self, request: TransitionRequest):
        with self.env.transaction():
            self.last_trace = []
            before = self.env.observe()
            intended = request.model_dump(mode="json")

            def fail(code, message, feasible=False, steps=0):
                return report(
                    before,
                    self.env.observe(),
                    success=False,
                    feasible=feasible,
                    code=code,
                    message=message,
                    intended=intended,
                    steps=steps,
                )

            if request.object != before.object_state.name:
                return fail(FailureCode.INVALID_ACTION, "Unknown target object")
            if not CONTRACTS[request.operation].implemented:
                return fail(
                    FailureCode.UNSUPPORTED,
                    "Sliding/rolling execution awaits calibrated contact control; mode observation is available",
                )
            if request.operation == "transfer_support":
                return self._transfer(request, before)
            finger = request.finger
            initial = before.fingers[finger].contacts
            if request.operation == "maintain_contact" and not initial:
                return fail(FailureCode.CONTACT_LOST, "Cannot maintain an absent contact")
            if request.operation == "break_contact":
                if (
                    len(supporting_fingers(before, request.constraints, exclude=(finger,)))
                    < request.constraints.min_support_contacts
                ):
                    return fail(
                        FailureCode.INSUFFICIENT_SUPPORT,
                        "Release would leave too few verified supporting fingers",
                    )
                if before.object_state.free_body and hasattr(self.env, "support_certificate"):
                    certificate = self.env.support_certificate(
                        exclude=(finger,),
                        max_force_per_finger_N=request.constraints.max_normal_force_N,
                    )
                    if not certificate.feasible:
                        return fail(
                            FailureCode.INFEASIBLE,
                            "Release fails static support-wrench check: " + certificate.reason,
                        )
                if not initial:
                    return report(
                        before, before, success=True, intended=intended, message="Already released"
                    )
            try:
                target, normal, local = self.env.contact_target(finger, request.object_point)
            except ValueError as exc:
                return fail(FailureCode.INFEASIBLE, str(exc))
            if initial and request.operation in ("maintain_contact", "break_contact"):
                contact = max(initial, key=lambda c: c.normal_force_N or 0.0)
                target = np.asarray(contact.point_object)
                normal = np.asarray(contact.normal_world)
                local = np.asarray(contact.point_finger)
                body = contact.finger_link
            else:
                body = LINK_NAMES[finger][-1]
            # Normal is attached to the object, not held fixed in world space.
            normal_object = matrix(before.object_state.pose_world)[:3, :3].T @ normal
            stable = 0
            phase = "approach"
            established_at = None
            preload = 0.0
            max_steps = int(np.floor(request.termination.timeout_s / self.env.dt))
            for step in range(max_steps):
                state = self.env.observe()
                require_support = True
                roles = dict(state.finger_roles)
                if request.operation == "break_contact":
                    roles[finger] = FingerRole.RELEASE
                bad = violation(
                    state, request.constraints, roles=roles, require_support=require_support
                )
                if bad:
                    return fail(*bad, feasible=True, steps=step)
                own = state.fingers[finger]
                obj = matrix(state.object_state.pose_world)
                normal = obj[:3, :3] @ normal_object
                if request.operation == "break_contact":
                    good = not own.contacts
                else:
                    good = (
                        bool(own.contacts)
                        and sum(c.normal_force_N or 0.0 for c in own.contacts)
                        >= request.constraints.min_support_force_N
                    )
                    good = (
                        good
                        and max(
                            (c.tangential_speed_m_s for c in own.contacts), default=float("inf")
                        )
                        <= request.constraints.max_slip_m_s
                    )
                    good = (
                        good
                        and min(
                            (
                                np.linalg.norm(np.asarray(c.point_object) - target)
                                for c in own.contacts
                            ),
                            default=float("inf"),
                        )
                        <= request.termination.contact_error_m
                    )
                stable = stable + 1 if good else 0
                if stable >= request.termination.stable_steps:
                    if established_at is None:
                        established_at = state.time_s
                    phase = "hold"
                    if (
                        request.operation != "maintain_contact"
                        or state.time_s - established_at >= request.termination.hold_s
                    ):
                        if request.operation == "break_contact":
                            self.env.roles[finger] = FingerRole.FREE
                        return report(
                            before,
                            self.env.observe(),
                            success=True,
                            intended=intended,
                            steps=step,
                            message="Measured contact postconditions satisfied",
                        )
                elif established_at is not None:
                    return fail(
                        FailureCode.CONTACT_LOST,
                        "Contact maintenance failed after initial confirmation",
                        True,
                        step,
                    )
                targets = []
                # Track each existing support's actual link-local contact patch.
                for f, s in state.fingers.items():
                    if f == finger or state.finger_roles[f] != FingerRole.SUPPORT or not s.contacts:
                        continue
                    c = max(s.contacts, key=lambda c: c.normal_force_N or 0.0)
                    force = sum(c.normal_force_N or 0.0 for c in s.contacts)
                    desired_support_force = min(
                        request.constraints.max_normal_force_N * 0.9,
                        max(
                            request.constraints.min_support_force_N * 1.5,
                            sum(c.normal_force_N or 0 for c in before.fingers[f].contacts),
                            0.1,
                        ),
                    )
                    correction = self.force_gain * np.clip(desired_support_force - force, -1, 1)
                    support_point = transform_point(obj, c.point_object) + np.asarray(
                        c.normal_world
                    ) * (c.separation_m - correction)
                    targets.append(
                        PointTarget(
                            c.finger_link,
                            c.point_finger,
                            tuple(support_point),
                            3.0,
                        )
                    )
                point = transform_point(obj, target)
                if request.operation == "break_contact":
                    point = point + normal * 0.008
                else:
                    force = sum(c.normal_force_N or 0.0 for c in own.contacts)
                    desired = min(
                        request.constraints.max_normal_force_N * 0.25,
                        max(0.1, request.constraints.min_support_force_N * 2),
                    )
                    # This is a compliant target offset, not permitted physical
                    # penetration. Solved force/self-contact guards still stop
                    # at each 2 ms step, independently of this target bound.
                    if own.contacts:
                        # Once touching, use the measured normal and material
                        # patch. An approach normal at a sharp edge need not be
                        # the normal that actually transmits force after rolling.
                        c = min(
                            own.contacts,
                            key=lambda c: np.linalg.norm(np.asarray(c.point_object) - target),
                        )
                        normal = np.asarray(c.normal_world)
                        local = np.asarray(c.point_finger)
                        body = c.finger_link
                        surface = transform_point(obj, c.point_object)
                        current_patch = surface + normal * c.separation_m
                        tangent = point - surface
                        tangent -= normal * np.dot(normal, tangent)
                        tangent *= min(1.0, 0.0001 / max(np.linalg.norm(tangent), 1e-12))
                        normal_delta = self.force_gain * np.clip(desired - force, -0.4, 0.4)
                        point = current_patch + tangent - normal * normal_delta
                    else:
                        preload = float(
                            np.clip(
                                preload + self.force_gain * np.clip(desired - force, -1, 1),
                                -0.001,
                                0.004,
                            )
                        )
                        point = point - normal * preload
                targets.append(PointTarget(body, tuple(local), tuple(point), 2.0))
                q = self.env.native_q()
                # Move at most 0.5 mm per control interval toward a distant contact.
                d = self.ik.data
                d.qpos[:] = q
                import mujoco

                mujoco.mj_forward(self.ik.model, d)
                incremental = []
                for target_point in targets:
                    b = self.ik.model.body(target_point.body).id
                    current = d.xpos[b] + d.xmat[b].reshape(3, 3) @ target_point.local_point
                    delta = np.asarray(target_point.world_point) - current
                    delta *= min(1.0, 0.0005 / max(np.linalg.norm(delta), 1e-12))
                    incremental.append(
                        PointTarget(
                            target_point.body,
                            target_point.local_point,
                            tuple(current + delta),
                            target_point.weight,
                        )
                    )
                ik = self.ik.solve(
                    q,
                    incremental,
                    allow_wrist=request.constraints.allow_wrist_motion,
                    max_delta_rad=request.constraints.max_joint_speed_rad_s * self.env.dt,
                    max_penetration_m=request.constraints.max_self_penetration_m,
                    tolerance_m=0.0006,
                )
                if not ik.feasible:
                    return fail(FailureCode.IK_FAILURE, ik.reason, True, step)
                self.last_trace.append(
                    {
                        "time_s": state.time_s,
                        "phase": phase,
                        "finger": finger.value,
                        "contact_state": own.contact_state.value,
                        "support_count": len(supporting_fingers(state, request.constraints)),
                        "force_N": sum(c.normal_force_N or 0.0 for c in own.contacts),
                        "preload_m": preload,
                        "ik_error_m": ik.max_error_m,
                    }
                )
                command = integrate_ik_target(
                    self.env, ik.qpos, q, allow_wrist=request.constraints.allow_wrist_motion
                )
                self.env.step_targets(
                    command,
                    allow_wrist_motion=request.constraints.allow_wrist_motion,
                    constraints=request.constraints,
                    require_support=True,
                )
                if self.env.last_constraint_violation:
                    return fail(*self.env.last_constraint_violation, feasible=True, steps=step + 1)
            return fail(
                FailureCode.TIMEOUT,
                "Contact postconditions were not sustained before timeout",
                True,
                max_steps,
            )

    def _transfer(self, request, before):
        intended = request.model_dump(mode="json")
        if request.finger not in supporting_fingers(before, request.constraints):
            return report(
                before,
                before,
                success=False,
                feasible=False,
                code=FailureCode.INSUFFICIENT_SUPPORT,
                message="Transfer donor is not measured support",
                intended=intended,
            )
        started = self.env.time_s
        already_supporting = request.to_finger in supporting_fingers(before, request.constraints)
        if not already_supporting:
            self.env.roles[request.to_finger] = FingerRole.REPOSITION
        establish = request.model_copy(
            update={
                "operation": "establish_contact",
                "finger": request.to_finger,
                "to_finger": None,
            }
        )
        if already_supporting:
            # Preserve a loaded receiver's pose and role while confirming it.
            # An arbitrary new force target can unload a valid gravity grasp.
            for _ in range(request.termination.stable_steps):
                self.env.step_targets(
                    self.env.last_targets, constraints=request.constraints, require_support=True
                )
                if self.env.last_constraint_violation:
                    break
            current = self.env.observe()
            confirmed = (
                not self.env.last_constraint_violation
                and request.to_finger in supporting_fingers(current, request.constraints)
            )
            receiving = report(
                before,
                current,
                success=confirmed,
                code=None if confirmed else FailureCode.INSUFFICIENT_SUPPORT,
                message="Existing receiver support confirmed"
                if confirmed
                else "Receiver support lost during confirmation",
                intended=establish.model_dump(mode="json"),
                steps=request.termination.stable_steps,
            )
        else:
            receiving = self.execute(establish)
        trace = list(self.last_trace)
        if not receiving.success:
            # Keep the donor loaded and expose the intermediate state for repair.
            receiving.intended = intended
            receiving.message = "Receiver failed; donor was not released. " + receiving.message
            return receiving
        self.env.roles[request.to_finger] = FingerRole.SUPPORT
        state = self.env.observe()
        if (
            len(supporting_fingers(state, request.constraints, exclude=(request.finger,)))
            < request.constraints.min_support_contacts
        ):
            return report(
                before,
                state,
                success=False,
                feasible=False,
                code=FailureCode.INSUFFICIENT_SUPPORT,
                message="Receiver established, but releasing donor still violates support count",
                intended=intended,
            )
        remaining = request.termination.timeout_s - (self.env.time_s - started)
        if remaining <= 0:
            return report(
                before,
                state,
                success=False,
                code=FailureCode.TIMEOUT,
                message="Transfer budget exhausted before donor release",
                intended=intended,
            )
        release = TransitionRequest(
            operation="break_contact",
            object=request.object,
            finger=request.finger,
            constraints=request.constraints,
            termination=request.termination.model_copy(update={"timeout_s": remaining}),
        )
        released = self.execute(release)
        self.last_trace = trace + self.last_trace
        return report(
            before,
            self.env.observe(),
            success=released.success,
            feasible=released.feasible,
            code=released.failure_code,
            message=released.message,
            intended=intended,
            steps=receiving.control_steps + released.control_steps,
        )
