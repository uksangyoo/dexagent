"""A–F interfaces over a shared scene, IK implementation, and joint servo.

These adapters do not contain an agent. Agent-generated decisions and budgets
are recorded by the experiment runner; interface dimension is not a score.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dexagent.controllers.fingertip_ik import PointTarget, integrate_ik_target
from dexagent.controllers.guards import violation
from dexagent.gap.feedback import report
from dexagent.grammar.finger_roles import FINGER_NUMBERS, LINK_NAMES, Finger
from dexagent.grammar.schema import DexterousAction, FailureCode, TransitionRequest
from dexagent.sim.object_state import matrix, transform_point


class Representation(StrEnum):
    JOINTS = "A"
    SYNERGIES = "B"
    FINGERTIPS = "C"
    PRIMITIVES = "D"
    STATIC_GRAMMAR = "E"
    DYNAMIC_GRAMMAR = "F"


@dataclass
class SynergyBasis:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    training_case_ids: tuple

    @classmethod
    def fit(cls, postures, case_ids, *, split, rank=6):
        if split != "train":
            raise ValueError("Fit synergy bases on training cases only")
        q = np.asarray(postures, dtype=float)
        if q.ndim != 2 or q.shape[1] != 20 or len(q) <= rank or not np.isfinite(q).all():
            raise ValueError("Expected finite training hand postures, shape [N,20], N>rank")
        if not 1 <= rank <= 20 or len(case_ids) != len(q):
            raise ValueError("Invalid rank/provenance")
        mean = q.mean(0)
        _, singular, vt = np.linalg.svd(q - mean, full_matrices=False)
        if singular[rank - 1] < 1e-8:
            raise ValueError("Training data does not span the requested synergy rank")
        return cls(
            mean, vt[:rank].T, singular[:rank] / np.sqrt(len(q) - 1), tuple(sorted(set(case_ids)))
        )

    @property
    def fingerprint(self):
        return hashlib.sha256(
            self.mean.tobytes()
            + self.basis.tobytes()
            + self.scale.tobytes()
            + repr(self.training_case_ids).encode()
        ).hexdigest()

    def decode(self, coefficients):
        c = np.asarray(coefficients, dtype=float)
        if c.shape != self.scale.shape or not np.isfinite(c).all() or np.any(abs(c) > 1):
            raise ValueError("Invalid normalized synergy coefficients")
        return self.mean + self.basis @ (c * self.scale)


class RepresentationAdapter:
    def __init__(
        self,
        kind,
        connector,
        constraints,
        *,
        synergy=None,
        fixed_roles=None,
        primitive_templates=None,
        max_action_seconds=5.0,
    ):
        self.kind = Representation(kind)
        self.conn = connector
        self.env = connector.env
        self.constraints = constraints
        self.synergy = synergy
        self.fixed_roles = dict(fixed_roles or self.env.roles)
        self.templates = primitive_templates or {}
        self.remaining_simulated_seconds = float("inf")
        self.max_action_seconds = float(max_action_seconds)
        if not np.isfinite(self.max_action_seconds) or self.max_action_seconds <= 0:
            raise ValueError("Expected a positive finite shared action horizon")
        if constraints.allow_wrist_motion:
            raise ValueError(
                "A–F currently requires a fixed wrist; a shared arm interface is pending"
            )
        if self.kind == Representation.STATIC_GRAMMAR:
            if set(self.fixed_roles) != set(Finger):
                raise ValueError("E requires all five fixed roles")
            self.env.roles = dict(self.fixed_roles)
        if self.kind == Representation.SYNERGIES and synergy is None:
            raise ValueError("B requires a fitted training-only synergy basis")

    def execute(self, payload):
        before = self.env.observe()
        try:
            if self.kind not in (
                Representation.JOINTS,
                Representation.SYNERGIES,
                Representation.FINGERTIPS,
            ):
                return self._execute(payload)
            # Low-level interfaces can execute for the same physical horizon
            # as a grammar call, rather than spending one agent decision per
            # 20 ms tick. A is a normalized joint rate over this interval; B/C
            # keep resolving their posture/Cartesian target during the interval.
            duration = float(payload.get("duration_s", 0.2))
            if not np.isfinite(duration) or not self.env.dt <= duration <= self.max_action_seconds:
                raise ValueError("duration_s must fit the shared action horizon")
            ticks = int(
                np.floor(min(duration, self.remaining_simulated_seconds) / self.env.dt + 1e-8)
            )
            if ticks < 1:
                return report(
                    before,
                    before,
                    success=False,
                    feasible=False,
                    code=FailureCode.TIMEOUT,
                    message="Physical-time budget exhausted",
                    intended=payload,
                )
            increments = []
            for _ in range(ticks):
                result = self._execute(payload)
                increments.append(result)
                if not result.success:
                    break
            result.elapsed_s = max(0.0, self.env.observe().time_s - before.time_s)
            result.control_steps = sum(r.control_steps for r in increments)
            result.actual_transitions = [e for r in increments for e in r.actual_transitions]
            result.lost_contact = sorted({f for r in increments for f in r.lost_contact})
            result.slipped = sorted({f for r in increments for f in r.slipped})
            return result
        except (ValueError, KeyError, TypeError) as exc:
            after = self.env.observe()
            moved = after.time_s > before.time_s
            return report(
                before,
                after,
                success=False,
                feasible=moved,
                code=FailureCode.CONTROLLER_FAILURE if moved else FailureCode.INVALID_ACTION,
                message=str(exc),
                intended=payload,
            )

    def _execute(self, payload):
        before = self.env.observe()

        def rejected(message):
            return report(
                before,
                before,
                success=False,
                feasible=False,
                code=FailureCode.INFEASIBLE,
                message=message,
                intended=payload,
            )

        def capped(value):
            a, b = value.constraints, self.constraints
            if (
                a.max_normal_force_N > b.max_normal_force_N
                or a.max_joint_speed_rad_s > b.max_joint_speed_rad_s
                or a.max_self_penetration_m > b.max_self_penetration_m
                or a.max_object_penetration_m > b.max_object_penetration_m
                or a.max_slip_m_s > b.max_slip_m_s
                or a.max_object_distance_from_palm_m > b.max_object_distance_from_palm_m
                or a.min_support_contacts < b.min_support_contacts
                or a.min_support_force_N < b.min_support_force_N
                or (b.prevent_object_drop and not a.prevent_object_drop)
            ):
                raise ValueError("Action exceeds the experiment's shared physical limits")
            if a.allow_wrist_motion != b.allow_wrist_motion:
                raise ValueError("All representations must use the same wrist-motion regime")
            limit = min(
                value.termination.timeout_s,
                self.remaining_simulated_seconds,
                self.max_action_seconds,
            )
            return value.model_copy(
                update={"termination": value.termination.model_copy(update={"timeout_s": limit})}
            )

        if self.kind in (Representation.STATIC_GRAMMAR, Representation.DYNAMIC_GRAMMAR):
            op = payload.get("operation", "move_object")
            if op == "assign_roles":
                if self.kind == Representation.STATIC_GRAMMAR:
                    return rejected("E freezes roles for the full episode")
                result = self.conn.hand.assign_roles(payload["roles"])
            elif op == "transition_contact":
                request = capped(TransitionRequest.model_validate(payload["request"]))
                if (
                    self.kind == Representation.STATIC_GRAMMAR
                    and request.operation == "transfer_support"
                ):
                    # A release changes the finger's functional participation.
                    return rejected(
                        "This transition changes roles; unavailable under the fixed-role ablation"
                    )
                result = self.conn.hand.transition_contact(request)
                if self.kind == Representation.STATIC_GRAMMAR:
                    self.env.roles = dict(self.fixed_roles)
            elif op == "move_object":
                action = capped(DexterousAction.model_validate(payload["action"]))
                if (
                    self.kind == Representation.STATIC_GRAMMAR
                    and action.finger_roles != self.fixed_roles
                ):
                    return rejected("E cannot change its fixed finger roles")
                result = self.conn.hand.move_object(action)
            else:
                return rejected("Unknown grammar operation")
            from dexagent.grammar.schema import ExecutionReport

            return ExecutionReport.model_validate(result["report"])
        if self.kind == Representation.PRIMITIVES:
            name = payload.get("primitive")
            if name not in self.templates:
                return rejected("Primitive is outside the frozen library")
            # Templates are frozen full actions; only the declared angle/distance
            # scalar is exposed, not roles, ordering, or constraint bypasses.
            action = self.templates[name].model_copy(deep=True)
            data = action.model_dump(mode="json")
            if action.mode == "rotate":
                data["target"]["angle_deg"] = payload["magnitude"]
            elif action.mode == "translate":
                data["target"]["distance_m"] = payload["magnitude"]
            else:
                return rejected("Primitive template has no exposed scalar")
            return self.conn.hand.motion.execute(capped(DexterousAction.model_validate(data)))
        bad = violation(before, self.constraints, require_support=False)
        if bad:
            return report(
                before,
                before,
                success=False,
                feasible=False,
                code=bad[0],
                message=bad[1],
                intended=payload,
            )
        q = self.env.native_q()
        hand = np.concatenate(list(self.env.finger_q.values()))
        if self.kind == Representation.JOINTS:
            a = np.asarray(payload["delta"], dtype=float)
            if a.shape != (20,) or not np.isfinite(a).all() or np.any(abs(a) > 1):
                return rejected("A expects twenty normalized joint increments")
            q[hand] = (
                self.env.last_targets[hand]
                + a * self.constraints.max_joint_speed_rad_s * self.env.dt
            )
        elif self.kind == Representation.SYNERGIES:
            decoded = self.synergy.decode(payload["coefficients"])
            delta = np.clip(
                decoded - self.env.last_targets[hand],
                -self.constraints.max_joint_speed_rad_s * self.env.dt,
                self.constraints.max_joint_speed_rad_s * self.env.dt,
            )
            q[hand] = self.env.last_targets[hand] + delta
        else:
            if set(payload["points_palm"]) != {f.value for f in Finger}:
                return rejected("C requires all five palm-frame fingertip positions")
            targets = []
            palm = matrix(before.palm_pose_world)
            import mujoco

            data = self.conn.hand.contacts.ik.data
            data.qpos[:] = q
            mujoco.mj_forward(self.env.kin_model, data)
            for f, i in FINGER_NUMBERS.items():
                world = transform_point(palm, payload["points_palm"][f.value])
                site = self.env.kin_model.site(f"tip_{i}")
                current = data.site_xpos[site.id].copy()
                delta = np.asarray(world) - current
                world = current + delta * min(1.0, 0.0005 / max(np.linalg.norm(delta), 1e-12))
                targets.append(PointTarget(LINK_NAMES[f][-1], tuple(site.pos), tuple(world)))
            solved = self.conn.hand.contacts.ik.solve(
                q,
                targets,
                max_delta_rad=self.constraints.max_joint_speed_rad_s * self.env.dt,
                max_penetration_m=self.constraints.max_self_penetration_m,
                tolerance_m=0.0006,
            )
            if not solved.feasible:
                return rejected(solved.reason)
            q = integrate_ik_target(self.env, solved.qpos, q)
        for j in self.env.actuated_joint_ids:
            i = self.env.kin_model.jnt_qposadr[j]
            lo, hi = self.env.kin_model.jnt_range[j]
            if self.env.kin_model.jnt_limited[j] and not lo <= q[i] <= hi:
                return rejected("Joint limit violated")
        ik = self.conn.hand.contacts.ik
        if (
            min(
                ik.min_self_distance(self.env.native_q() + a * (q - self.env.native_q()))
                for a in np.linspace(0, 1, 5)
            )
            < -self.constraints.max_self_penetration_m
        ):
            return rejected("Interpolated joint command violates the shared self-collision limit")
        self.env.step_targets(q, constraints=self.constraints)
        after = self.env.observe()
        bad = self.env.last_constraint_violation or violation(
            after, self.constraints, require_support=False
        )
        return report(
            before,
            after,
            success=bad is None,
            code=None if bad is None else bad[0],
            message="control increment" if bad is None else bad[1],
            intended=payload,
            steps=1,
        )
