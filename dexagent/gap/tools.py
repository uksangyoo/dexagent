"""Typed hand.* operations registered directly in GaP's ToolRegistry."""

from typing import TypedDict

from pydantic import ValidationError

from dexagent.controllers.contact_controller import ContactController
from dexagent.controllers.fingertip_ik import FingertipIK
from dexagent.controllers.object_motion_controller import ObjectMotionController
from dexagent.gap.feedback import report
from dexagent.grammar.action_compiler import InfeasibleAction, compile_action
from dexagent.grammar.contact_transitions import CONTRACTS
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import (
    DexterousAction,
    FailureCode,
    SupportWrenchRequest,
    TransitionRequest,
)


class ToolResult(TypedDict):
    success: bool
    status: str
    report: dict


class StateResult(TypedDict):
    state: dict


class CatalogResult(TypedDict):
    action_schema: dict
    transition_schema: dict
    transitions: dict


class SupportResult(TypedDict):
    feasible: bool
    certificate: dict
    time_s: float
    assumptions: str


class HandTools:
    def __init__(self, env):
        self.env = env
        ik = FingertipIK(env.kin_model)
        self.contacts = ContactController(env, ik)
        self.motion = ObjectMotionController(env, ik)
        self.history = []

    def _result(self, value):
        if value.final_state is not None and value.elapsed_s > 0:
            start = value.final_state.time_s - value.elapsed_s
            events = [
                e
                for e in getattr(self.env, "contact_events", [])
                if start < e["time_s"] <= value.final_state.time_s
            ]
            if events:
                value.actual_transitions = events
                expected_release = set()
                if value.intended.get("operation") == "break_contact" or (
                    value.intended.get("operation") == "transfer_support" and value.success
                ):
                    expected_release.add(value.intended["finger"])
                value.lost_contact = sorted(
                    {
                        Finger(e["finger"])
                        for e in events
                        if e["to"] == "free" and e["finger"] not in expected_release
                    }
                )
                value.slipped = sorted(
                    {Finger(e["finger"]) for e in events if e["to"] == "sliding"}
                )
        self.history.append(value)
        return {
            "success": value.success,
            "status": "succeeded" if value.success else "failed",
            "report": value.model_dump(mode="json"),
        }

    def observe(self) -> StateResult:
        return {"state": self.env.observe().model_dump(mode="json")}

    def check_support_wrench(self, request: SupportWrenchRequest) -> SupportResult:
        """Evaluate a proposed release without changing the robot or its roles."""
        from dataclasses import asdict

        from dexagent.controllers.wrench_feasibility import certify_support
        from dexagent.sim.object_state import matrix

        request = SupportWrenchRequest.model_validate(request)
        with self.env.transaction():
            state = self.env.observe()
            transform = matrix(state.object_state.pose_world)
            com = transform[:3, 3] + transform[:3, :3] @ self.env.body_com[self.env.object_body]
            mass = float(self.env.model.body_mass.numpy()[self.env.object_body])
            certificate = certify_support(
                {
                    f: s.contacts
                    for f, s in state.fingers.items()
                    if f not in request.exclude_fingers
                },
                center_of_mass_world=com,
                mass_kg=mass,
                friction=request.friction,
                max_force_per_finger_N=request.max_force_per_finger_N,
                disturbance_force=request.disturbance_force_world_N,
                disturbance_torque=request.disturbance_torque_world_Nm,
            )
            return {
                "feasible": certificate.feasible,
                "certificate": asdict(certificate),
                "time_s": state.time_s,
                "assumptions": "Static point contacts, specified friction and per-finger force bounds; excludes palm/table support. Does not certify dynamics, joint torque capacity, or reachability.",
            }

    def schema(self) -> CatalogResult:
        from dataclasses import asdict

        return {
            "action_schema": DexterousAction.model_json_schema(),
            "transition_schema": TransitionRequest.model_json_schema(),
            "transitions": {k: asdict(v) for k, v in CONTRACTS.items()},
        }

    def last_report(self) -> ToolResult:
        if not self.history:
            state = self.env.observe()
            return self._result(
                report(
                    state,
                    state,
                    success=False,
                    feasible=False,
                    code=FailureCode.INVALID_ACTION,
                    message="No operation has executed",
                )
            )
        value = self.history[-1]
        return {
            "success": value.success,
            "status": "succeeded" if value.success else "failed",
            "report": value.model_dump(mode="json"),
        }

    def assign_roles(self, roles: dict[str, str]) -> ToolResult:
        with self.env.transaction():
            before = self.env.observe()
            try:
                parsed = {Finger(k): FingerRole(v) for k, v in roles.items()}
                if set(parsed) != set(Finger):
                    raise ValueError("Specify all five roles")
            except ValueError as exc:
                return self._result(
                    report(
                        before,
                        before,
                        success=False,
                        feasible=False,
                        code=FailureCode.INVALID_ACTION,
                        message=str(exc),
                    )
                )
            self.env.roles = parsed
            return self._result(
                report(
                    before,
                    self.env.observe(),
                    success=True,
                    intended={"finger_roles": roles},
                    message="Intent roles assigned; measured contact state is unchanged and execution support remains unverified",
                )
            )

    def transition_contact(self, request: TransitionRequest) -> ToolResult:
        try:
            parsed = TransitionRequest.model_validate(request)
            return self._result(self.contacts.execute(parsed))
        except ValidationError as exc:
            state = self.env.observe()
            return self._result(
                report(
                    state,
                    state,
                    success=False,
                    feasible=False,
                    code=FailureCode.INVALID_ACTION,
                    message=str(exc),
                )
            )

    def move_object(self, action: DexterousAction) -> ToolResult:
        try:
            parsed = DexterousAction.model_validate(action)
            return self._result(self.motion.execute(parsed))
        except ValidationError as exc:
            state = self.env.observe()
            return self._result(
                report(
                    state,
                    state,
                    success=False,
                    feasible=False,
                    code=FailureCode.INVALID_ACTION,
                    message=str(exc),
                )
            )

    def validate_action(self, action: DexterousAction) -> ToolResult:
        state = self.env.observe()
        try:
            parsed = DexterousAction.model_validate(action)
            compile_action(parsed, state, fixture_pose_world=self.env.fixture_pose_world)
        except (ValidationError, InfeasibleAction) as exc:
            return self._result(
                report(
                    state,
                    state,
                    success=False,
                    feasible=False,
                    code=exc.code
                    if isinstance(exc, InfeasibleAction)
                    else FailureCode.INVALID_ACTION,
                    message=str(exc),
                )
            )
        return self._result(
            report(
                state,
                state,
                success=True,
                message="Contact/frame preflight passed; reachability is checked by the controller before each increment",
            )
        )

    def register(self, registry):
        descriptions = {
            "observe": "Read measured finger contacts, solved forces, slip, and object pose/twist.",
            "schema": "Get complete JSON schemas, units, and contact-operation contracts.",
            "last_report": "Read the last operation result for graph exits and repair.",
            "assign_roles": "Assign intended finger functions; does not create physical contacts.",
            "transition_contact": "Execute a guarded contact transition and return physical failure diagnostics.",
            "move_object": "Compile object-frame motion to contact-preserving IK with measured constraint checks.",
            "validate_action": "Check schema, frame, contact, and support prerequisites without moving.",
            "check_support_wrench": "Check gravity/disturbance support after a proposed finger release using measured contacts and an explicit friction assumption.",
        }
        for name, summary in descriptions.items():
            registry.register_callable(
                f"hand.{name}",
                getattr(self, name),
                summary=summary,
                tags=("sim_step",) if name in ("transition_contact", "move_object") else (),
                requires_ctx=False,
            )
