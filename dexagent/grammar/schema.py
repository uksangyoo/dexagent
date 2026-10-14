"""Versioned, SI-unit contracts. Quaternions use GaP's scalar-first convention."""

import math
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .finger_roles import ContactMode, Finger, FingerRole

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vec3 = tuple[Finite, Finite, Finite]
Quat = tuple[Finite, Finite, Finite, Finite]
Frame = Literal["object", "fixture"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Pose(Contract):
    position: Vec3
    quaternion_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def unit_quaternion(self):
        if not math.isclose(np.linalg.norm(self.quaternion_wxyz), 1.0, abs_tol=1e-5):
            raise ValueError("quaternion_wxyz must be a unit quaternion")
        return self


class Twist(Contract):
    linear: Vec3 = (0.0, 0.0, 0.0)
    angular: Vec3 = (0.0, 0.0, 0.0)


class SurfaceContact(Contract):
    object: str
    finger_link: str
    shape_pair: tuple[int, int]
    point_world: Vec3
    point_palm: Vec3
    point_object: Vec3
    point_finger: Vec3
    normal_world: Vec3  # Outward from object toward finger.
    force_on_finger_world_N: Vec3 | None = None
    normal_force_N: Annotated[Finite, Field(ge=0)] | None = None
    tangential_speed_m_s: Annotated[Finite, Field(ge=0)]
    normal_speed_m_s: Finite
    rolling_angular_speed_rad_s: Annotated[Finite, Field(ge=0)] = 0.0
    separation_m: Finite
    sample_time_s: Finite
    force_source: Literal["mujoco_warp_constraint", "mujoco_constraint", "unavailable"]


class FingerState(Contract):
    joint_positions: tuple[Finite, Finite, Finite, Finite]
    joint_velocities: tuple[Finite, Finite, Finite, Finite]
    fingertip_pose_palm: Pose
    contact_state: ContactMode
    contacts: list[SurfaceContact] = Field(default_factory=list)
    classification_confidence: Annotated[Finite, Field(ge=0, le=1)] = 0.0
    classification_evidence: list[str] = Field(default_factory=list)


class ObjectState(Contract):
    name: str
    pose_world: Pose
    pose_palm: Pose
    twist_palm: Twist  # Object origin relative to the moving palm origin.
    twist_world: Twist
    externally_supported: bool = False
    free_body: bool = False
    external_contact_force_N: Annotated[Finite, Field(ge=0)] = 0.0
    palm_contact_force_N: Annotated[Finite, Field(ge=0)] = 0.0


class HandState(Contract):
    schema_version: Literal["1.0"] = "1.0"
    time_s: Finite
    contact_time_s: Finite
    palm_pose_world: Pose
    fingers: dict[Finger, FingerState]
    finger_roles: dict[Finger, FingerRole]
    object_state: ObjectState
    force_estimates_available: bool
    self_collision_penetration_m: Annotated[Finite, Field(ge=0)] = 0.0

    @model_validator(mode="after")
    def all_fingers(self):
        if set(self.fingers) != set(Finger) or set(self.finger_roles) != set(Finger):
            raise ValueError("State and roles must explicitly contain all five fingers")
        return self


class Constraints(Contract):
    min_support_contacts: Annotated[int, Field(ge=0, le=5)] = 2
    max_normal_force_N: Annotated[Finite, Field(gt=0)]  # No implicit physical limit.
    min_support_force_N: Annotated[Finite, Field(gt=0)] = 0.05
    max_slip_m_s: Annotated[Finite, Field(gt=0)] = 0.005
    max_self_penetration_m: Annotated[Finite, Field(ge=0)] = 0.001
    max_object_penetration_m: Annotated[Finite, Field(ge=0)] = 0.001
    prevent_object_drop: bool = True
    max_object_distance_from_palm_m: Annotated[Finite, Field(gt=0)] = 0.25
    max_joint_speed_rad_s: Annotated[Finite, Field(gt=0)] = 1.0
    allow_wrist_motion: bool = False
    calibration_id: str = "simulation_provisional"

    @model_validator(mode="after")
    def ordered_force_limits(self):
        if self.min_support_force_N > self.max_normal_force_N:
            raise ValueError("Minimum support force exceeds the force limit")
        return self


class SupportWrenchRequest(Contract):
    friction: Annotated[Finite, Field(ge=0, le=3)]
    max_force_per_finger_N: Annotated[Finite, Field(gt=0)]
    exclude_fingers: list[Finger] = Field(default_factory=list)
    disturbance_force_world_N: Vec3 = (0.0, 0.0, 0.0)
    disturbance_torque_world_Nm: Vec3 = (0.0, 0.0, 0.0)


class Termination(Contract):
    rotation_error_deg: Annotated[Finite, Field(gt=0, le=180)] = 3.0
    position_error_m: Annotated[Finite, Field(gt=0)] = 0.002
    contact_error_m: Annotated[Finite, Field(gt=0)] = 0.003
    timeout_s: Annotated[Finite, Field(gt=0, le=120)] = 5.0
    stable_steps: Annotated[int, Field(ge=2)] = 5
    hold_s: Annotated[Finite, Field(ge=0)] = 0.2


class RotateTarget(Contract):
    axis: Vec3
    angle_deg: Finite
    frame: Frame = "object"

    @model_validator(mode="after")
    def nonzero_axis(self):
        if np.linalg.norm(self.axis) < 1e-8:
            raise ValueError("rotation axis must be nonzero")
        return self


class TranslateTarget(Contract):
    direction: Vec3
    distance_m: Annotated[Finite, Field(ge=0)]
    frame: Frame = "object"

    @model_validator(mode="after")
    def nonzero_direction(self):
        if np.linalg.norm(self.direction) < 1e-8:
            raise ValueError("translation direction must be nonzero")
        return self


class ReorientTarget(Contract):
    pose: Pose
    frame: Frame = "fixture"


class CableFeedTarget(Contract):
    length_m: Finite
    frame: Frame = "fixture"


class CableWrapTarget(RotateTarget):
    tension_target_N: Annotated[Finite, Field(gt=0)]
    frame: Literal["fixture"] = "fixture"


class DexterousAction(Contract):
    mode: Literal["rotate", "translate", "reorient", "advance_cable", "wrap_cable"]
    object: str
    target: RotateTarget | TranslateTarget | ReorientTarget | CableFeedTarget | CableWrapTarget
    finger_roles: dict[Finger, FingerRole]
    constraints: Constraints
    termination: Termination = Field(default_factory=Termination)

    @model_validator(mode="after")
    def mode_contract(self):
        types = {
            "rotate": RotateTarget,
            "translate": TranslateTarget,
            "reorient": ReorientTarget,
            "advance_cable": CableFeedTarget,
            "wrap_cable": CableWrapTarget,
        }
        if type(self.target) is not types[self.mode]:
            raise ValueError(f"Target does not match action mode {self.mode}")
        if set(self.finger_roles) != set(Finger):
            raise ValueError("All five finger roles must be explicit")
        if FingerRole.DRIVE not in self.finger_roles.values():
            raise ValueError("Object motion requires at least one DRIVE finger")
        if (
            list(self.finger_roles.values()).count(FingerRole.SUPPORT)
            < self.constraints.min_support_contacts
        ):
            raise ValueError("Too few SUPPORT role assignments for requested constraint")
        return self


class TransitionRequest(Contract):
    operation: Literal[
        "establish_contact",
        "break_contact",
        "maintain_contact",
        "slide_contact",
        "roll_contact",
        "transfer_support",
    ]
    object: str
    finger: Finger
    object_point: Vec3 | None = None
    to_finger: Finger | None = None
    constraints: Constraints
    termination: Termination = Field(default_factory=Termination)

    @model_validator(mode="after")
    def operation_contract(self):
        if (
            self.operation
            in {"establish_contact", "slide_contact", "roll_contact", "transfer_support"}
            and self.object_point is None
        ):
            raise ValueError("This transition requires an object-frame surface point")
        if self.operation == "transfer_support":
            if self.to_finger is None or self.to_finger == self.finger:
                raise ValueError("Support transfer requires a distinct receiving finger")
        elif self.to_finger is not None:
            raise ValueError("to_finger is only meaningful for support transfer")
        return self


class FailureCode(StrEnum):
    INVALID_ACTION = "invalid_action"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    FORCE_UNAVAILABLE = "force_unavailable"
    FORCE_LIMIT = "force_limit"
    COLLISION = "self_collision"
    OBJECT_PENETRATION = "object_penetration"
    CONTACT_LOST = "contact_lost"
    SLIP = "slip"
    DROPPED = "object_dropped"
    TIMEOUT = "timeout"
    IK_FAILURE = "ik_failure"
    STALE_STATE = "stale_state"
    CONTROLLER_FAILURE = "controller_failure"


class ExecutionReport(Contract):
    success: bool
    failure_code: FailureCode | None = None
    message: str = ""
    feasible: bool
    intended: dict = Field(default_factory=dict)
    actual_transitions: list[dict] = Field(default_factory=list)
    lost_contact: list[Finger] = Field(default_factory=list)
    slipped: list[Finger] = Field(default_factory=list)
    constraint_violations: list[str] = Field(default_factory=list)
    object_position_error_m: Finite | None = None
    object_rotation_error_deg: Finite | None = None
    elapsed_s: Annotated[Finite, Field(ge=0)] = 0.0
    control_steps: Annotated[int, Field(ge=0)] = 0
    final_state: HandState | None = None
    repair_hints: list[str] = Field(default_factory=list)
