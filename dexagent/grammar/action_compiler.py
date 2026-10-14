"""Compile object effects into contact material-point targets; no joint scripts."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.controllers.fingertip_ik import PointTarget
from dexagent.controllers.guards import violation
from dexagent.grammar.finger_roles import FingerRole
from dexagent.grammar.schema import FailureCode, Pose
from dexagent.sim.object_state import matrix, pose, transform_point


class InfeasibleAction(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class CompiledAction:
    action: object
    target_world: Pose
    anchors: dict
    initial_world: Pose

    def point_targets(self, desired_world):
        return [
            PointTarget(
                link,
                tuple(local),
                tuple(transform_point(desired_world, objpoint)),
                2.0 if self.action.finger_roles[f] == FingerRole.SUPPORT else 1.0,
            )
            for f, (link, local, objpoint) in self.anchors.items()
        ]


def compile_action(action, state, *, fixture_pose_world=None):
    if action.object != state.object_state.name:
        raise InfeasibleAction(
            FailureCode.INVALID_ACTION, "Action names an object absent from this environment"
        )
    if action.mode in ("advance_cable", "wrap_cable"):
        raise InfeasibleAction(
            FailureCode.UNSUPPORTED,
            "Cable motion requires a deformable-state and tension controller; rigid-object IK cannot execute it",
        )
    bad = violation(state, action.constraints, roles=action.finger_roles)
    if bad:
        raise InfeasibleAction(*bad)
    current = matrix(state.object_state.pose_world)
    target = current.copy()
    if action.target.frame == "fixture":
        if fixture_pose_world is None:
            raise InfeasibleAction(FailureCode.INFEASIBLE, "Fixture frame is not defined")
        frame = matrix(fixture_pose_world)
    else:
        frame = current
    if action.mode == "rotate":
        axis = np.asarray(action.target.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        # Fixture rotation is about the object's origin, with axes in fixture coordinates.
        target[:3, :3] = (
            Rotation.from_rotvec(
                frame[:3, :3] @ axis * np.deg2rad(action.target.angle_deg)
            ).as_matrix()
            @ current[:3, :3]
        )
    elif action.mode == "translate":
        direction = np.asarray(action.target.direction, dtype=float)
        direction /= np.linalg.norm(direction)
        target[:3, 3] += frame[:3, :3] @ direction * action.target.distance_m
    else:
        target = frame @ matrix(action.target.pose)
    anchors = {}
    for f, role in action.finger_roles.items():
        if role in (FingerRole.SUPPORT, FingerRole.DRIVE):
            contacts = state.fingers[f].contacts
            if not contacts:
                raise InfeasibleAction(
                    FailureCode.CONTACT_LOST,
                    f"{f} has role {role} without object contact; establish it first",
                )
            contact = max(contacts, key=lambda c: c.normal_force_N or 0.0)
            anchors[f] = (
                contact.finger_link,
                np.asarray(contact.point_finger),
                # Preserve the measured normal gap/compression. Driving the
                # finger material patch to the undeformed object surface would
                # unload a compliant contact, despite a SUPPORT declaration.
                np.asarray(contact.point_object)
                + current[:3, :3].T @ np.asarray(contact.normal_world) * contact.separation_m,
            )
        elif role == FingerRole.RELEASE and state.fingers[f].contacts:
            raise InfeasibleAction(
                FailureCode.INFEASIBLE,
                f"Release {f} with an explicit transition before object motion",
            )
    return CompiledAction(action, pose(target), anchors, state.object_state.pose_world)
