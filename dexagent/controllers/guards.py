"""Measured execution constraints; role declarations do not prove support."""

import numpy as np

from dexagent.grammar.finger_roles import ContactMode, FingerRole
from dexagent.grammar.schema import FailureCode


def supporting_fingers(state, constraints, *, roles=None, exclude=()):
    roles = state.finger_roles if roles is None else roles
    return [
        f
        for f, s in state.fingers.items()
        if f not in exclude
        and roles[f] == FingerRole.SUPPORT
        and s.contact_state in (ContactMode.STICKING, ContactMode.ROLLING)
        and sum(c.normal_force_N or 0.0 for c in s.contacts) >= constraints.min_support_force_N
        and max((c.tangential_speed_m_s for c in s.contacts), default=float("inf"))
        <= constraints.max_slip_m_s
    ]


def violation(state, constraints, *, roles=None, require_support=True):
    if not state.force_estimates_available:
        return (
            FailureCode.FORCE_UNAVAILABLE,
            "Cannot verify force/support constraints without solved force estimates",
        )
    for f, s in state.fingers.items():
        depth = max((-c.separation_m for c in s.contacts), default=0.0)
        if depth > constraints.max_object_penetration_m:
            return (
                FailureCode.OBJECT_PENETRATION,
                f"{f}: object penetration {depth:.4g} m exceeds {constraints.max_object_penetration_m:g} m",
            )
        total = sum(c.normal_force_N or 0.0 for c in s.contacts)
        if total > constraints.max_normal_force_N:
            return (
                FailureCode.FORCE_LIMIT,
                f"{f}: summed normal force {total:.4g} N exceeds {constraints.max_normal_force_N:g} N",
            )
    if state.self_collision_penetration_m > constraints.max_self_penetration_m:
        return FailureCode.COLLISION, f"self penetration {state.self_collision_penetration_m:.4g} m"
    if (
        constraints.prevent_object_drop
        and np.linalg.norm(state.object_state.pose_palm.position)
        > constraints.max_object_distance_from_palm_m
    ):
        return FailureCode.DROPPED, "Object left the configured palm workspace"
    if (
        require_support
        and len(supporting_fingers(state, constraints, roles=roles))
        < constraints.min_support_contacts
    ):
        return FailureCode.INSUFFICIENT_SUPPORT, "Too few measured low-slip SUPPORT contacts"
    return None
