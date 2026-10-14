"""Declarative contracts shared by the controller and agent-facing catalog."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionContract:
    preconditions: tuple[str, ...]
    targets: tuple[str, ...]
    success: tuple[str, ...]
    failures: tuple[str, ...]
    implemented: bool = True


CONTRACTS = {
    "establish_contact": TransitionContract(
        ("reachable object-frame surface point", "other support preserved"),
        ("approach surface normal", "bounded normal preload"),
        ("contact near requested point", "positive solved normal force", "stable low slip"),
        ("ik_failure", "force_limit", "contact_lost", "timeout"),
    ),
    "maintain_contact": TransitionContract(
        ("finger has target-object contact",),
        ("follow object material contact point", "bounded preload"),
        ("contact, force, and slip valid for hold duration",),
        ("contact_lost", "slip", "force_limit"),
    ),
    "break_contact": TransitionContract(
        ("remaining fingers satisfy support constraint",),
        ("retreat along outward object normal",),
        ("no solved contact for stable_steps",),
        ("insufficient_support", "ik_failure", "timeout"),
    ),
    "transfer_support": TransitionContract(
        ("donor is verified support", "distinct receiver", "reachable receiver point"),
        ("establish receiver", "verify receiver support", "release donor"),
        ("receiver supports", "donor released", "support count never falls below minimum"),
        ("insufficient_support", "contact_lost", "force_limit", "timeout"),
    ),
    "slide_contact": TransitionContract(
        ("calibrated tangential/normal force controller",),
        ("surface-constrained sliding",),
        ("target patch reached with bounded load",),
        ("unsupported",),
        False,
    ),
    "roll_contact": TransitionContract(
        ("calibrated surface-curvature and rolling controller",),
        ("no-slip contact migration",),
        ("persistent rolling evidence and target patch",),
        ("unsupported",),
        False,
    ),
}
