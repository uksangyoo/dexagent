"""Structured feedback available to GaP repair agents and experiment logs."""

from dexagent.grammar.finger_roles import ContactMode
from dexagent.grammar.schema import ExecutionReport, FailureCode


def report(
    before,
    after,
    *,
    success,
    feasible=True,
    code=None,
    message="",
    intended=None,
    steps=0,
    position_error=None,
    rotation_error=None,
):
    changes = []
    lost = []
    slipped = []
    expected_release = set()
    if intended and (
        intended.get("operation") == "break_contact"
        or (intended.get("operation") == "transfer_support" and success)
    ):
        expected_release.add(intended["finger"])
    for f, s in after.fingers.items():
        old = before.fingers[f].contact_state
        if s.contact_state != old:
            changes.append({"finger": f.value, "from": old.value, "to": s.contact_state.value})
        if (
            old != ContactMode.FREE
            and s.contact_state == ContactMode.FREE
            and f.value not in expected_release
        ):
            lost.append(f)
        if s.contact_state == ContactMode.SLIDING:
            slipped.append(f)
    hints = []
    if lost:
        hints.append(
            "Establish replacement support before releasing or reassigning the lost supporting finger"
        )
    if slipped:
        hints.append(
            "Reduce motion increment or change supporting contacts; do not increase force beyond the configured cap"
        )
    if not feasible:
        hints.append(
            "Expand this action into contact establishment, support verification, and smaller object motions"
        )
    if code == FailureCode.TIMEOUT:
        hints.append(
            "Inspect contact-point error and measured force; choose another reachable surface patch or split the approach into smaller transitions"
        )
    physical_constraints = {
        FailureCode.INSUFFICIENT_SUPPORT,
        FailureCode.FORCE_LIMIT,
        FailureCode.COLLISION,
        FailureCode.OBJECT_PENETRATION,
        FailureCode.CONTACT_LOST,
        FailureCode.SLIP,
        FailureCode.DROPPED,
    }
    return ExecutionReport(
        success=success,
        feasible=feasible,
        failure_code=code,
        message=message,
        intended=intended or {},
        actual_transitions=changes,
        lost_contact=lost,
        slipped=slipped,
        constraint_violations=[message] if code in physical_constraints else [],
        control_steps=steps,
        elapsed_s=max(0.0, after.time_s - before.time_s),
        final_state=after,
        object_position_error_m=position_error,
        object_rotation_error_deg=rotation_error,
        repair_hints=hints,
    )
