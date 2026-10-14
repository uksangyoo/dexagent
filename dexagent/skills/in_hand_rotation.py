from .composition import HandOperation, workflow


def in_hand_rotate(action, establishment=()):
    operations = [
        HandOperation(
            "hand.assign_roles",
            {"roles": {f.value: r.value for f, r in action.finger_roles.items()}},
        )
    ]
    operations += [
        HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
        for r in establishment
    ]
    operations.append(HandOperation("hand.move_object", {"action": action.model_dump(mode="json")}))
    return workflow("in_hand_rotate", operations)


def in_hand_translate(action, establishment=()):
    if action.mode != "translate":
        raise ValueError("Expected a translation action")
    return in_hand_rotate(action, establishment)


def ratchet_tool(action, support_transfers, cycles):
    if cycles < 1:
        raise ValueError("cycles must be positive")
    ops = []
    for _ in range(cycles):
        ops.append(
            HandOperation(
                "hand.assign_roles",
                {"roles": {f.value: r.value for f, r in action.finger_roles.items()}},
            )
        )
        ops.append(HandOperation("hand.move_object", {"action": action.model_dump(mode="json")}))
        ops += [
            HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
            for r in support_transfers
        ]
    return workflow("ratchet_tool", ops)
