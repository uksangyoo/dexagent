from .composition import HandOperation, workflow


def feed_cable(action, transfers=()):
    if action.mode != "advance_cable":
        raise ValueError("Expected a cable-feed action")
    # Constructible grammar composition; rigid-only environments return
    # UNSUPPORTED rather than passing a fictitious rigid cable success.
    operations = [
        HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
        for r in transfers
    ]
    operations.append(HandOperation("hand.move_object", {"action": action.model_dump(mode="json")}))
    return workflow("feed_cable", operations)
