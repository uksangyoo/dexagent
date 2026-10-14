from .composition import HandOperation, workflow


def regrasp(transitions):
    return workflow(
        "regrasp",
        [
            HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
            for r in transitions
        ],
    )


def pinch(establishment):
    if len(establishment) < 2:
        raise ValueError("A pinch needs at least two contact establishments")
    if any(r.operation != "establish_contact" for r in establishment):
        raise ValueError("pinch expects contact establishment requests")
    return workflow(
        "pinch",
        [
            HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
            for r in establishment
        ],
    )
