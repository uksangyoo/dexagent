from .composition import HandOperation, workflow


def finger_gait(transfers, motions=()):
    if any(r.operation != "transfer_support" for r in transfers):
        raise ValueError("Finger gait requires receiver-first support transfers")
    ops = []
    for i, transfer in enumerate(transfers):
        ops.append(
            HandOperation("hand.transition_contact", {"request": transfer.model_dump(mode="json")})
        )
        if i < len(motions):
            ops.append(
                HandOperation("hand.move_object", {"action": motions[i].model_dump(mode="json")})
            )
    return workflow("finger_gait", ops)
