"""Screw readiness and fastening require independent fixture/thread readouts."""

from dataclasses import dataclass

from .common import TaskVerdict


@dataclass(frozen=True)
class ScrewMeasurement:
    axis_error_deg: float
    tip_position_error_m: float
    hand_contact_count: int
    thread_engaged: bool
    axial_advance_m: float
    dropped: bool


class ScrewHandling:
    def __init__(self, required_advance_m=0.002):
        self.ready = False
        self.dropped = False
        self.required = required_advance_m

    def update(self, m):
        self.ready |= (
            m.axis_error_deg <= 3 and m.tip_position_error_m <= 0.002 and m.hand_contact_count >= 2
        )
        self.dropped |= m.dropped
        success = (
            self.ready
            and m.thread_engaged
            and m.axial_advance_m >= self.required
            and not self.dropped
        )
        return TaskVerdict(
            bool(success),
            bool(success or self.dropped),
            {"insertion_ready": self.ready, "axial_advance_m": m.axial_advance_m},
            "success" if success else "dropped" if self.dropped else "running",
        )
