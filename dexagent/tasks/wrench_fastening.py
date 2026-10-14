"""Measured milestone predicate for a wrench/fastener physics adapter."""

from dataclasses import dataclass

from .common import TaskVerdict


@dataclass(frozen=True)
class WrenchMeasurement:
    time_s: float
    lift_m: float
    in_hand_rotation_error_deg: float
    wrench_nut_contact: bool
    seating_error_m: float
    axis_error_deg: float
    nut_angle_rad: float  # Unwrapped, positive in tightening direction.
    thread_advance_m: float
    hand_contact_count: int
    dropped: bool


class WrenchFastening:
    def __init__(self, initial, cycles=3, angle_per_cycle_rad=0.3):
        self.initial = initial
        self.cycles = cycles
        self.angle = angle_per_cycle_rad
        self.picked = False
        self.reoriented = False
        self.engaged = False
        self.turn_start = initial.nut_angle_rad
        self.completed = 0
        self.must_disengage = False
        self.dropped = False

    def update(self, m):
        self.dropped |= m.dropped
        self.picked |= m.lift_m > 0.03 and m.hand_contact_count >= 2
        self.reoriented |= self.picked and m.in_hand_rotation_error_deg <= 3.0
        engaged = (
            self.reoriented
            and m.wrench_nut_contact
            and m.seating_error_m <= 0.003
            and m.axis_error_deg <= 5.0
        )
        if self.must_disengage:
            if not engaged:
                self.must_disengage = False
                self.engaged = False
        elif engaged:
            if not self.engaged:
                self.turn_start = m.nut_angle_rad
            if m.nut_angle_rad - self.turn_start >= self.angle:
                self.completed += 1
                self.must_disengage = True
            self.engaged = True
        else:
            self.engaged = False
        success = (
            self.completed >= self.cycles
            and m.thread_advance_m > self.initial.thread_advance_m
            and not self.dropped
        )
        return TaskVerdict(
            bool(success),
            bool(success or self.dropped),
            {
                "picked": self.picked,
                "reoriented": self.reoriented,
                "tightening_cycles": self.completed,
                "thread_advance_m": m.thread_advance_m,
            },
            "success" if success else "dropped" if self.dropped else "running",
        )
