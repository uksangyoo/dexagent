"""Cable metrics require material-coordinate and tension measurements."""

from dataclasses import dataclass

import numpy as np

from .common import TaskVerdict


def wrapped_angle(centerline_fixture, fixture_radius_m, cable_radius_m, radial_tolerance_m=0.01):
    points = np.asarray(centerline_fixture, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("Expected ordered cable centerline samples")
    radius = np.linalg.norm(points[:, :2], axis=1)
    near = np.abs(radius - (fixture_radius_m + cable_radius_m)) <= radial_tolerance_m
    # Only contiguous surface-adjacent cable counts; separated arcs cannot be
    # joined through a distant free segment to manufacture a wrap.
    runs = []
    start = None
    for i, valid in enumerate(np.r_[near, False]):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            if i - start >= 3:
                angles = np.unwrap(np.arctan2(points[start:i, 1], points[start:i, 0]))
                runs.append(abs(float(angles[-1] - angles[0])))
            start = None
    return max(runs, default=0.0)


@dataclass(frozen=True)
class CableMeasurement:
    material_coordinate_at_grip_m: float
    tension_N: float
    wrap_angle_rad: float
    dropped: bool
    tension_calibrated: bool


class CableWrapping:
    def __init__(self, initial, feed_length_m, wrap_angle_rad, tension_range_N):
        if not initial.tension_calibrated:
            raise ValueError("Cable benchmark requires calibrated tension observations")
        self.initial = initial
        self.length = feed_length_m
        self.angle = wrap_angle_rad
        self.range = tension_range_N
        self.tension_violation = False
        self.dropped = False

    def update(self, m):
        if not m.tension_calibrated:
            raise ValueError("Tension estimate became invalid")
        self.tension_violation |= not self.range[0] <= m.tension_N <= self.range[1]
        self.dropped |= m.dropped
        feed = m.material_coordinate_at_grip_m - self.initial.material_coordinate_at_grip_m
        success = (
            feed >= self.length
            and m.wrap_angle_rad >= self.angle
            and not self.tension_violation
            and not self.dropped
        )
        return TaskVerdict(
            bool(success),
            bool(success or self.dropped or self.tension_violation),
            {
                "feed_length_m": feed,
                "wrap_angle_rad": m.wrap_angle_rad,
                "tension_violation": self.tension_violation,
            },
            "success"
            if success
            else "tension_violation"
            if self.tension_violation
            else "dropped"
            if self.dropped
            else "running",
        )
