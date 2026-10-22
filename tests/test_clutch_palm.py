"""Clutch palm driver unit tests (no hardware / no sim)."""
import numpy as np

from dexagent.teleop.clutch import ClutchPalmDriver
from dexagent.teleop._clutch_core import matrix_to_rotvec


def test_engage_hold_disengage():
    d = ClutchPalmDriver(world_frame_tracker=True)
    xyz0 = np.array([0.55, 0.0, 0.85])
    R0 = np.eye(3)
    d.hold(xyz0, R0)
    assert not d.engaged
    tracker0 = np.concatenate([xyz0, matrix_to_rotvec(R0)])
    d.engage(tracker0, xyz0, R0)
    assert d.engaged
    # Move tracker +2 cm in x → palm should follow.
    tracker1 = tracker0.copy()
    tracker1[0] += 0.02
    xyz, R = d.step(tracker1, 0.1)
    assert xyz[0] > xyz0[0]
    d.disengage()
    xyz2, _ = d.step(tracker1, 0.1)
    assert np.allclose(xyz2, xyz)
