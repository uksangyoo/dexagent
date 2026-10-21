"""Start the arm palm-up, so the cube can begin resting mid-palm."""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.workcell import hand_down, register

FLIP = np.diag([-1.0, -1.0, 1.0])
NAME = "ur5e_dg5fs_right_palmup_cube"


def palm_up_rotation(tilt):
    """Palm normal up, fingers raised by `tilt` (the palm slopes toward the wrist)."""
    return hand_down() @ FLIP @ Rotation.from_rotvec([0, tilt, 0]).as_matrix()


def register_palm_up(palm_xyz, tilt):
    """Register the palm-up embodiment. Returns (its name, palm rotation)."""
    R = palm_up_rotation(tilt)
    return register(NAME, palm_xyz, R, seed=(-2.0, -1.0, 1.7, -0.7, -3.6, 0.0)), R
