"""Palm-up start for the baoding scene: the shared workcell built palm-up with a cupped hand."""
import numpy as np
from dexagent.rss.scene import table_scene
from dexagent.rss.workcell import register

NAME = "ur5e_dg5fs_right_palmup"


def palm_up(yaw=0.0, tilt=(0.0, 0.0)):
    """Palm face up (+x palm -> +z world), fingers toward `yaw`; `tilt` is a small rotation vector (world x, y)."""
    from scipy.spatial.transform import Rotation
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[0, s, c], [0, -c, s], [1, 0, 0.]])
    return Rotation.from_rotvec([tilt[0], tilt[1], 0]).as_matrix() @ R


def palm_up_scene(name, home_xyz, cup, yaw=0.0):
    return table_scene(name, robot=register(NAME, home_xyz, palm_up(yaw), cup))
