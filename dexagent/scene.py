"""The table cell every hand scene starts from: a wood table, the arm and hand, two cameras."""
from dataclasses import replace
import numpy as np
from .workcell import TABLE_Z, register

WOOD = (0.60, 0.55, 0.43)
# The work area: objects start around here (world frame, on the table top).
WORK = np.array([0.55, 0.0, TABLE_Z])


def physics(substeps=20):
    import robosimstudio as rss
    from robosimstudio.specs.arrangement import backend
    return replace(rss.MUJOCO_500HZ, substeps=substeps,
                   backends=(("*", backend("mujoco", nconmax=4096, njmax=8192, iterations=60, ls_iterations=30,
                                           enable_multiccd=True, cone="elliptic", impratio=10)),))


def table_scene(name, *, camera_eye=(0.11, -0.014, TABLE_Z + 0.39), look_at=(0.70, -0.014, TABLE_Z), fov_y=42.0,
                substeps=20, robot=None):
    """A scene with the table, the robot and `reference` (phone), `overview` and `closeup` cameras.

    The default `reference` camera is the phone in reference/hand_demos: about
    350 mm over the table, 390 mm behind the object, pitched 35 deg down, with
    a 42 deg vertical field of view (an iPhone 1x lens in 1080p video).
    `robot` is a registered embodiment name (default: `register()`, palm-down home).
    """
    import robosimstudio as rss
    from robosimstudio.specs import CameraSpec
    scene = rss.Scene(cell="void", name=name, physics=physics(substeps))
    scene.add(rss.slab(name="table", half=(1.3, 0.75, 0.02), at=(0.95, 0.0, TABLE_Z - 0.02), color=WOOD))
    scene.add(rss.robot(robot or register()))
    scene.add(CameraSpec(name="reference", width=1280, height=720, fov_y=fov_y, eye=tuple(camera_eye), look_at=tuple(look_at)))
    scene.add(CameraSpec(name="overview", width=1280, height=720, fov_y=50,
                         eye=(1.25, -0.75, TABLE_Z + 0.65), look_at=(0.45, -0.1, TABLE_Z + 0.1)))
    scene.add(CameraSpec(name="closeup", width=1280, height=720, fov_y=40,
                         eye=(0.85, -0.3, TABLE_Z + 0.3), look_at=(0.55, 0.0, TABLE_Z + 0.08)))
    return scene
