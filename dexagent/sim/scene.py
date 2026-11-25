"""The table cell every hand scene starts from: a wood table, the arm and hand, four cameras."""
from dataclasses import replace
import numpy as np
from .workcell import TABLE_Z, register

WOOD = (0.60, 0.55, 0.43)
# The work area: objects start around here (world frame, on the table top).
WORK = np.array([0.55, 0.0, TABLE_Z])
# Eye-in-hand pose in the palm frame (camera +z forward, +y down).
# Mounted on the thumb side of the palm, looking across the grasp (-y) with the table
# toward the bottom of the image (+x). xform = (px, py, pz, qx, qy, qz, qw).
WRIST_XFORM = (0.03, 0.12, 0.08, 0.5, 0.5, -0.5, 0.5)


def _birch(size=1024):
    """Pale birch laminate as in reference/hand_demos: near-straight, low-contrast
    grain, streaky along the grain, a few soft figure bands. The grain runs along
    the table's x axis, away from the phone camera as in the clips."""
    from scipy.ndimage import gaussian_filter
    u = np.arange(size)[None, :] / size

    def noise(sigma, seed):
        n = gaussian_filter(np.random.default_rng(seed).standard_normal((size, size)), sigma, mode="wrap")
        return n / (n.std() + 1e-9)

    phase = u + 0.006 * noise(size / 4, 1) + 0.0015 * noise(size / 25, 2)
    lines = (0.5 + 0.5 * np.sin(2 * np.pi * 70 * phase + 2.0 * noise((size / 3, 4), 3))) ** 8
    bands = (0.5 + 0.5 * np.sin(2 * np.pi * 5 * phase + 2.5 * noise(size / 5, 4))) ** 3
    streaks = noise((size / 6, 1.5), 5)  # long along the grain (rows), thin across it
    shade = 1 - 0.035 * lines - 0.045 * bands + 0.012 * streaks
    base = np.array([0.86, 0.79, 0.66])
    return np.clip(base[None, None, :] * shade[..., None], 0, 1).transpose(1, 0, 2)


def physics(substeps=20):
    import robosimstudio as rss
    from robosimstudio.specs.arrangement import backend
    return replace(rss.MUJOCO_500HZ, substeps=substeps,
                   backends=(("*", backend("mujoco", nconmax=4096, njmax=8192, iterations=60, ls_iterations=30,
                                           enable_multiccd=True, cone="elliptic", impratio=10)),))


def table_scene(name, *, camera_eye=(0.11, -0.014, TABLE_Z + 0.39), look_at=(0.70, -0.014, TABLE_Z), fov_y=42.0,
                substeps=20, robot=None):
    """A scene with the table, the robot and `reference`, `overview`, `closeup` and `wrist` cameras.

    The default `reference` camera is the phone in reference/hand_demos: about
    350 mm over the table, 390 mm behind the object, pitched 35 deg down, with
    a 42 deg vertical field of view (an iPhone 1x lens in 1080p video).
    `wrist` is eye-in-hand on the palm (`parent_frame="work"`), looking out of the palm face.
    `robot` is a registered embodiment name (default: `register()`, palm-down home).
    """
    import robosimstudio as rss
    from robosimstudio.core.render.finish import TEXTURES, Finish, register_color_finish, register_texture
    from robosimstudio.specs import CameraSpec
    # Under the RTX path tracer the table gets the clips' pale birch grain.
    if "dexagent_birch" not in TEXTURES:
        register_texture("dexagent_birch", _birch, version=3)
    register_color_finish(WOOD, Finish(roughness=0.6, texture="dexagent_birch", uv_scale=(1 / 0.8, 1 / 0.8)))
    scene = rss.Scene(cell="void", name=name, physics=physics(substeps))
    scene.add(rss.slab(name="table", half=(1.3, 0.75, 0.02), at=(0.95, 0.0, TABLE_Z - 0.02), color=WOOD))
    scene.add(rss.robot(robot or register()))
    scene.add(CameraSpec(name="reference", width=1280, height=720, fov_y=fov_y, eye=tuple(camera_eye), look_at=tuple(look_at)))
    # Opposite side of the table from `closeup` (closeup is from -y; this is from +y).
    scene.add(CameraSpec(name="overview", width=1280, height=720, fov_y=45,
                         eye=(0.55, 0.70, TABLE_Z + 0.35), look_at=(0.55, -0.05, TABLE_Z + 0.06)))
    scene.add(CameraSpec(name="closeup", width=1280, height=720, fov_y=40,
                         eye=(0.85, -0.3, TABLE_Z + 0.3), look_at=(0.55, 0.0, TABLE_Z + 0.08)))
    scene.add(CameraSpec(name="wrist", width=1280, height=720, fov_y=70,
                         parent_frame="work", xform=WRIST_XFORM))
    return scene
