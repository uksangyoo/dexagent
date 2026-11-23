"""Objects from reference/hand_demos, at measured or stated nominal sizes."""
import numpy as np


def quat_from_axis(direction):
    """xyzw quaternion turning +z onto `direction` (a capsule's long axis)."""
    from scipy.spatial.transform import Rotation
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    axis = np.cross([0, 0, 1.], d)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return (0., 0., 0., 1.) if d[2] > 0 else (1., 0., 0., 0.)
    return tuple(Rotation.from_rotvec(axis / s * np.arctan2(s, d[2])).as_quat())


# BIC Round Stic, cap off: 7.5 mm barrel, 140 mm long, about 5 g (nominal, not weighed).
PEN_RADIUS, PEN_HALF, PEN_MASS = 0.00375, 0.0662, 0.005
PEN_GREY = (0.22, 0.22, 0.23)


def pen(name="pen", at=(0.55, 0.0, 0.754), direction=(1, 0, 0), color=PEN_GREY, friction=0.8):
    """The pen as a capsule along its own z (total length 2 * (PEN_HALF + PEN_RADIUS))."""
    import robosimstudio as rss
    return rss.capsule(name=name, radius=PEN_RADIUS, half_height=PEN_HALF, at=tuple(at),
                       quat=quat_from_axis(direction), mass=PEN_MASS, friction=friction, color=color)
