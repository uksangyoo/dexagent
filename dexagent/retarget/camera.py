"""Reference-camera model matching ``dexagent.scene.table_scene`` defaults."""
from __future__ import annotations

import numpy as np

from dexagent.workcell import TABLE_Z

# Defaults mirrored from table_scene(...).
DEFAULT_EYE = np.array([0.11, -0.014, TABLE_Z + 0.39], float)
DEFAULT_LOOK = np.array([0.70, -0.014, TABLE_Z], float)
DEFAULT_FOV_Y = 42.0


def camera_basis(eye=DEFAULT_EYE, look_at=DEFAULT_LOOK):
    """Return ``(eye, R_wc)`` where columns of ``R_wc`` are camera x (right), y (down), z (forward)."""
    eye = np.asarray(eye, float).reshape(3)
    look_at = np.asarray(look_at, float).reshape(3)
    forward = look_at - eye
    forward = forward / np.linalg.norm(forward)
    # World up is +z; camera y points down.
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    R_wc = np.column_stack([right, down, forward])
    return eye, R_wc


def intrinsics(width: int, height: int, fov_y_deg: float = DEFAULT_FOV_Y):
    fy = 0.5 * height / np.tan(np.radians(fov_y_deg) * 0.5)
    fx = fy
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return fx, fy, cx, cy


def pixel_to_ray(u, v, width, height, eye=DEFAULT_EYE, look_at=DEFAULT_LOOK, fov_y_deg=DEFAULT_FOV_Y):
    """Camera-origin ray in world frame through pixel ``(u, v)`` (OpenCV: +u right, +v down)."""
    eye, R_wc = camera_basis(eye, look_at)
    fx, fy, cx, cy = intrinsics(width, height, fov_y_deg)
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.array([x, y, 1.0], float)
    d_cam = d_cam / np.linalg.norm(d_cam)
    d_world = R_wc @ d_cam
    return eye, d_world / np.linalg.norm(d_world)


def ray_plane(eye, direction, plane_z=TABLE_Z):
    """Intersect ray with horizontal plane ``z = plane_z``. Returns None if parallel/away."""
    dz = direction[2]
    if abs(dz) < 1e-9:
        return None
    t = (plane_z - eye[2]) / dz
    if t <= 0:
        return None
    return eye + t * direction


def pixel_on_table(u, v, width, height, plane_z=TABLE_Z, **cam):
    eye, direction = pixel_to_ray(u, v, width, height, **cam)
    return ray_plane(eye, direction, plane_z=plane_z)


def palm_frame_from_tips(wrist, middle_mcp, index_tip, pinky_mcp=None):
    """Build a right-handed palm frame from MediaPipe-like landmarks (world).

    +z along fingers (wrist → middle MCP), +x out of the palm (approx), +y toward thumb.
    """
    wrist = np.asarray(wrist, float)
    middle = np.asarray(middle_mcp, float)
    index = np.asarray(index_tip, float)
    z = middle - wrist
    z = z / (np.linalg.norm(z) + 1e-9)
    # Across the palm: index to pinky if given, else index relative to middle.
    if pinky_mcp is not None:
        across = np.asarray(index, float) - np.asarray(pinky_mcp, float)
    else:
        across = index - middle
    y = across - z * np.dot(across, z)
    if np.linalg.norm(y) < 1e-8:
        y = np.array([0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return wrist, R
