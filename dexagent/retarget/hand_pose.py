"""Map MediaPipe hand landmarks onto the DG5F-S palm / fingertip targets.

Uses MediaPipe *world* landmarks (metres) for finger shape **and** palm
orientation (roll / pitch / yaw), mapped into the robot frame through the
reference camera. Wrist *position* still comes from a table-plane lift of the
image wrist, so the arm stays in a reachable workspace.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dexagent.hand import PINCH_CENTRE
from dexagent.retarget.camera import DEFAULT_EYE, DEFAULT_LOOK, camera_basis
from dexagent.workcell import TABLE_Z, YAW, hand_down, hand_up

# MediaPipe indices.
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# DG5F open-hand middle tip is roughly this far from the palm origin along +z.
_DG_MIDDLE_LEN = 0.11

# MediaPipe hand world landmarks: y-up (person), z toward the camera. Map into
# OpenCV camera coords (x right, y down, z forward) before applying R_wc.
_MP_TO_CAM = np.diag([1.0, -1.0, -1.0])


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def yaw_from_xy(wrist_xy, middle_xy, fallback=YAW) -> float:
    d = np.asarray(middle_xy, float)[:2] - np.asarray(wrist_xy, float)[:2]
    if float(np.linalg.norm(d)) < 1e-6:
        return float(fallback)
    return float(np.arctan2(d[1], d[0]))


def blend_yaw(prev: float, new: float, *, max_step: float = 0.25, alpha: float = 0.55) -> float:
    delta = _wrap_pi(new - prev)
    delta = float(np.clip(delta, -max_step, max_step))
    return _wrap_pi(prev + alpha * delta)


def clamp_palm_xy(palm_xyz, centre=None, radius=0.22, min_base_dist=0.48):
    """Keep the palm in a reachable band around WORK, not crumpled against the arm."""
    from dexagent.scene import WORK

    centre = np.asarray(centre if centre is not None else WORK[:2], float)
    p = np.asarray(palm_xyz, float).copy()
    d = p[:2] - centre
    n = float(np.linalg.norm(d))
    if n > radius:
        p[:2] = centre + d * (radius / n)
    # Phone→table lift often parks the hand too near the UR5e base.
    r = float(np.linalg.norm(p[:2]))
    if r < min_base_dist and r > 1e-6:
        p[:2] *= min_base_dist / r
    p[2] = float(np.clip(p[2], TABLE_Z + 0.04, TABLE_Z + 0.40))
    return p


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation onto SO(3)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, float).reshape(3, 3))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _mp_palm_basis(world_lms: np.ndarray) -> np.ndarray:
    """Columns are palm +x (out/flex), +y (toward thumb), +z (along fingers) in MP coords."""
    W = world_lms[WRIST]
    z = world_lms[MIDDLE_MCP] - W
    z = z / (np.linalg.norm(z) + 1e-9)
    y = world_lms[INDEX_MCP] - world_lms[PINKY_MCP]
    y = y - z * np.dot(y, z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _fix_palm_basis(R_mp: np.ndarray, world_lms: np.ndarray) -> np.ndarray:
    """Flip axes so +x is pad-side and +y is thumb-side; keep a right-handed frame."""
    R = np.asarray(R_mp, float).copy()
    W = world_lms[WRIST]
    tips = np.array([R.T @ (world_lms[i] - W) for i in TIP_IDS], float)
    if float(tips[:, 0].mean()) < 0.0:
        R[:, 0] *= -1.0
        tips[:, 0] *= -1.0
    if float(tips[0, 1]) < float(tips[4, 1]):
        R[:, 1] *= -1.0
    if np.linalg.det(R) < 0.0:
        R[:, 1] *= -1.0
    return _orthonormalize(R)


def _resolve_R_flip(R: np.ndarray, prev_R: np.ndarray | None) -> np.ndarray:
    """Pick among 180° axis flips the rotation closest to ``prev_R``."""
    R = _orthonormalize(R)
    if prev_R is None:
        return R
    prev = _orthonormalize(prev_R)
    candidates = [
        R,
        R @ np.diag([1.0, -1.0, -1.0]),
        R @ np.diag([-1.0, 1.0, -1.0]),
        R @ np.diag([-1.0, -1.0, 1.0]),
    ]
    best, best_ang = R, 1e9
    for C in candidates:
        C = _orthonormalize(C)
        if np.linalg.det(C) < 0:
            continue
        ang = float(Rotation.from_matrix(prev.T @ C).magnitude())
        if ang < best_ang:
            best, best_ang = C, ang
    return best


def mirror_world_landmarks_to_right(world_lms: np.ndarray) -> np.ndarray:
    """Reflect a left hand's world landmarks into right-hand geometry (flip MP +x)."""
    out = np.asarray(world_lms, float).reshape(21, 3).copy()
    out[:, 0] *= -1.0
    return out


def _palm_normal_world_z(
    world_lms: np.ndarray,
    *,
    eye=DEFAULT_EYE,
    look_at=DEFAULT_LOOK,
) -> float:
    """World +z component of palm +x (out of the palm face). >0 ≈ palm up."""
    world_lms = np.asarray(world_lms, float).reshape(21, 3)
    R_mp = _fix_palm_basis(_mp_palm_basis(world_lms), world_lms)
    _, R_wc = camera_basis(eye, look_at)
    R_abs = _orthonormalize(R_wc @ _MP_TO_CAM @ R_mp)
    return float(R_abs[2, 0])


def estimate_palm_up(
    world_lms: np.ndarray,
    *,
    prev_up: bool | None = None,
    eye=DEFAULT_EYE,
    look_at=DEFAULT_LOOK,
    enter: float = 0.20,
    exit: float = -0.10,
) -> bool:
    """Hysteresis palm-up detector from MediaPipe world landmarks.

    The MP→camera palm +x we build is inverted vs anatomy on these clips
    (palm-down pinch gives ``nz > 0``), so we use ``score = -nz`` (>0 ⇒ palm up).
    """
    nz = _palm_normal_world_z(world_lms, eye=eye, look_at=look_at)
    score = -nz
    if prev_up is None:
        return score > 0.0
    if prev_up:
        return score > exit
    return score > enter


def palm_R_from_world_landmarks(
    world_lms: np.ndarray,
    yaw: float,
    *,
    prev_R=None,
    palm_up: bool = False,
    eye=DEFAULT_EYE,
    look_at=DEFAULT_LOOK,
    # Absolute MP→world orientation is noisy; keep residuals tiny so a mostly
    # still hand (e.g. IMG_4776) does not thrash the robot via palm twist.
    max_twist: float = 0.20,
    max_pitch: float = 0.15,
    max_roll: float = 0.12,
    orient_alpha: float = 0.25,
) -> np.ndarray:
    """Palm rotation for the *right* DG5F-S.

    Base pose is ``hand_down(yaw)`` or ``hand_up(yaw)`` from a palm-facing
    detector. MediaPipe may add a *small* clamped twist/pitch/roll on top.
    """
    world_lms = np.asarray(world_lms, float).reshape(21, 3)
    base = hand_up(yaw) if palm_up else hand_down(yaw)
    R_mp = _fix_palm_basis(_mp_palm_basis(world_lms), world_lms)
    _, R_wc = camera_basis(eye, look_at)
    R_abs = _orthonormalize(R_wc @ _MP_TO_CAM @ R_mp)
    R_abs = _resolve_R_flip(R_abs, prev_R if prev_R is not None else base)

    # Residual pose in the base palm frame: zyx = twist(about fingers), pitch, roll.
    delta = Rotation.from_matrix(base.T @ R_abs)
    twist, pitch, roll = delta.as_euler("zyx")
    twist = float(np.clip(twist, -max_twist, max_twist))
    pitch = float(np.clip(pitch, -max_pitch, max_pitch))
    roll = float(np.clip(roll, -max_roll, max_roll))
    R = base @ Rotation.from_euler("zyx", [twist, pitch, roll]).as_matrix()
    if prev_R is not None:
        a = float(np.clip(orient_alpha, 0.0, 1.0))
        R = Slerp([0.0, 1.0], Rotation.from_matrix([prev_R, R]))(a).as_matrix()
    return _orthonormalize(R)


def tips_palm_from_world_landmarks(world_lms: np.ndarray) -> np.ndarray:
    """Five fingertip positions in our palm frame (metres), from MP world landmarks."""
    world_lms = np.asarray(world_lms, float).reshape(21, 3)
    R_mp = _fix_palm_basis(_mp_palm_basis(world_lms), world_lms)
    W = world_lms[WRIST]
    tips_w = np.array([R_mp.T @ (world_lms[i] - W) for i in TIP_IDS], float)
    mcps_w = np.array([R_mp.T @ (world_lms[i] - W) for i in MCP_IDS], float)
    mid_len = float(np.linalg.norm(tips_w[2]))
    scale = _DG_MIDDLE_LEN / max(mid_len, 0.04)
    tips_w *= scale
    mcps_w *= scale
    mid_mcp = mcps_w[1]
    offset = np.array(
        [
            PINCH_CENTRE[0] - mid_mcp[0],
            0.0 - mid_mcp[1],
            PINCH_CENTRE[2] - mid_mcp[2],
        ],
        float,
    )
    tips = tips_w + offset
    tips[:, 0] = np.clip(tips[:, 0], 0.01, 0.13)
    tips[:, 1] = np.clip(tips[:, 1], -0.09, 0.09)
    tips[:, 2] = np.clip(tips[:, 2], 0.02, 0.18)
    return tips


def hover_from_landmarks_uv(landmarks_uv, base: float = 0.10) -> float:
    """Table hover from image-space hand size / wrist height (image z is often flat)."""
    lm = np.asarray(landmarks_uv, float).reshape(21, 2)
    span = float(np.linalg.norm(lm[MIDDLE_TIP] - lm[WRIST]))
    wrist_v = float(lm[WRIST, 1])
    # Mild lift cues only — UV jitter was pumping height on quiet clips.
    hover = base + 0.25 * (span - 0.40) + 0.08 * (0.90 - wrist_v)
    return float(np.clip(hover, 0.06, 0.28))


def amplify_wrist_xy(wrist_xy, *, centre=None, gain: float = 1.0, radius: float = 0.22):
    """Optional table-plane gain around the work centre (default: identity)."""
    from dexagent.scene import WORK

    centre = np.asarray(centre if centre is not None else WORK[:2], float)
    w = np.asarray(wrist_xy, float)[:2]
    out = centre + gain * (w - centre)
    d = out - centre
    n = float(np.linalg.norm(d))
    if n > radius:
        out = centre + d * (radius / n)
    return out


def anchor_wrist_xy(wrist_xy, anchor_xy, *, centre=None, radius: float = 0.22):
    """Map a table-lifted wrist into the workcell via motion relative to the first detection.

    Absolute phone→table XY is badly biased (often ~0.3 m from the arm base). Treat
    the first seen wrist as ``WORK`` and only apply deltas so the arm stays reachable.
    """
    from dexagent.scene import WORK

    centre = np.asarray(centre if centre is not None else WORK[:2], float)
    w = np.asarray(wrist_xy, float)[:2]
    a = np.asarray(anchor_xy, float)[:2]
    out = centre + (w - a)
    d = out - centre
    n = float(np.linalg.norm(d))
    if n > radius:
        out = centre + d * (radius / n)
    return out


def palm_from_wrist_xy(wrist_xy, yaw: float, hover: float = 0.10, R=None):
    """Palm pose with pinch centre above ``wrist_xy`` at height ``hover``."""
    R = hand_down(yaw) if R is None else _orthonormalize(R)
    pinch = np.array([float(wrist_xy[0]), float(wrist_xy[1]), TABLE_Z + hover], float)
    palm = pinch - R @ PINCH_CENTRE
    return palm, R


def retarget_hand(
    world_lms,
    wrist_xy,
    yaw: float,
    *,
    hover: float | None = None,
    landmarks_uv=None,
    prev_R=None,
    prev_palm_up: bool | None = None,
    eye=DEFAULT_EYE,
    look_at=DEFAULT_LOOK,
    handedness: str | None = None,
):
    """Return ``(palm_xyz, palm_R, tips_world, palm_up)`` from MP world landmarks + table wrist.

    The workcell hand is a right DG5F-S. Left MediaPipe detections are mirrored
    into right-hand geometry before IK targets are built. Palm-up (back of hand
    toward the table) uses ``hand_up`` instead of ``hand_down``.
    """
    world_lms = np.asarray(world_lms, float).reshape(21, 3)
    if handedness and handedness.lower().startswith("left"):
        world_lms = mirror_world_landmarks_to_right(world_lms)
    tips_palm = tips_palm_from_world_landmarks(world_lms)
    palm_up = estimate_palm_up(
        world_lms, prev_up=prev_palm_up, eye=eye, look_at=look_at
    )
    if hover is None:
        hover = hover_from_landmarks_uv(landmarks_uv) if landmarks_uv is not None else 0.10
    # Palm-up needs a bit more clearance so the open palm sits above the table.
    if palm_up:
        hover = max(hover, 0.12)
    R = palm_R_from_world_landmarks(
        world_lms,
        yaw,
        prev_R=prev_R,
        palm_up=palm_up,
        eye=eye,
        look_at=look_at,
    )
    palm, R = palm_from_wrist_xy(wrist_xy, yaw, hover=hover, R=R)
    palm = clamp_palm_xy(palm)
    tips_world = np.array([palm + R @ t for t in tips_palm], float)
    return palm, R, tips_world, palm_up


def pinch_tips_world(palm_xyz, palm_R, half_width: float) -> np.ndarray:
    """Fallback when world landmarks are missing: simple thumb/index pinch."""
    from dexagent.hand import PINCH_AXIS

    half = float(np.clip(half_width, 0.008, 0.035))
    locals_ = [
        PINCH_CENTRE + half * PINCH_AXIS,
        PINCH_CENTRE - half * PINCH_AXIS,
        PINCH_CENTRE + np.array([0.0, -0.04, 0.02]),
        PINCH_CENTRE + np.array([0.0, -0.055, 0.015]),
        PINCH_CENTRE + np.array([0.0, -0.07, 0.01]),
    ]
    return np.array([palm_xyz + palm_R @ p for p in locals_], float)
