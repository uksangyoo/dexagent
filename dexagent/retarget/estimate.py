"""Naive offline pose estimation from a phone clip into tracks.json.

Hand: MediaPipe HandLandmarker (tasks API), landmarks lifted via the scene
``reference`` camera onto (offset) table planes. Object (cube): HSV purple blob
→ table plane. Intentionally crude — a baseline, not HaMeR/FoundationPose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dexagent.retarget.camera import (
    DEFAULT_EYE,
    DEFAULT_FOV_Y,
    DEFAULT_LOOK,
    pixel_on_table,
)
from dexagent.retarget.hand_pose import (
    MIDDLE_MCP,
    WRIST,
    anchor_wrist_xy,
    blend_yaw,
    hover_from_landmarks_uv,
    palm_from_wrist_xy,
    pinch_tips_world,
    retarget_hand,
    yaw_from_xy,
)
from dexagent.retarget.stabilize import smooth_track
from dexagent.retarget.tracks import Track, TrackFrame, save_tracks
from dexagent.scene import WORK
from dexagent.workcell import TABLE_Z, YAW

_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

_CUBE_HSV_LO = np.array([120, 40, 40], np.uint8)
_CUBE_HSV_HI = np.array([170, 255, 255], np.uint8)


def _read_frames(clip: Path, max_frames: int | None = None, stride: int = 1):
    """Return ``(rgb_frames, source_indices)`` where indices are 0-based frames in the clip."""
    import imageio.v3 as iio

    reader = iio.imiter(clip, plugin="FFMPEG")
    frames = []
    indices = []
    for i, frame in enumerate(reader):
        if i % stride:
            continue
        frames.append(np.asarray(frame))
        indices.append(i)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise RuntimeError(f"no frames read from {clip}")
    return frames, indices


def _fps(clip: Path, default: float = 30.0) -> float:
    import imageio.v3 as iio

    try:
        meta = iio.immeta(clip, plugin="FFMPEG")
        fps = meta.get("fps")
        if isinstance(fps, (int, float)) and fps > 1:
            return float(fps)
    except Exception:
        pass
    return default


def _cube_centroid_uv(bgr: np.ndarray):
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CUBE_HSV_LO, _CUBE_HSV_HI)
    mask = cv2.medianBlur(mask, 5)
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None
    return float(xs.mean()), float(ys.mean())


def _default_hand(anchor_xyz, yaw=YAW, half_width=0.028, hover=0.10):
    palm, R = palm_from_wrist_xy(anchor_xyz, yaw, hover=hover)
    tips = pinch_tips_world(palm, R, half_width)
    return palm, R, tips


def _waiting_hand(yaw=YAW):
    """Idle pose while the human hand is still off-camera (matches workcell home)."""
    return _default_hand(np.array([0.45, -0.30, TABLE_Z]), yaw, half_width=0.028, hover=0.30)


def _make_landmarker():
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    if not _MODEL.is_file():
        raise FileNotFoundError(
            f"MediaPipe hand model missing at {_MODEL}. "
            "Download hand_landmarker.task into dexagent/retarget/models/."
        )
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def estimate_clip(
    clip: Path | str,
    *,
    object_name: str = "cube",
    object_half: float = 0.02,
    object_xyz0=None,
    object_quat0=None,
    detect_color: bool | None = None,
    eye=DEFAULT_EYE,
    look_at=DEFAULT_LOOK,
    fov_y: float = DEFAULT_FOV_Y,
    stride: int = 2,
    max_frames: int | None = None,
    table_z: float = TABLE_Z,
) -> Track:
    """Estimate hand (+ optional colour-keyed object) tracks.

    Colour keying (purple HSV) is only for the foam cube. Other objects keep
    ``object_xyz0`` / ``object_quat0`` (or ``WORK``) every frame — hand-only
    motion with a static teleport pose until a better object tracker is wired.
    """
    if detect_color is None:
        detect_color = object_name == "cube"
    import cv2
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat

    clip = Path(clip)
    fps = _fps(clip)
    rgb_frames, source_indices = _read_frames(clip, max_frames=max_frames, stride=stride)
    height, width = rgb_frames[0].shape[:2]
    landmarker = _make_landmarker()

    frames: list[TrackFrame] = []
    if object_xyz0 is not None:
        last_xyz = np.asarray(object_xyz0, float).reshape(3).copy()
    else:
        last_xyz = WORK + np.array([0.0, 0.0, object_half])
    # Hold the object's start pose until the hand is seen, so the sim does not
    # animate the cube while the human hand is still off-screen.
    object_xyz_hold = last_xyz.copy()
    if object_quat0 is not None:
        last_quat = np.asarray(object_quat0, float).reshape(4).copy()
    else:
        last_quat = np.array([0.0, 0.0, np.sin(YAW / 2), np.cos(YAW / 2)], float)
    hand_hits = 0
    object_hits = 0
    palm_up_hits = 0
    last_hand = None  # (palm, R, tips, yaw, landmarks_uv, palm_up)
    yaw = float(YAW)
    seen_hand = False
    palm_up = False
    wrist_anchor = None  # first table-lifted wrist XY → treated as WORK

    try:
        for i, rgb in enumerate(rgb_frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Absolute clip time from the source frame index (not the strided loop index).
            t = float(source_indices[i]) / fps
            ts_ms = int(round(1000 * t))

            if detect_color:
                uv = _cube_centroid_uv(bgr)
                if uv is not None:
                    pt = pixel_on_table(
                        uv[0],
                        uv[1],
                        width,
                        height,
                        plane_z=table_z + object_half,
                        eye=eye,
                        look_at=look_at,
                        fov_y_deg=fov_y,
                    )
                    if pt is not None:
                        last_xyz = pt
                        object_hits += 1

            mp_image = MpImage(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, ts_ms)
            palm_xyz = palm_R = tip_xyz = None
            landmarks_uv = None
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                landmarks_uv = np.array([[lm.x, lm.y] for lm in lms], float).reshape(21, 2)
                wrist_uv = (lms[WRIST].x * width, lms[WRIST].y * height)
                mid_uv = (lms[MIDDLE_MCP].x * width, lms[MIDDLE_MCP].y * height)
                wrist_w = pixel_on_table(
                    wrist_uv[0], wrist_uv[1], width, height, plane_z=table_z,
                    eye=eye, look_at=look_at, fov_y_deg=fov_y,
                )
                mid_w = pixel_on_table(
                    mid_uv[0], mid_uv[1], width, height, plane_z=table_z,
                    eye=eye, look_at=look_at, fov_y_deg=fov_y,
                )
                if wrist_w is not None and mid_w is not None:
                    if wrist_anchor is None:
                        wrist_anchor = np.asarray(wrist_w[:2], float).copy()
                    wrist_xy = anchor_wrist_xy(wrist_w, wrist_anchor)
                    mid_xy = anchor_wrist_xy(mid_w, wrist_anchor)
                    yaw = blend_yaw(yaw, yaw_from_xy(wrist_xy, mid_xy, fallback=yaw), max_step=0.15, alpha=0.35)
                    hover = hover_from_landmarks_uv(landmarks_uv)
                    world_lms = None
                    if result.hand_world_landmarks:
                        wl = result.hand_world_landmarks[0]
                        world_lms = np.array([[lm.x, lm.y, lm.z] for lm in wl], float).reshape(21, 3)
                    handedness = None
                    if result.handedness:
                        handedness = result.handedness[0][0].category_name
                    prev_R = None if last_hand is None else last_hand[1]
                    prev_up = None if last_hand is None else last_hand[5]
                    if world_lms is not None:
                        palm_xyz, palm_R, tip_xyz, palm_up = retarget_hand(
                            world_lms,
                            wrist_xy,
                            yaw,
                            hover=hover,
                            landmarks_uv=landmarks_uv,
                            prev_R=prev_R,
                            prev_palm_up=prev_up,
                            eye=eye,
                            look_at=look_at,
                            handedness=handedness,
                        )
                    else:
                        # Fallback: image-space pinch only (no 3D finger shape).
                        palm_xyz, palm_R = palm_from_wrist_xy(wrist_xy, yaw, hover=hover)
                        tip_xyz = pinch_tips_world(palm_xyz, palm_R, 0.02)
                        palm_up = False
                    last_hand = (palm_xyz, palm_R, tip_xyz, yaw, landmarks_uv.copy(), palm_up)
                    hand_hits += 1
                    if palm_up:
                        palm_up_hits += 1
                    seen_hand = True

            if palm_xyz is None:
                kept_lm = landmarks_uv
                if last_hand is not None:
                    palm_xyz, palm_R, tip_xyz, yaw, lm_prev, palm_up = last_hand
                    landmarks_uv = kept_lm if kept_lm is not None else lm_prev
                else:
                    # Hand still off-camera: stay at home, do not hover over the object.
                    palm_xyz, palm_R, tip_xyz = _waiting_hand(yaw)
                    landmarks_uv = kept_lm
                    palm_up = False

            obj_xyz = last_xyz if seen_hand else object_xyz_hold
            frames.append(
                TrackFrame(
                    t=t,
                    palm_xyz=np.asarray(palm_xyz, float),
                    palm_R=np.asarray(palm_R, float),
                    tips=np.asarray(tip_xyz, float).reshape(5, 3),
                    objects={object_name: (np.asarray(obj_xyz, float), last_quat.copy())},
                    landmarks_uv=None if landmarks_uv is None else np.asarray(landmarks_uv, float).reshape(21, 2),
                )
            )
    finally:
        landmarker.close()

    # Drop lead-in where the hand is not yet fully detected (no landmarks).
    start = next((i for i, fr in enumerate(frames) if fr.landmarks_uv is not None), None)
    if start is None:
        raise RuntimeError(f"no hand detected in {clip}; cannot build a retarget track")
    trim_start_s = float(frames[start].t)
    trimmed = frames[start:]
    # Re-base so track t=0 is the first in-frame hand (sim/compare start together).
    trimmed = [
        TrackFrame(
            t=float(fr.t - trim_start_s),
            palm_xyz=fr.palm_xyz,
            palm_R=fr.palm_R,
            tips=fr.tips,
            objects=fr.objects,
            landmarks_uv=fr.landmarks_uv,
        )
        for fr in trimmed
    ]

    track = Track(
        frames=trimmed,
        fps=fps / stride,
        clip=str(clip),
        meta={
            "estimator": "mediapipe_world_landmarks+hsv_cube"
            if detect_color
            else "mediapipe_world_landmarks+static_object",
            "hand_hits": hand_hits,
            "palm_up_hits": palm_up_hits,
            "object_hits": object_hits,
            "object_name": object_name,
            "detect_color": detect_color,
            "n_frames": len(trimmed),
            "stride": stride,
            "image_size": [width, height],
            "clip_fps": fps,
            "source_frame0": int(source_indices[start]),
            "source_frame1": int(source_indices[-1]),
            "trim_start_s": trim_start_s,
            "trimmed_pre_hand_frames": int(start),
        },
    )
    print(
        f"trimmed {start} pre-hand frames; track starts at clip t={trim_start_s:.2f}s "
        f"({len(trimmed)} frames, {track.duration:.2f}s)",
        flush=True,
    )
    # Prefer stability: noisy palm twist was worse than under-tracking fingers.
    return smooth_track(
        track,
        alpha=0.40,
        max_palm_step_m=0.035,
        max_tip_step_m=0.04,
        max_obj_step_m=0.04,
    )


def estimate_and_save(clip, out: Path, **kwargs) -> Track:
    track = estimate_clip(clip, **kwargs)
    save_tracks(track, out)
    return track
