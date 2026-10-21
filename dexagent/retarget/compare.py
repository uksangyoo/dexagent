"""Temporally aligned human | sim comparison film with MediaPipe hand overlay.

    python -m dexagent.retarget.compare \\
      --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov \\
      --tracks tasks/hand_demos/cube_pinch/tracks/from_video.json \\
      --sim artifacts/rss/cube_pinch/retarget_from_video.mp4 \\
      --out artifacts/rss/cube_pinch/retarget_compare.mp4

Both panels start when the hand is first fully detected (``meta.trim_start_s``
in the source clip). Track time ``t=0`` is that moment; the human panel samples
the clip at ``trim_start_s + t``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dexagent.retarget.tracks import Track, TrackFrame, load_tracks

# MediaPipe hand connections (landmark index pairs).
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
_FINGERTIPS = (4, 8, 12, 16, 20)


def draw_hand(rgb: np.ndarray, landmarks_uv: np.ndarray | None, *, color=(0, 255, 128)) -> np.ndarray:
    """Draw MediaPipe skeleton on an RGB frame. ``landmarks_uv`` is (21, 2) in [0, 1]."""
    import cv2

    out = np.ascontiguousarray(rgb.copy())
    if landmarks_uv is None:
        return out
    lm = np.asarray(landmarks_uv, float).reshape(21, 2)
    h, w = out.shape[:2]
    pts = [(int(round(u * w)), int(round(v * h))) for u, v in lm]
    bgr = out[:, :, ::-1].copy()
    for a, b in _HAND_EDGES:
        cv2.line(bgr, pts[a], pts[b], color[::-1], 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 5 if i in _FINGERTIPS else 3
        cv2.circle(bgr, p, r, color[::-1], -1, cv2.LINE_AA)
    return bgr[:, :, ::-1]


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(rgb.copy())
    bgr = out[:, :, ::-1].copy()
    cv2.rectangle(bgr, (0, 0), (min(out.shape[1], 12 * len(text) + 24), 36), (0, 0, 0), -1)
    cv2.putText(bgr, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr[:, :, ::-1]


def _fit_height(rgb: np.ndarray, height: int) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if h == height:
        return rgb
    scale = height / h
    nw = max(1, int(round(w * scale)))
    return cv2.resize(rgb, (nw, height), interpolation=cv2.INTER_AREA)


def _clip_fps(clip: Path, default: float = 30.0) -> float:
    import imageio.v3 as iio

    try:
        meta = iio.immeta(clip, plugin="FFMPEG")
        fps = meta.get("fps")
        if isinstance(fps, (int, float)) and fps > 1:
            return float(fps)
    except Exception:
        pass
    return default


def _read_all_frames(path: Path) -> list:
    import imageio.v3 as iio

    frames = [np.asarray(f) for f in iio.imiter(path, plugin="FFMPEG")]
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def _nearest_track_frame(track: Track, t: float) -> TrackFrame:
    ts = np.array([fr.t for fr in track.frames], float)
    i = int(np.argmin(np.abs(ts - t)))
    return track.frames[i]


def _sim_index_for_time(t: float, t0: float, t1: float, n_sim: int) -> int:
    """Map absolute clip time ``t`` onto a sim-film frame (1:1 with track time)."""
    if n_sim <= 1:
        return 0
    if t <= t0:
        return 0
    if t >= t1:
        return n_sim - 1
    u = (t - t0) / max(t1 - t0, 1e-9)
    return int(np.clip(round(u * (n_sim - 1)), 0, n_sim - 1))


def build_compare(
    clip: Path | str,
    track: Track | Path | str,
    sim: Path | str | list | np.ndarray,
    *,
    out_path: Path | str,
    fps: float | None = None,
    panel_height: int = 720,
    speed: float | None = None,
) -> Path:
    """Write a side-by-side MP4: human (+ hand overlay) | sim, aligned on clip time.

    If the track was time-stretched with ``with_speed``, pass the same ``speed``
    (or rely on ``track.meta['speed']``) so the human panel stays in sync.
    """
    import imageio.v3 as iio

    clip = Path(clip)
    out_path = Path(out_path)
    if not isinstance(track, Track):
        track = load_tracks(track)
    if isinstance(sim, (str, Path)):
        sim_frames = _read_all_frames(Path(sim))
    else:
        sim_frames = list(sim)
        if not sim_frames:
            raise ValueError("sim film has no frames")

    clip_frames = _read_all_frames(clip)
    clip_fps = float(track.meta.get("clip_fps") or _clip_fps(clip))
    # Track t=0 is the first fully detected hand; map back into the source clip.
    clip_t0 = float(track.meta.get("trim_start_s", 0.0))
    play_speed = float(speed if speed is not None else track.meta.get("speed", 1.0))
    if play_speed <= 0.0:
        raise ValueError(f"speed must be positive, got {play_speed}")
    t0 = float(track.frames[0].t)
    t1 = float(track.frames[-1].t)
    track_duration = max(t1 - t0, 1e-6)

    out_fps = float(fps) if fps is not None else max(1.0, len(sim_frames) / track_duration)
    n = max(1, int(round(track_duration * out_fps)) + 1)
    times = t0 + np.arange(n) / out_fps
    times[-1] = t1

    panels = []
    for t in times:
        # Stretched track time → realtime clip time (speed=0.25 → human also looks 4× slow).
        abs_t = clip_t0 + float(t - t0) * play_speed
        ci = int(np.clip(round(abs_t * clip_fps), 0, len(clip_frames) - 1))
        si = _sim_index_for_time(float(t), t0, t1, len(sim_frames))
        fr = _nearest_track_frame(track, float(t))
        left = draw_hand(clip_frames[ci], fr.landmarks_uv)
        left = _label(_fit_height(left, panel_height), "human")
        right = _label(_fit_height(np.asarray(sim_frames[si]), panel_height), "sim")
        if left.shape[0] != right.shape[0]:
            h = min(left.shape[0], right.shape[0])
            left, right = _fit_height(left, h), _fit_height(right, h)
        panels.append(np.concatenate([left, right], axis=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(panels), fps=max(1, int(round(out_fps))))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, type=Path, help="original phone video")
    ap.add_argument("--tracks", required=True, type=Path, help="from_video tracks.json")
    ap.add_argument("--sim", required=True, type=Path, help="retarget_from_video.mp4")
    ap.add_argument("--out", required=True, type=Path, help="output compare mp4")
    ap.add_argument("--fps", type=float, default=None, help="compare fps (default: infer from sim/track)")
    ap.add_argument("--height", type=int, default=720, help="panel height in pixels")
    ap.add_argument("--speed", type=float, default=None, help="playback speed vs clip (default: track meta)")
    a = ap.parse_args(argv)
    path = build_compare(
        a.clip, a.tracks, a.sim, out_path=a.out, fps=a.fps, panel_height=a.height, speed=a.speed
    )
    print(f"compare {path}", flush=True)


if __name__ == "__main__":
    main()
