"""Track schema: per-frame palm, fingertips, and object poses in the work-surface frame.

Units are meters. Quaternions are ``(qx, qy, qz, qw)``. Palm rotation is stored as
a 3x3 row-major matrix (same convention as ``Controller`` / scipy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_HZ = 50.0


@dataclass
class TrackFrame:
    t: float
    palm_xyz: np.ndarray
    palm_R: np.ndarray
    tips: np.ndarray  # (5, 3) world positions, thumb..little
    objects: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # objects[name] = (xyz (3,), quat_xyzw (4,))
    landmarks_uv: np.ndarray | None = None  # (21, 2) MediaPipe normalized [0,1]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "t": float(self.t),
            "palm": {
                "xyz": np.asarray(self.palm_xyz, float).reshape(3).tolist(),
                "R": np.asarray(self.palm_R, float).reshape(3, 3).tolist(),
            },
            "tips": np.asarray(self.tips, float).reshape(5, 3).tolist(),
            "objects": {
                name: {
                    "xyz": np.asarray(xyz, float).reshape(3).tolist(),
                    "quat_xyzw": np.asarray(quat, float).reshape(4).tolist(),
                }
                for name, (xyz, quat) in self.objects.items()
            },
        }
        if self.landmarks_uv is not None:
            out["landmarks_uv"] = np.asarray(self.landmarks_uv, float).reshape(21, 2).tolist()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackFrame:
        palm = d["palm"]
        objects = {}
        for name, od in (d.get("objects") or {}).items():
            objects[name] = (
                np.asarray(od["xyz"], float).reshape(3),
                np.asarray(od["quat_xyzw"], float).reshape(4),
            )
        lm = d.get("landmarks_uv")
        return cls(
            t=float(d["t"]),
            palm_xyz=np.asarray(palm["xyz"], float).reshape(3),
            palm_R=np.asarray(palm["R"], float).reshape(3, 3),
            tips=np.asarray(d["tips"], float).reshape(5, 3),
            objects=objects,
            landmarks_uv=None if lm is None else np.asarray(lm, float).reshape(21, 2),
        )


@dataclass
class Track:
    frames: list[TrackFrame]
    fps: float = CONTROL_HZ
    clip: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return float(self.frames[-1].t - self.frames[0].t)


def with_speed(track: Track, speed: float) -> Track:
    """Return a copy whose timeline is stretched by ``1/speed`` (e.g. 0.25 → 4× slower).

    Pose samples are unchanged; only ``t`` is scaled so replay gets more control
    steps for the same motion. ``speed`` is relative to the source clip (1.0 =
    realtime). Tracks on disk stay realtime; call this at play time.
    """
    speed = float(speed)
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")
    if abs(speed - 1.0) < 1e-12:
        return track
    scale = 1.0 / speed
    frames = [
        TrackFrame(
            t=float(fr.t) * scale,
            palm_xyz=fr.palm_xyz,
            palm_R=fr.palm_R,
            tips=fr.tips,
            objects=fr.objects,
            landmarks_uv=fr.landmarks_uv,
        )
        for fr in track.frames
    ]
    meta = dict(track.meta)
    meta["speed"] = speed
    meta["realtime_duration_s"] = float(track.duration)
    return Track(frames=frames, fps=track.fps / scale, clip=track.clip, meta=meta)


def validate_track(track: Track) -> None:
    if not track.frames:
        raise ValueError("track has no frames")
    if track.fps <= 0:
        raise ValueError(f"fps must be positive, got {track.fps}")
    t_prev = -np.inf
    for i, fr in enumerate(track.frames):
        if fr.t < t_prev:
            raise ValueError(f"frame {i}: time goes backwards ({fr.t} < {t_prev})")
        t_prev = fr.t
        if fr.palm_xyz.shape != (3,):
            raise ValueError(f"frame {i}: palm_xyz must be shape (3,)")
        if fr.palm_R.shape != (3, 3):
            raise ValueError(f"frame {i}: palm_R must be shape (3, 3)")
        if fr.tips.shape != (5, 3):
            raise ValueError(f"frame {i}: tips must be shape (5, 3)")
        if fr.landmarks_uv is not None and fr.landmarks_uv.shape != (21, 2):
            raise ValueError(f"frame {i}: landmarks_uv must be shape (21, 2)")
        det = float(np.linalg.det(fr.palm_R))
        if abs(det - 1.0) > 0.15:
            raise ValueError(f"frame {i}: palm_R det={det:.3f}, expected ~1")


def save_tracks(track: Track, path: Path | str) -> Path:
    validate_track(track)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dexagent.retarget.tracks/v1",
        "fps": float(track.fps),
        "frame_count": len(track.frames),
        "clip": track.clip,
        "meta": track.meta,
        "frames": [fr.to_dict() for fr in track.frames],
    }
    path.write_text(json.dumps(payload))
    return path


def load_tracks(path: Path | str) -> Track:
    path = Path(path)
    data = json.loads(path.read_text())
    schema = data.get("schema", "")
    if schema and schema != "dexagent.retarget.tracks/v1":
        raise ValueError(f"unsupported track schema {schema!r} in {path}")
    frames = [TrackFrame.from_dict(fr) for fr in data["frames"]]
    track = Track(
        frames=frames,
        fps=float(data.get("fps", CONTROL_HZ)),
        clip=data.get("clip"),
        meta=dict(data.get("meta") or {}),
    )
    validate_track(track)
    return track


def _slerp_R(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R0, R1]))
    return slerp(float(np.clip(u, 0.0, 1.0))).as_matrix()


def resample_tracks(track: Track, hz: float = CONTROL_HZ) -> Track:
    """Linear/slerp resample onto a uniform ``hz`` grid covering the track duration."""
    validate_track(track)
    if len(track.frames) == 1:
        return Track(frames=list(track.frames), fps=hz, clip=track.clip, meta=dict(track.meta))
    t0 = track.frames[0].t
    t1 = track.frames[-1].t
    n = max(1, int(round((t1 - t0) * hz)) + 1)
    times = t0 + np.arange(n) / hz
    times[-1] = t1
    src_t = np.array([fr.t for fr in track.frames], float)
    out: list[TrackFrame] = []
    obj_names = sorted({n for fr in track.frames for n in fr.objects})
    for t in times:
        i = int(np.searchsorted(src_t, t, side="right") - 1)
        i = int(np.clip(i, 0, len(track.frames) - 2))
        a, b = track.frames[i], track.frames[i + 1]
        span = max(b.t - a.t, 1e-9)
        u = float(np.clip((t - a.t) / span, 0.0, 1.0))
        objects = {}
        for name in obj_names:
            if name not in a.objects or name not in b.objects:
                continue
            xa, qa = a.objects[name]
            xb, qb = b.objects[name]
            from scipy.spatial.transform import Rotation, Slerp

            xyz = (1 - u) * xa + u * xb
            quat = Slerp([0.0, 1.0], Rotation.from_quat([qa, qb]))(u).as_quat()
            objects[name] = (xyz, quat)
        # Nearest landmark UVs (do not blend — overlay should stay on image pixels).
        lm = a.landmarks_uv if u < 0.5 else b.landmarks_uv
        if lm is None:
            lm = a.landmarks_uv if a.landmarks_uv is not None else b.landmarks_uv
        out.append(
            TrackFrame(
                t=float(t),
                palm_xyz=(1 - u) * a.palm_xyz + u * b.palm_xyz,
                palm_R=_slerp_R(a.palm_R, b.palm_R, u),
                tips=(1 - u) * a.tips + u * b.tips,
                objects=objects,
                landmarks_uv=None if lm is None else lm.copy(),
            )
        )
    meta = dict(track.meta)
    meta["resampled_hz"] = hz
    return Track(frames=out, fps=hz, clip=track.clip, meta=meta)
