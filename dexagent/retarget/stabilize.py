"""Temporal cleanup for naive video tracks (rate limits + EMA / SLERP smoothing)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dexagent.retarget.tracks import Track, TrackFrame, validate_track


def _ema(prev, cur, alpha: float):
    return (1.0 - alpha) * prev + alpha * cur


def _slerp_R(R0, R1, alpha: float):
    return Slerp([0.0, 1.0], Rotation.from_matrix([R0, R1]))(float(alpha)).as_matrix()


def _clamp_step(prev, cur, max_step: float):
    d = cur - prev
    n = float(np.linalg.norm(d))
    if n <= max_step or n < 1e-12:
        return cur
    return prev + d * (max_step / n)


def smooth_track(
    track: Track,
    *,
    alpha: float = 0.55,
    max_palm_step_m: float = 0.06,
    max_tip_step_m: float = 0.07,
    max_obj_step_m: float = 0.06,
) -> Track:
    """Causal EMA + per-frame travel caps. Call after estimate, before replay."""
    validate_track(track)
    if len(track.frames) <= 1:
        return track

    out: list[TrackFrame] = []
    f0 = track.frames[0]
    palm = f0.palm_xyz.copy()
    R = f0.palm_R.copy()
    tips = f0.tips.copy()
    objects = {n: (xyz.copy(), quat.copy()) for n, (xyz, quat) in f0.objects.items()}
    lm0 = None if f0.landmarks_uv is None else f0.landmarks_uv.copy()
    out.append(
        TrackFrame(
            t=f0.t,
            palm_xyz=palm.copy(),
            palm_R=R.copy(),
            tips=tips.copy(),
            objects=dict(objects),
            landmarks_uv=lm0,
        )
    )

    for fr in track.frames[1:]:
        palm_t = _clamp_step(palm, fr.palm_xyz, max_palm_step_m)
        palm = _ema(palm, palm_t, alpha)
        R = _slerp_R(R, fr.palm_R, alpha)
        tips_t = tips.copy()
        for i in range(5):
            tips_t[i] = _clamp_step(tips[i], fr.tips[i], max_tip_step_m)
        tips = _ema(tips, tips_t, alpha)
        objs = {}
        for name, (xyz, quat) in fr.objects.items():
            if name in objects:
                prev_xyz, prev_q = objects[name]
                xyz_t = _clamp_step(prev_xyz, xyz, max_obj_step_m)
                xyz_s = _ema(prev_xyz, xyz_t, alpha)
                q_s = Slerp([0.0, 1.0], Rotation.from_quat([prev_q, quat]))(alpha).as_quat()
                objs[name] = (xyz_s, q_s)
            else:
                objs[name] = (np.asarray(xyz, float).copy(), np.asarray(quat, float).copy())
        objects = objs
        lm = fr.landmarks_uv
        if lm is None:
            lm = lm0
        else:
            lm0 = lm.copy()
        out.append(
            TrackFrame(
                t=fr.t,
                palm_xyz=palm.copy(),
                palm_R=R.copy(),
                tips=tips.copy(),
                objects={n: (xyz.copy(), quat.copy()) for n, (xyz, quat) in objects.items()},
                landmarks_uv=None if lm is None else np.asarray(lm, float).reshape(21, 2).copy(),
            )
        )

    meta = dict(track.meta)
    meta["smoothed"] = {
        "alpha": alpha,
        "max_palm_step_m": max_palm_step_m,
        "max_tip_step_m": max_tip_step_m,
        "max_obj_step_m": max_obj_step_m,
    }
    return Track(frames=out, fps=track.fps, clip=track.clip, meta=meta)
