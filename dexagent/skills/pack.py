"""Pack N successful demos into a mean / phase HOM skill reference.

    python -m dexagent.skills.pack tasks/hand_demos/cube_pinch --n 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexagent.hom.features import recompute_hom_arrays
from dexagent.skills.schema import list_demos, load_demo

ROOT = Path(__file__).resolve().parents[2]


def _lift_anchor(arrays: dict) -> int:
    """Frame index where object height first rises > 40 mm above start."""
    z = arrays["object_pose"][:, 2]
    z0 = float(z[0])
    hit = np.where(z - z0 > 0.04)[0]
    return int(hit[0]) if len(hit) else len(z) // 2


def _resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Resample a time-major trajectory to ``n`` frames (linear)."""
    traj = np.asarray(traj, float)
    if len(traj) == n:
        return traj.copy()
    if len(traj) == 1:
        return np.repeat(traj, n, axis=0)
    x = np.linspace(0.0, 1.0, len(traj))
    xi = np.linspace(0.0, 1.0, n)
    if traj.ndim == 1:
        return np.interp(xi, x, traj)
    out = np.zeros((n,) + traj.shape[1:], float)
    flat = traj.reshape(len(traj), -1)
    for j in range(flat.shape[1]):
        out.reshape(n, -1)[:, j] = np.interp(xi, x, flat[:, j])
    return out


def pack_skill(scene_dir: Path, *, n: int = 5, frames: int = 200, recompute_hom: bool = True) -> Path:
    scene_dir = Path(scene_dir)
    if scene_dir.is_file():
        scene_dir = scene_dir.parent
    demos_dir = scene_dir / "demos"
    paths = list_demos(demos_dir, success_only=True)[:n]
    if not paths:
        raise SystemExit(f"no successful demos in {demos_dir}")

    packed = []
    for p in paths:
        meta, arrays = load_demo(p)
        if recompute_hom or "hom_sep" not in arrays:
            half = meta.get("box_half", [0.02, 0.02, 0.02])
            arrays.update(
                recompute_hom_arrays(
                    arrays["tip_xyz"],
                    arrays["object_pose"],
                    arrays["t"],
                    box_half=half,
                )
            )
        anchor = _lift_anchor(arrays)
        # Align so lift onset is at 50% of the packed horizon.
        pre = max(1, anchor)
        post = max(1, len(arrays["t"]) - anchor)
        # Take full trajectory resampled.
        entry = {
            "path": str(p),
            "palm_xyz": _resample(arrays["palm_xyz"], frames),
            "hand_q": _resample(arrays["hand_q"], frames),
            "hom_sep": _resample(arrays["hom_sep"], frames),
            "hom_slide": _resample(arrays["hom_slide"], frames),
            "hom_roll": _resample(arrays["hom_roll"], frames),
            "hom_spin": _resample(arrays["hom_spin"], frames),
            "hom_dist": _resample(arrays["hom_dist"], frames),
            "object_pose": _resample(arrays["object_pose"], frames),
            "anchor_frac": float(anchor / max(len(arrays["t"]) - 1, 1)),
            "pre_frames": pre,
            "post_frames": post,
        }
        # Phase params from distances / sep around pinch.
        dist = arrays["hom_dist"][:, :2].mean(axis=1)
        sep = arrays["hom_sep"][:, :2].mean(axis=1)
        entry["phase"] = {
            "min_dist_mm": round(1000 * float(dist.min()), 2),
            "mean_close_v_sep": round(float(sep[sep < 0].mean()) if np.any(sep < 0) else 0.0, 4),
            "lift_anchor_frac": entry["anchor_frac"],
        }
        packed.append(entry)

    def mean_std(key):
        stack = np.stack([e[key] for e in packed], axis=0)
        return stack.mean(axis=0), stack.std(axis=0)

    skill = {
        "schema": "dexagent.skills.pack/v1",
        "skill_id": scene_dir.name,
        "n_demos": len(packed),
        "frames": frames,
        "demo_paths": [e["path"] for e in packed],
        "phase_params": [e["phase"] for e in packed],
        "exemplar": packed[0]["path"],
    }
    out_dir = demos_dir / "skill"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_out = {}
    for key in ("palm_xyz", "hand_q", "hom_sep", "hom_slide", "hom_roll", "hom_spin", "hom_dist", "object_pose"):
        mu, sd = mean_std(key)
        arrays_out[f"{key}_mean"] = mu
        arrays_out[f"{key}_std"] = sd
    # Exemplar (first successful demo) — preferred for open-loop replay; mean HOM is for policy priors.
    for key in ("palm_xyz", "hand_q", "hom_sep", "hom_slide", "hom_roll", "hom_spin", "hom_dist", "object_pose"):
        arrays_out[f"{key}_exemplar"] = packed[0][key]
    np.savez_compressed(out_dir / "skill.npz", **arrays_out)
    # JSON-serializable summary (phase only + shapes).
    skill["arrays"] = {k: list(v.shape) for k, v in arrays_out.items()}
    skill["phase_mean"] = {
        k: float(np.mean([p[k] for p in skill["phase_params"]]))
        for k in skill["phase_params"][0]
    }
    (out_dir / "skill.json").write_text(json.dumps(skill, indent=2))
    print(f"packed {len(packed)} demos → {out_dir / 'skill.json'}", flush=True)
    return out_dir / "skill.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--no-recompute-hom", action="store_true")
    a = ap.parse_args(argv)
    pack_skill(a.scene, n=a.n, frames=a.frames, recompute_hom=not a.no_recompute_hom)


if __name__ == "__main__":
    main()
