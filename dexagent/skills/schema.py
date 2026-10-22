"""On-disk demo schema for HOM skill collection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEMO_SCHEMA = "dexagent.skills.demo/v1"
CONTROL_HZ = 50.0

# Arrays written into demo.npz (time-major).
ARRAY_KEYS = (
    "t",
    "engaged",
    "pose_valid",
    "palm_xyz",
    "palm_quat_xyzw",
    "arm_q",
    "hand_q",
    "manus_q",
    "tracker_pose",
    "object_pose",
    "tip_xyz",
    "contact_force",
    "hom_sep",
    "hom_slide",
    "hom_roll",
    "hom_spin",
    "hom_dist",
)


def save_demo(directory: Path, meta: dict, arrays: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("schema", DEMO_SCHEMA)
    meta.setdefault("fps", CONTROL_HZ)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(directory / "demo.npz", **payload)
    return directory


def load_demo(directory: Path) -> tuple[dict, dict]:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "demo.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    return meta, arrays


def list_demos(root: Path, *, success_only: bool = True) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name == "failed":
            continue
        if not (p / "meta.json").is_file():
            continue
        meta = json.loads((p / "meta.json").read_text())
        if success_only and not meta.get("success", False):
            continue
        out.append(p)
    failed = root / "failed"
    if not success_only and failed.is_dir():
        out.extend(sorted(d for d in failed.iterdir() if (d / "meta.json").is_file()))
    return out
