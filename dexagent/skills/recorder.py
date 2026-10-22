"""Buffer and flush teleop demo episodes to disk."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.skills.schema import CONTROL_HZ, save_demo


class DemoRecorder:
    def __init__(self, demos_dir: Path | str, *, skill_id: str = "cube_pinch", object_name: str = "cube"):
        self.demos_dir = Path(demos_dir)
        self.skill_id = skill_id
        self.object_name = object_name
        self.recording = False
        self._rows: list[dict] = []
        self._t0 = 0.0
        self._meta_extra: dict = {}

    def start(self, **meta_extra) -> None:
        self._rows = []
        self._t0 = time.time()
        self._meta_extra = dict(meta_extra)
        self.recording = True

    def cancel(self) -> None:
        self.recording = False
        self._rows = []

    def append(self, row: dict) -> None:
        if self.recording:
            self._rows.append(row)

    def stop_and_save(self, *, success: bool, metrics: dict | None = None) -> Path | None:
        if not self._rows:
            self.recording = False
            return None
        arrays = _stack_rows(self._rows)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        parent = self.demos_dir if success else self.demos_dir / "failed"
        out = parent / stamp
        meta = {
            "skill_id": self.skill_id,
            "object": self.object_name,
            "success": bool(success),
            "n_frames": len(self._rows),
            "fps": CONTROL_HZ,
            "wall_s": time.time() - self._t0,
            "metrics": metrics or {},
            **self._meta_extra,
        }
        save_demo(out, meta, arrays)
        self.recording = False
        self._rows = []
        return out


def _stack_rows(rows: list[dict]) -> dict:
    def stack(key, dtype=float):
        return np.asarray([r[key] for r in rows], dtype=dtype)

    palm_R = np.asarray([r["palm_R"] for r in rows], float)
    quat = Rotation.from_matrix(palm_R).as_quat()  # xyzw
    out = {
        "t": stack("t"),
        "engaged": stack("engaged", np.int8),
        "pose_valid": stack("pose_valid", np.int8),
        "palm_xyz": stack("palm_xyz"),
        "palm_quat_xyzw": quat,
        "arm_q": stack("arm_q"),
        "hand_q": stack("hand_q"),
        "manus_q": stack("manus_q"),
        "tracker_pose": stack("tracker_pose"),
        "object_pose": stack("object_pose"),
        "tip_xyz": stack("tip_xyz"),
        "contact_force": stack("contact_force"),
    }
    if "hom_sep" in rows[0]:
        out["hom_sep"] = stack("hom_sep")
        out["hom_slide"] = stack("hom_slide")
        out["hom_roll"] = stack("hom_roll")
        out["hom_spin"] = stack("hom_spin")
        out["hom_dist"] = stack("hom_dist")
    return out


def row_from_controller(
    controller,
    *,
    t: float,
    engaged: bool,
    pose_valid: bool,
    manus_q,
    tracker_pose,
    object_name: str = "cube",
    hom: dict | None = None,
) -> dict:
    """Build one recorder row from the live controller + teleop sample."""
    obj = np.asarray(controller.sim[object_name].pose, float)
    forces = controller.contact_forces(object_name)
    tips = controller.tips_world()
    row = {
        "t": float(t),
        "engaged": int(bool(engaged)),
        "pose_valid": int(bool(pose_valid)),
        "palm_xyz": np.asarray(controller.xyz, float).copy(),
        "palm_R": np.asarray(controller.R, float).copy(),
        "arm_q": np.asarray(controller.arm, float).copy(),
        "hand_q": np.asarray(controller.hand, float).copy(),
        "manus_q": np.asarray(manus_q, float).reshape(6).copy(),
        "tracker_pose": np.asarray(tracker_pose, float).reshape(6).copy(),
        "object_pose": obj.copy(),
        "tip_xyz": tips.copy(),
        "contact_force": forces.copy(),
    }
    if hom is not None:
        row.update(hom)
    return row
