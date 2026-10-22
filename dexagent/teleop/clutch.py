"""Clutch-based Vive tracker → palm SE(3) driver for the table workcell.

Self-contained: uses ``dexagent.teleop._clutch_core`` only (no rio-hw).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dexagent.teleop._clutch_core import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    pose_to_matrix,
    yaw_rotation,
)
from dexagent.workcell import TABLE_Z


class TableWorkspace:
    """Axis-aligned box around the table work area (world frame)."""

    def __init__(
        self,
        x=(0.38, 0.78),
        y=(-0.28, 0.28),
        z=(TABLE_Z + 0.02, TABLE_Z + 0.40),
    ):
        self.x = tuple(x)
        self.y = tuple(y)
        self.z = tuple(z)

    def clamp(self, position: np.ndarray) -> np.ndarray:
        p = np.asarray(position, float).copy()
        p[0] = float(np.clip(p[0], self.x[0], self.x[1]))
        p[1] = float(np.clip(p[1], self.y[0], self.y[1]))
        p[2] = float(np.clip(p[2], self.z[0], self.z[1]))
        return p


class _NoClampWorkspace:
    def clamp(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(position, float).copy()


class ClutchPalmDriver:
    """Map Vive tracker motion (while engaged) onto a commanded palm pose."""

    def __init__(
        self,
        *,
        yaw_cal_path: Path | str | None = None,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        max_pos_speed: float = 0.35,
        max_rot_speed: float = 1.0,
        orientation_enabled: bool = True,
        workspace: TableWorkspace | None = None,
        world_frame_tracker: bool = False,
    ):
        yaw_deg = 0.0
        if world_frame_tracker:
            axis_map = np.eye(3)
        else:
            if yaw_cal_path is not None:
                path = Path(yaw_cal_path)
                if path.is_file():
                    yaw_deg = float(json.loads(path.read_text()).get("yaw_deg", 0.0))
            axis_map = yaw_rotation(math.radians(yaw_deg)) @ AXIS_MAP_STEAMVR_TO_ROBOT
        self.retargeter = ClutchRetargeter(
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            axis_map=axis_map,
            orientation_enabled=orientation_enabled,
        )
        self.retargeter.workspace = _NoClampWorkspace()
        self.workspace = workspace or TableWorkspace()
        self.yaw_deg = yaw_deg
        self._last_xyz = np.zeros(3)
        self._last_R = np.eye(3)

    @property
    def engaged(self) -> bool:
        return bool(self.retargeter.engaged)

    def hold(self, xyz, R) -> None:
        xyz = np.asarray(xyz, float)
        R = np.asarray(R, float)
        self.retargeter.hold(xyz, R)
        self._last_xyz = xyz.copy()
        self._last_R = R.copy()

    def engage(self, tracker_pose_6, palm_xyz, palm_R) -> None:
        self.retargeter.engage(pose_to_matrix(tracker_pose_6), palm_xyz, palm_R)
        self._last_xyz = np.asarray(palm_xyz, float).copy()
        self._last_R = np.asarray(palm_R, float).copy()

    def disengage(self) -> None:
        self.retargeter.disengage()

    def step(self, tracker_pose_6, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.engaged:
            return self._last_xyz.copy(), self._last_R.copy()
        xyz, R = self.retargeter.target(pose_to_matrix(tracker_pose_6), dt)
        xyz = self.workspace.clamp(xyz)
        self.retargeter.hold(xyz, R)
        self._last_xyz = xyz.copy()
        self._last_R = R.copy()
        return xyz, R
