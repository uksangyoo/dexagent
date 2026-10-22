"""Self-contained Vive tracker reader via OpenVR / SteamVR (no rio-hw)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexagent.teleop._clutch_core import matrix_to_rotvec

try:
    import openvr
except ImportError:  # pragma: no cover
    openvr = None  # type: ignore


def _device_serial(vr_system: Any, index: int) -> str:
    try:
        return vr_system.getStringTrackedDeviceProperty(index, openvr.Prop_SerialNumber_String)
    except openvr.OpenVRError:
        return f"device-{index}"


def _find_trackers(vr_system: Any) -> list[int]:
    return [
        i
        for i in range(openvr.k_unMaxTrackedDeviceCount)
        if vr_system.isTrackedDeviceConnected(i)
        and vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker
    ]


def _pose_matrix(matrix: Any) -> np.ndarray:
    result = np.eye(4)
    for row in range(3):
        for col in range(4):
            result[row, col] = matrix[row][col]
    return result


class ViveReader:
    """Poll one Vive tracker pose from a running SteamVR session.

    Publishes SteamVR standing-frame poses as ``[x, y, z, rx, ry, rz]``
    (meters + axis-angle). No headset required (null HMD driver).
    """

    def __init__(self, *, serial: str | None = None):
        if openvr is None:
            raise RuntimeError("openvr is not installed. pip install openvr")
        self.serial = serial
        self._vr = None
        self._label: str | None = None
        self._last_pose = np.zeros(6, dtype=np.float64)
        self._last_valid = False

    def start(self) -> None:
        vr = openvr.init(openvr.VRApplication_Background)
        trackers = _find_trackers(vr)
        if not trackers:
            openvr.shutdown()
            raise RuntimeError(
                "No Vive trackers in OpenVR. Start SteamVR, power the tracker, and pair it."
            )
        available = [_device_serial(vr, i) for i in trackers]
        print(f"[Vive] Trackers: {', '.join(available)}", flush=True)
        if self.serial and self.serial not in available:
            openvr.shutdown()
            raise RuntimeError(f"Tracker {self.serial!r} not found. Available: {available}")
        label = self.serial or available[0]
        print(f"[Vive] Following {label}", flush=True)
        self._vr = vr
        self._label = label

    def close(self) -> None:
        self._vr = None
        self._label = None
        try:
            openvr.shutdown()
        except Exception:
            pass

    def read(self) -> tuple[np.ndarray, bool]:
        """Return ``(tracker_pose_6, pose_valid)``. Republishes last pose when invalid."""
        vr, label = self._vr, self._label
        if vr is None or label is None:
            return self._last_pose.copy(), False
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        for index in _find_trackers(vr):
            if _device_serial(vr, index) != label:
                continue
            pose = poses[index]
            if not pose.bPoseIsValid:
                break
            matrix = _pose_matrix(pose.mDeviceToAbsoluteTracking)
            tracker_pose = np.concatenate([matrix[:3, 3], matrix_to_rotvec(matrix[:3, :3])])
            self._last_pose = tracker_pose.astype(float)
            self._last_valid = True
            return self._last_pose.copy(), True
        return self._last_pose.copy(), False
