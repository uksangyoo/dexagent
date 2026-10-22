"""Teleop input sources: self-contained Vive+Manus, or mock (no rio-hw)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexagent.scene import WORK
from dexagent.workcell import YAW, hand_down


@dataclass
class TeleopSample:
    tracker_pose: np.ndarray  # (6,) [x,y,z,rx,ry,rz]
    pose_valid: bool
    manus_q: np.ndarray  # (6,) openness
    timestamp: float


class MockSource:
    """Scripted fake tracker + Manus (wiring tests only)."""

    def __init__(self, *, period_s: float = 8.0, auto_keys: bool = False, hz: float = 50.0):
        self.period_s = period_s
        self.auto_keys = auto_keys
        self.hz = hz
        self.steps_per_ep = max(1, int(round(period_s * hz)))
        self._step = 0
        self.t0 = time.time()
        self._key_queue: list[str] = []
        self._last_phase = ""

    def start(self) -> None:
        self.t0 = time.time()
        self._step = 0
        self._key_queue = ["c"] if self.auto_keys else []
        self._last_phase = ""

    def close(self) -> None:
        pass

    def sample(self) -> TeleopSample:
        u = (self._step % self.steps_per_ep) / float(self.steps_per_ep)
        self._step += 1
        palm_xyz, palm_R, manus = _scripted(u)
        from dexagent.teleop._clutch_core import matrix_to_rotvec

        tracker = np.concatenate([palm_xyz, matrix_to_rotvec(palm_R)])
        return TeleopSample(tracker, True, manus, time.time())

    def pop_keys(self) -> list[str]:
        keys = list(self._key_queue)
        self._key_queue.clear()
        return keys


def _phase_name(u: float) -> str:
    if u < 0.15:
        return "approach"
    if u < 0.30:
        return "descend"
    if u < 0.42:
        return "pinch"
    if u < 0.62:
        return "lift"
    if u < 0.78:
        return "hold"
    if u < 0.90:
        return "retreat"
    return "done"


def _scripted(u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = WORK + np.array([0.0, 0.0, 0.02])
    R = hand_down(YAW)
    above = cube + np.array([0.0, 0.0, 0.10])
    at = cube + np.array([0.0, 0.0, 0.01])
    lifted = cube + np.array([0.0, 0.0, 0.14])
    from dexagent.hand import PINCH_CENTRE

    def palm_at(p):
        return p - R @ PINCH_CENTRE

    open_m = np.ones(6)
    closed_m = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.4])
    if u < 0.15:
        s = u / 0.15
        xyz = (1 - s) * palm_at(above + [0, 0.05, 0.05]) + s * palm_at(above)
        manus = open_m
    elif u < 0.30:
        s = (u - 0.15) / 0.15
        xyz = (1 - s) * palm_at(above) + s * palm_at(at)
        manus = open_m
    elif u < 0.42:
        s = (u - 0.30) / 0.12
        xyz = palm_at(at)
        manus = (1 - s) * open_m + s * closed_m
    elif u < 0.62:
        s = (u - 0.42) / 0.20
        xyz = (1 - s) * palm_at(at) + s * palm_at(lifted)
        manus = closed_m
    elif u < 0.78:
        yaw = YAW + 0.15 * math.sin(2 * math.pi * (u - 0.62) / 0.16)
        R = hand_down(yaw)
        return palm_at(lifted), R, closed_m
    elif u < 0.90:
        s = (u - 0.78) / 0.12
        xyz = (1 - s) * palm_at(lifted) + s * palm_at(above)
        manus = (1 - s) * closed_m + s * open_m
    else:
        xyz = palm_at(above)
        manus = open_m
    return xyz, R, manus


class HardwareSource:
    """Self-contained Vive tracker + Manus glove (openvr + manus_glove only)."""

    def __init__(
        self,
        *,
        manus_cal: str | Path | None = None,
        vive_serial: str | None = None,
        **_ignored,
    ):
        if manus_cal is None:
            for cand in (Path("manus_glove_cal.json"), Path(__file__).resolve().parents[2] / "manus_glove_cal.json"):
                if cand.is_file():
                    manus_cal = cand
                    break
        self.manus_cal = Path(manus_cal) if manus_cal else None
        self.vive_serial = vive_serial
        self._vive = None
        self._manus = None

    def start(self) -> None:
        from dexagent.teleop.manus import ManusReader
        from dexagent.teleop.vive import ViveReader

        self._vive = ViveReader(serial=self.vive_serial)
        self._manus = ManusReader(calibration_file=self.manus_cal)
        self._vive.start()
        self._manus.start()

    def close(self) -> None:
        for dev in (self._manus, self._vive):
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
        self._manus = self._vive = None

    def sample(self) -> TeleopSample:
        pose, valid = self._vive.read()
        manus = self._manus.read()
        return TeleopSample(
            tracker_pose=np.asarray(pose, float),
            pose_valid=bool(valid),
            manus_q=np.asarray(manus, float),
            timestamp=time.time(),
        )


def make_source(kind: str, **kwargs):
    kind = kind.lower()
    if kind in ("mock", "synthetic"):
        return MockSource(**{k: v for k, v in kwargs.items() if k in ("period_s", "auto_keys", "hz")})
    if kind in ("hw", "hardware", "vive", "glove"):
        return HardwareSource(
            **{k: v for k, v in kwargs.items() if k in ("manus_cal", "vive_serial")}
        )
    raise ValueError(f"unknown teleop source {kind!r}")
