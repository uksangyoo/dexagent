"""Naive wrist + fingertip IK: map a TrackFrame onto UR5e + DG5F-S joint targets."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.hand import OPEN
from dexagent.workcell import Kinematics


class NaiveRetargeter:
    """Open-loop IK from world palm pose + five fingertip world positions."""

    def __init__(self, kin: Kinematics | None = None):
        self.kin = kin or Kinematics()
        self.arm = np.zeros(6)
        self.hand = OPEN.copy()
        # Seed arm near a typical workcell reach so the first solve converges.
        self.arm[:] = (0.0, -1.2, 1.5, -1.9, -1.57, 0.0)

    def seed_from_sim(self, arm: np.ndarray, hand: np.ndarray | None = None) -> None:
        self.arm = np.asarray(arm, float).copy()
        if hand is not None:
            self.hand = np.asarray(hand, float).copy()

    def solve(self, palm_xyz, palm_R, tips_world, *, arm_tol=0.012, finger_warn=0.015,
              max_palm_step=0.08, tip_box=((0.01, -0.09, 0.015), (0.13, 0.09, 0.18)),
              hand_blend: float = 0.75):
        """Return ``(arm_q, hand_q, info)``.

        Finger targets are expressed in the *commanded* palm frame after arm IK,
        clamped to a reachable pad box so noisy video tips cannot explode IK.
        Palm motion is rate-limited relative to the last commanded palm.
        """
        xyz = np.asarray(palm_xyz, float).reshape(3)
        R = np.asarray(palm_R, float).reshape(3, 3)
        tips = np.asarray(tips_world, float).reshape(5, 3)
        # Rate-limit palm translation from the last solved pose.
        prev_xyz, _ = self.kin.palm_pose(self.arm)
        delta = xyz - prev_xyz
        n = float(np.linalg.norm(delta))
        if n > max_palm_step:
            xyz = prev_xyz + delta * (max_palm_step / n)
        info = {"arm_ok": True, "finger_err_mm": []}
        try:
            self.arm = self.kin.solve_arm(xyz, R, self.arm, tol=arm_tol)
        except ValueError as exc:
            info["arm_ok"] = False
            info["arm_error"] = str(exc)
        P, Rm = self.kin.palm_pose(self.arm)
        lo, hi = np.asarray(tip_box[0], float), np.asarray(tip_box[1], float)
        q = self.hand.copy()
        for i in range(5):
            target = Rm.T @ (tips[i] - P)
            target = np.clip(target, lo, hi)
            q[4 * i : 4 * i + 4], err = self.kin.solve_finger(
                i, target, q[4 * i : 4 * i + 4]
            )
            info["finger_err_mm"].append(round(1000 * err, 2))
            if err > finger_warn:
                info.setdefault("warnings", []).append(
                    f"finger {i} IK residual {err * 1000:.1f} mm"
                )
        # Prefer the new finger solution; light blend only to damp IK chatter.
        a = float(np.clip(hand_blend, 0.0, 1.0))
        self.hand = (1.0 - a) * self.hand + a * q
        return self.arm.copy(), self.hand.copy(), info


def palm_R_from_quat_xyzw(quat_xyzw) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
