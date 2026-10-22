"""Online HOM feature logger and offline recompute from demo.npz."""
from __future__ import annotations

import numpy as np

from dexagent.hom.gcf import gcf_from_tip_and_box, relative_contact_twist

# Thumb + index are the pinch pair for cube_pinch.
PINCH_FINGERS = (0, 1)
N_FINGERS = 5


class HomFeatureLogger:
    """Track GCFs for pinch fingers vs a box object; emit HOM velocity rows."""

    def __init__(
        self,
        *,
        box_half=(0.02, 0.02, 0.02),
        fingers=PINCH_FINGERS,
        ema: float = 0.35,
    ):
        self.box_half = np.asarray(box_half, float)
        self.fingers = tuple(fingers)
        self.ema = float(ema)
        self._prev: dict[int, dict] = {}
        self._v_ema: dict[int, dict] = {}

    def reset(self) -> None:
        self._prev.clear()
        self._v_ema.clear()

    def update(self, tip_xyz, object_xyz, dt: float) -> dict:
        """Return recorder hom_* arrays shaped for all 5 fingers (unused = 0)."""
        tips = np.asarray(tip_xyz, float).reshape(5, 3)
        center = np.asarray(object_xyz, float).reshape(3)
        sep = np.zeros(N_FINGERS)
        slide = np.zeros((N_FINGERS, 2))
        roll = np.zeros((N_FINGERS, 2))
        spin = np.zeros(N_FINGERS)
        dist = np.zeros(N_FINGERS)
        for i in self.fingers:
            gcf = gcf_from_tip_and_box(tips[i], center, self.box_half)
            dist[i] = gcf["dist"]
            vel = relative_contact_twist(self._prev.get(i), gcf, dt)
            prev_ema = self._v_ema.get(i)
            if prev_ema is None:
                sm = vel
            else:
                a = self.ema
                sm = {
                    "v_sep": (1 - a) * prev_ema["v_sep"] + a * vel["v_sep"],
                    "v_slide": (1 - a) * prev_ema["v_slide"] + a * vel["v_slide"],
                    "v_roll": (1 - a) * prev_ema["v_roll"] + a * vel["v_roll"],
                    "v_spin": (1 - a) * prev_ema["v_spin"] + a * vel["v_spin"],
                }
            self._v_ema[i] = sm
            self._prev[i] = gcf
            sep[i] = sm["v_sep"]
            slide[i] = sm["v_slide"]
            roll[i] = sm["v_roll"]
            spin[i] = sm["v_spin"]
        return {
            "hom_sep": sep,
            "hom_slide": slide,
            "hom_roll": roll,
            "hom_spin": spin,
            "hom_dist": dist,
        }


def recompute_hom_arrays(
    tip_xyz: np.ndarray,
    object_pose: np.ndarray,
    t: np.ndarray,
    *,
    box_half=(0.02, 0.02, 0.02),
    fingers=PINCH_FINGERS,
    ema: float = 0.35,
) -> dict:
    """Offline recompute HOM features from logged tip / object trajectories."""
    logger = HomFeatureLogger(box_half=box_half, fingers=fingers, ema=ema)
    n = len(t)
    sep = np.zeros((n, N_FINGERS))
    slide = np.zeros((n, N_FINGERS, 2))
    roll = np.zeros((n, N_FINGERS, 2))
    spin = np.zeros((n, N_FINGERS))
    dist = np.zeros((n, N_FINGERS))
    for k in range(n):
        dt = float(t[k] - t[k - 1]) if k else 1.0 / 50.0
        row = logger.update(tip_xyz[k], object_pose[k, :3], dt)
        sep[k] = row["hom_sep"]
        slide[k] = row["hom_slide"]
        roll[k] = row["hom_roll"]
        spin[k] = row["hom_spin"]
        dist[k] = row["hom_dist"]
    return {
        "hom_sep": sep,
        "hom_slide": slide,
        "hom_roll": roll,
        "hom_spin": spin,
        "hom_dist": dist,
    }
