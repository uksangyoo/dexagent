"""Map Manus 6-DoF finger openness onto the DG5F-S 20 joint angles.

Manus order (openness in ``[0, 1]``, 0=closed, 1=open)::

    [pinky, ring, middle, index, thumb_flex, thumb_rot]

DG5F-S order: thumb..little × 4 joints each (see ``dexagent.workcell.HAND``).
"""
from __future__ import annotations

import numpy as np

from dexagent.hand import CURL, OPEN, PINCH_AXIS, PINCH_CENTRE
from dexagent.workcell import Kinematics

# Manus channel → DG5F finger index (non-thumb).
_MANUS_FINGER = {0: 4, 1: 3, 2: 2, 3: 1}  # pinky, ring, middle, index


def _closed_pinch(half_width: float = 0.018) -> np.ndarray:
    """Thumb+index at a firm pinch; other fingers tucked."""
    kin = Kinematics()
    q = OPEN.copy()
    for i in (2, 3, 4):
        q[4 * i:4 * i + 4] = CURL if i != 4 else [0.3, CURL[1] - 0.2, CURL[2], CURL[3]]
    targets = {
        0: PINCH_CENTRE + half_width * PINCH_AXIS,
        1: PINCH_CENTRE - half_width * PINCH_AXIS,
    }
    for i, xyz in targets.items():
        q[4 * i:4 * i + 4], _ = kin.solve_finger(i, np.asarray(xyz, float), q[4 * i:4 * i + 4])
    return q


class ManusHandMapper:
    """Per-finger curl synergy: interpolate OPEN ↔ CLOSED from Manus openness."""

    def __init__(self, *, half_width: float = 0.018, tuck=(2, 3, 4)):
        self.open = OPEN.copy()
        self.half_width = float(half_width)
        self.tuck = tuple(tuck)
        self._closed = None

    @property
    def closed(self) -> np.ndarray:
        if self._closed is None:
            self._closed = _closed_pinch(self.half_width)
        return self._closed

    def map(self, manus_q) -> np.ndarray:
        """``manus_q`` shape (6,) openness → DG5F-S ``hand_q`` shape (20,)."""
        m = np.asarray(manus_q, float).reshape(6)
        m = np.clip(m, 0.0, 1.0)
        q = self.open.copy()
        closed = self.closed

        # Non-thumb fingers: openness blends open → closed curl.
        for manus_i, finger_i in _MANUS_FINGER.items():
            o = float(m[manus_i])
            sl = slice(4 * finger_i, 4 * finger_i + 4)
            q[sl] = o * self.open[sl] + (1.0 - o) * closed[sl]

        # Thumb: flex blends joints 1–3; rot blends joint 0 (swivel).
        thumb_flex = float(m[4])
        thumb_rot = float(m[5])
        q[0] = thumb_rot * self.open[0] + (1.0 - thumb_rot) * closed[0]
        q[1:4] = thumb_flex * self.open[1:4] + (1.0 - thumb_flex) * closed[1:4]
        return q
