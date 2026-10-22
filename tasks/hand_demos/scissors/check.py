"""Scorer for scissors, independent of the policy. Reads simulator state only:
the scissors' pose (part A, frame at the pivot) and the hinge coordinate."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("scissors",)
TABLE_Z = 0.75
OPEN_DEG = 10.0      # blades count as open above this hinge angle
CLOSED_DEG = 3.0     # and as closed below this
CYCLES = 2
CLEAR = 0.020        # lowest point of the scissors this far above the table counts as in the air
REST = 0.015         # pivot at most this far above the table at the end
# Extreme points of the nominal scissors in their own frame (x along the blades, z up):
# blade tip, and the outer corners of the handle loop, for part A; part B is the
# same mirrored in y and turned by the hinge angle about z. Half thickness 5.5 mm.
_A = np.array([[0.096, 0.0, 0.0], [-0.0875, 0.0425, 0.0], [-0.0425, 0.0425, 0.0], [-0.0875, 0.0055, 0.0]])
_B = _A * [1, -1, 1]
_HALF_T = 0.0055


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        labels = list(sim.model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("scissors_hinge"))
        self.qi = int(sim.model.joint_q_start.numpy()[j])
        p, _ = self._pose()
        self.p0 = p
        self.max_lift = 0.0          # pivot rise above its start height
        self.max_clear = -1.0        # lowest point above the table
        self.air_s = 0.0
        self.max_open_air = 0.0
        self.peaks = []              # peak opening of each completed open/close cycle in the air
        self._opened = False         # opened while in the air, not yet closed again
        self._peak = 0.0

    def _pose(self):
        p = np.array(self.sim["scissors"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def _hinge(self):
        return float(self.sim.state.joint_q.numpy()[self.qi])

    def _clearance(self, p, R, a):
        c, s = np.cos(a), np.sin(a)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        pts = np.vstack([_A, _B @ Rz.T]) @ R.T + p
        return float(pts[:, 2].min() - abs(R[2, 2]) * _HALF_T - TABLE_Z)

    def step(self):
        p, R = self._pose()
        a = self._hinge()
        deg = np.degrees(a)
        self.max_lift = max(self.max_lift, p[2] - self.p0[2])
        clear = self._clearance(p, R, a)
        self.max_clear = max(self.max_clear, clear)
        air = clear > CLEAR
        if air:
            self.air_s += 0.02
            self.max_open_air = max(self.max_open_air, deg)
        if not air:
            self._opened, self._peak = False, 0.0   # a cycle must happen entirely in the air
        elif not self._opened:
            if deg > OPEN_DEG:
                self._opened, self._peak = True, deg
        else:
            self._peak = max(self._peak, deg)
            if deg < CLOSED_DEG:
                self.peaks.append(round(float(self._peak), 1))
                self._opened, self._peak = False, 0.0

    def result(self):
        p, R = self._pose()
        deg = np.degrees(self._hinge())
        rest = float(p[2] - TABLE_Z)
        flat = abs(R[2, 2]) > 0.95
        on_table = -0.002 < rest < REST
        closed = deg < CLOSED_DEG
        m = {"max_pivot_lift_mm": round(1000 * self.max_lift, 1),
             "max_clearance_mm": round(1000 * self.max_clear, 1),
             "time_in_air_s": round(self.air_s, 2),
             "max_open_in_air_deg": round(float(self.max_open_air), 1),
             "cycles_in_air": len(self.peaks), "cycle_peaks_deg": self.peaks,
             "final_hinge_deg": round(float(deg), 1), "rest_height_mm": round(1000 * rest, 1),
             "resting_flat": bool(flat),
             "final_offset_mm": round(1000 * float(np.linalg.norm(p[:2] - self.p0[:2])), 1)}
        ok = len(self.peaks) >= CYCLES and on_table and flat and closed
        return {"success": bool(ok), "metrics": m}
