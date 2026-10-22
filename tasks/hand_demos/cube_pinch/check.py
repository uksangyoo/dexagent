"""Scorer for cube_pinch, independent of the policy. Reads object state only."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("cube",)
TABLE_Z = 0.75


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.p0, R0 = self._pose()
        self.yaw0 = self._yaw(R0)
        self.max_lift = 0.0
        self.max_turn_lifted = 0.0

    def _pose(self):
        p = np.array(self.sim["cube"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    @staticmethod
    def _yaw(R):
        # Cube yaw modulo 90 deg (4-fold symmetric about z when upright).
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def step(self):
        p, R = self._pose()
        lift = p[2] - self.p0[2]
        self.max_lift = max(self.max_lift, lift)
        if lift > 0.06:
            d = (self._yaw(R) - self.yaw0 + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self.max_turn_lifted = max(self.max_turn_lifted, abs(d))

    def result(self):
        p, R = self._pose()
        off = float(np.linalg.norm(p[:2] - self.p0[:2]))
        upright = abs(R[2, 2]) > 0.99 or max(abs(R[2, 0]), abs(R[2, 1])) > 0.99
        on_table = abs(p[2] - self.p0[2]) < 0.005
        m = {"max_lift_mm": round(1000 * self.max_lift, 1),
             "max_turn_while_lifted_deg": round(float(np.degrees(self.max_turn_lifted)), 1),
             "final_offset_mm": round(1000 * off, 1), "resting_flat": bool(upright), "on_table": bool(on_table)}
        ok = self.max_lift > 0.08 and np.degrees(self.max_turn_lifted) > 25 and off < 0.015 and upright and on_table
        return {"success": bool(ok), "metrics": m}
