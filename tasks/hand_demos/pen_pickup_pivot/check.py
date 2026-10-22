"""Scorer for pen_pickup_pivot, independent of the policy. Reads object state only."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("pen",)
TABLE_Z = 0.75
UPRIGHT_DEG = 15.0      # within this of vertical counts as upright when held clear
STAND_DEG = 20.0        # within this of vertical counts as standing on its end on the table
ON_TABLE_M = 0.003      # lower end within this of the table counts as standing on it
CLEAR_M = 0.03          # lower end at least this far above the table counts as lifted clear


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.radius, self.half = self._dims()
        p, axis = self._pose()
        self.axis0 = axis
        self.max_low = 0.0
        self.max_pivot = 0.0
        self.min_tilt_on_tip = 90.0
        self.min_tilt_clear = 90.0

    def _dims(self):
        """Capsule radius and cylinder half-length of the pen, from the collider geometry."""
        shape = self.sim.scene.handles.objects["pen"].shapes[0]
        r, h = np.asarray(self.sim.model.shape_scale.numpy()[shape], float)[:2]
        return float(r), float(h)

    def _pose(self):
        p = np.array(self.sim["pen"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()[:, 2]

    def _low(self, p, axis):
        """Height of the pen's lowest point above the table [m]."""
        return float(p[2] - self.half * abs(axis[2]) - self.radius - TABLE_Z)

    @staticmethod
    def _tilt(axis):
        return float(np.degrees(np.arccos(min(1.0, abs(axis[2])))))

    def step(self):
        p, axis = self._pose()
        low, tilt = self._low(p, axis), self._tilt(axis)
        self.max_low = max(self.max_low, low)
        self.max_pivot = max(self.max_pivot, float(np.degrees(np.arccos(min(1.0, abs(axis @ self.axis0))))))
        if low < ON_TABLE_M:
            self.min_tilt_on_tip = min(self.min_tilt_on_tip, tilt)
        if low > CLEAR_M:
            self.min_tilt_clear = min(self.min_tilt_clear, tilt)

    def result(self):
        p, axis = self._pose()
        low, tilt = self._low(p, axis), self._tilt(axis)
        stood = self.min_tilt_on_tip < STAND_DEG
        held_clear = low > CLEAR_M
        upright = tilt < UPRIGHT_DEG
        m = {"max_pivot_deg": round(self.max_pivot, 1),
             "min_tilt_standing_on_table_deg": round(self.min_tilt_on_tip, 1),
             "stood_on_end": bool(stood),
             "max_lower_end_height_mm": round(1000 * self.max_low, 1),
             "min_tilt_while_clear_deg": round(self.min_tilt_clear, 1),
             "final_lower_end_height_mm": round(1000 * low, 1),
             "final_tilt_from_vertical_deg": round(tilt, 1),
             "final_pivot_deg": round(float(np.degrees(np.arccos(min(1.0, abs(axis @ self.axis0))))), 1)}
        ok = stood and held_clear and upright
        return {"success": bool(ok), "metrics": m}
