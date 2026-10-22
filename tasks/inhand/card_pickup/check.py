"""Scorer for card_pickup, independent of the policy. Reads simulator state only
(tile and book poses, robot joint positions)."""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.workcell import Kinematics

OBJECTS = ("tile",)


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.kin = Kinematics()
        h = sim.scene.handles.arms["right"]
        self.arm_idx, self.hand_idx = list(h.arm_coords), list(h.gripper_coords)
        self.p0, self.R0 = self._pose()
        self.max_lift = 0.0
        self.max_slide_on_book = 0.0
        self.max_tilt = 0.0

    def _pose(self):
        p = np.array(self.sim["tile"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def _pinch_mid(self):
        q = self.sim.state.joint_q.numpy()
        self.kin.set(arm=q[self.arm_idx], hand=q[self.hand_idx])
        tips = np.array([self.kin.data.site_xpos[s].copy() for s in self.kin.tip_site])
        return (tips[0] + tips[1]) / 2          # thumb and index fingertips

    def step(self):
        p, R = self._pose()
        lift = p[2] - self.p0[2]
        self.max_lift = max(self.max_lift, lift)
        if abs(lift) < 0.005:                   # still lying on the book: how far it has been slid
            self.max_slide_on_book = max(self.max_slide_on_book, float(np.linalg.norm(p[:2] - self.p0[:2])))
        self.max_tilt = max(self.max_tilt, float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1)))))

    def result(self):
        p, R = self._pose()
        lift = p[2] - self.p0[2]
        d_pinch = float(np.linalg.norm(p - self._pinch_mid()))
        m = {"max_slide_on_book_mm": round(1000 * self.max_slide_on_book, 1),
             "max_tilt_deg": round(self.max_tilt, 1),
             "max_lift_mm": round(1000 * self.max_lift, 1),
             "final_lift_mm": round(1000 * lift, 1),
             "tile_to_pinch_mm": round(1000 * d_pinch, 1)}
        ok = lift > 0.08 and d_pinch < 0.06
        return {"success": bool(ok), "metrics": m}
