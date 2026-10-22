"""Scorer for pen_table_slide, independent of the policy. Reads simulator state only:
the pen's pose, the robot's joint positions (for fingertip positions) and contacts."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("pen",)
TABLE_Z = 0.75
ON_TABLE = 0.003      # pen's lower end within 3 mm of the table: standing on it
TOUCH_N = 0.2         # a finger counts as touching the pen above this contact force
SLIP_MIN = 0.015      # the pen must slide at least 15 mm up through the fingers
LIFT_MIN = 0.060      # final lower end at least 60 mm above the table (clear of the 60 mm guide)


class Monitor:
    def __init__(self, sim):
        from dexagent.workcell import Kinematics
        self.sim = sim
        self.kin = Kinematics()
        h = sim.scene.handles.arms["right"]
        self.arm_idx, self.hand_idx = list(h.arm_coords), list(h.gripper_coords)
        labels = list(sim.model.shape_label)
        shapes = list(h.shapes)
        self.fingers = [[s for s in shapes if s < len(labels) and key in labels[s] and not labels[s].endswith("_visual")]
                        for key in [f"/link_{i}_" for i in range(1, 6)]]
        self.pen_shapes = list(sim.scene.handles.objects["pen"].shapes)
        p, a = self._pen()
        # The pen spawns standing on its tip 0.5 mm above the table: centre-to-end length.
        self.end = float(p[2] - TABLE_Z - 0.0005)
        self.cum = 0.0           # pen travel along its axis relative to the thumb tip, while slid
        self.cum_min = 0.0
        self.slip = 0.0
        self.prev = None         # previous qualifying sample's axial coordinate
        self.max_tip = 0.0
        self.max_force = 0.0
        self.min_tip_while_slid = None

    def _pen(self):
        q = np.array(self.sim["pen"].pose, float)
        a = Rotation.from_quat(q[3:]).as_matrix()[:, 2]
        return q[:3], (a if a[2] >= 0 else -a)

    def _forces(self):
        c = self.sim.contacts
        return np.array([float(np.linalg.norm(c.pairwise_force(g, self.pen_shapes))) if g else 0.0
                         for g in self.fingers])

    def _thumb_tip(self):
        q = self.sim.state.joint_q.numpy()
        self.kin.set(arm=q[self.arm_idx], hand=q[self.hand_idx])
        return self.kin.data.site_xpos[self.kin.tip_site[0]].copy()

    def _state(self):
        p, a = self._pen()
        tip_h = float(p[2] - self.end * a[2] - TABLE_Z)
        f = self._forces()
        held = f[0] > TOUCH_N and f[1:].max() > TOUCH_N
        return p, a, tip_h, f, held

    def step(self):
        p, a, tip_h, f, held = self._state()
        self.max_tip = max(self.max_tip, tip_h)
        self.max_force = max(self.max_force, float(f.sum()))
        if held and tip_h < ON_TABLE:
            s = float((p - self._thumb_tip()) @ a)
            if self.prev is not None:
                self.cum += s - self.prev
                self.cum_min = min(self.cum_min, self.cum)
                self.slip = max(self.slip, self.cum - self.cum_min)
            self.prev = s
            self.min_tip_while_slid = tip_h if self.min_tip_while_slid is None else min(self.min_tip_while_slid, tip_h)
        else:
            self.prev = None     # travel across a gap in the grasp or off the table does not count

    def result(self):
        p, a, tip_h, f, held = self._state()
        tilt = float(np.degrees(np.arccos(min(1.0, abs(a[2])))))
        m = {"slip_on_table_mm": round(1000 * self.slip, 1),
             "max_tip_height_mm": round(1000 * self.max_tip, 1),
             "final_tip_height_mm": round(1000 * tip_h, 1),
             "final_tilt_deg": round(tilt, 1),
             "final_held": bool(held),
             "final_finger_forces_N": [round(float(x), 2) for x in f],
             "max_total_finger_force_N": round(self.max_force, 2)}
        ok = self.slip > SLIP_MIN and tip_h > LIFT_MIN and held
        return {"success": bool(ok), "metrics": m}
