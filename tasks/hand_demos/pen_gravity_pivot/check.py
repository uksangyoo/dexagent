"""Scorer for pen_gravity_pivot, independent of the policy.

Reads simulator state only: the pen's pose, and the measured robot joints
(through forward kinematics, for the palm frame and the fingertip pads).
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.workcell import Kinematics

OBJECTS = ("pen",)
DT = 0.02
LEVEL_LIFT, LEVEL_TILT, LEVEL_STEPS = 0.07, 15.0, 20     # >70 mm up, within 15 deg of horizontal, for 0.4 s
PIVOT_DEG = 60.0                                          # pen turns this far in the palm frame
FINAL_TILT, FINAL_LIFT = 20.0, 0.05                       # held near vertical, >50 mm above start
NEAR = 0.02                                               # pad centre to pen axis, for "held by the fingers"
WINDOW = 50                                               # the last 1 s


def _ang(u, v):
    """Angle between two lines (unsigned axes) [deg]."""
    return float(np.degrees(np.arccos(np.clip(abs(float(u @ v)), 0.0, 1.0))))


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.kin = Kinematics()
        h = sim.scene.handles.arms["right"]
        self.arm_idx, self.hand_idx = list(h.arm_coords), list(h.gripper_coords)
        self.p0, a0 = self._pen()
        self.a_prev = a0
        self.level_run = 0
        self.level_palm = None            # pen axis in the palm frame at the end of the latest level hold
        self.level_world = None
        self.level_s = 0.0                # longest level hold [s]
        self.max_lift = 0.0
        self.max_pivot_palm = 0.0
        self.max_swept_world = 0.0
        self.max_rate = 0.0
        self.last = deque(maxlen=WINDOW)

    def _pen(self):
        p = np.array(self.sim["pen"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()[:, 2]

    def _robot(self):
        q = self.sim.state.joint_q.numpy()
        self.kin.set(arm=q[self.arm_idx], hand=q[self.hand_idx])
        d = self.kin.data
        Rp = d.xmat[self.kin.palm].reshape(3, 3).copy()
        pads = np.array([d.site_xpos[s].copy() for s in self.kin.tip_site])
        return Rp, pads

    def step(self):
        p, a = self._pen()
        Rp, pads = self._robot()
        ap = Rp.T @ a
        lift = p[2] - self.p0[2]
        self.max_lift = max(self.max_lift, lift)
        self.max_rate = max(self.max_rate, _ang(a, self.a_prev) / DT)
        self.a_prev = a
        tilt_h = 90.0 - _ang(a, np.array([0, 0, 1.0]))
        if lift > LEVEL_LIFT and tilt_h < LEVEL_TILT:
            self.level_run += 1
            self.level_s = max(self.level_s, self.level_run * DT)
            if self.level_run >= LEVEL_STEPS:
                self.level_palm, self.level_world = ap.copy(), a.copy()
        else:
            self.level_run = 0
        pivot = _ang(ap, self.level_palm) if self.level_palm is not None else 0.0
        if self.level_palm is not None:
            self.max_pivot_palm = max(self.max_pivot_palm, pivot)
            self.max_swept_world = max(self.max_swept_world, _ang(a, self.level_world))
        # Pads touching the pen: within NEAR of its axis and within its length.
        rel = pads - p
        along = rel @ a
        dist = np.linalg.norm(rel - np.outer(along, a), axis=1)
        n_near = int(np.sum((dist < NEAR) & (np.abs(along) < 0.085)))
        self.last.append((_ang(a, np.array([0, 0, 1.0])), lift, pivot, n_near))

    def result(self):
        if self.last:
            tilt, lift, pivot, near = (np.array(x) for x in zip(*self.last))
        else:
            tilt = lift = pivot = near = np.zeros(1)
        level_done = self.level_palm is not None
        final_ok = (level_done and float(tilt.max()) < FINAL_TILT and float(lift.min()) > FINAL_LIFT
                    and float(pivot.min()) > PIVOT_DEG and int(near.min()) >= 2)
        m = {"max_lift_mm": round(1000 * self.max_lift, 1),
             "level_hold_s": round(self.level_s, 2),
             "held_level": bool(level_done),
             "max_pivot_in_palm_deg": round(self.max_pivot_palm, 1),
             "max_swept_world_deg": round(self.max_swept_world, 1),
             "max_turn_rate_dps": round(self.max_rate, 0),
             "final_pivot_in_palm_deg": round(float(pivot[-1]), 1),
             "final_tilt_from_vertical_deg": round(float(tilt[-1]), 1),
             "final_tilt_max_last_1s_deg": round(float(tilt.max()), 1),
             "final_lift_min_last_1s_mm": round(1000 * float(lift.min()), 1),
             "pads_on_pen_min_last_1s": int(near.min())}
        return {"success": bool(final_ok), "metrics": m}
