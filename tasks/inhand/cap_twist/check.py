"""Scorer for cap_twist, independent of the policy. Reads simulator state only:
the lid hinge angle (joint state) and the palm body pose."""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("jar",)
GOAL_DEG = 360.0          # net counter-clockwise lid turn, seen from above
TURNING_DEG_PER_STEP = 0.1  # lid faster than 5 deg/s counts as "turning"
PALM_YAW_TOL_DEG = 10.0   # palm rotation about vertical, range over turning steps
PALM_MOVE_TOL_M = 0.015   # palm position spread over turning steps


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        m = sim.model
        joints = list(m.joint_label)
        j = next(i for i, l in enumerate(joints) if l.endswith("lid_twist"))
        self.lid_q = int(m.joint_q_start.numpy()[j])
        bodies = list(m.body_label)
        self.palm = next(i for i, l in enumerate(bodies) if l.endswith("/palm"))
        self.a0 = self._angle()
        self.prev_a = self.a0
        p, R = self._palm()
        self.prev_R = R
        self.yaw = 0.0                 # palm rotation about world z, integrated from the start
        self.max_turn = 0.0
        self.min_turn = 0.0
        self.turn_yaws, self.turn_pos = [], []
        self.turning_steps = 0

    def _angle(self):
        return float(self.sim.state.joint_q.numpy()[self.lid_q])

    def _palm(self):
        x = self.sim.state.body_q.numpy()[self.palm]
        return np.array(x[:3], float), Rotation.from_quat(x[3:7]).as_matrix()

    def step(self):
        a = self._angle()
        p, R = self._palm()
        self.yaw += float(Rotation.from_matrix(R @ self.prev_R.T).as_rotvec()[2])
        self.prev_R = R
        turn = a - self.a0
        self.max_turn = max(self.max_turn, turn)
        self.min_turn = min(self.min_turn, turn)
        if abs(np.degrees(a - self.prev_a)) > TURNING_DEG_PER_STEP:
            self.turning_steps += 1
            self.turn_yaws.append(self.yaw)
            self.turn_pos.append(p)
        self.prev_a = a

    def result(self):
        turn = np.degrees(self._angle() - self.a0)
        if self.turn_yaws:
            yaw_range = float(np.degrees(np.ptp(self.turn_yaws)))
            P = np.array(self.turn_pos)
            spread = float(np.max(np.linalg.norm(P - P.mean(0), axis=1)) * 2)
        else:
            yaw_range, spread = 0.0, 0.0
        palm_still = yaw_range <= PALM_YAW_TOL_DEG and spread <= PALM_MOVE_TOL_M
        m = {"lid_turn_deg": round(float(turn), 1),
             "max_lid_turn_deg": round(float(np.degrees(self.max_turn)), 1),
             "min_lid_turn_deg": round(float(np.degrees(self.min_turn)), 1),
             "turning_s": round(self.turning_steps * 0.02, 2),
             "palm_yaw_range_while_turning_deg": round(yaw_range, 2),
             "palm_spread_while_turning_mm": round(1000 * spread, 1),
             "palm_still": bool(palm_still)}
        ok = turn >= GOAL_DEG and palm_still
        return {"success": bool(ok), "metrics": m}
