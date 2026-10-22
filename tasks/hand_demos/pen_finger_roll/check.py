"""Scorer for pen_finger_roll, independent of the policy. Reads simulator state only:
the pen pose and the arm joint angles (for the palm frame, by forward kinematics)."""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.workcell import Kinematics

OBJECTS = ("pen",)
TABLE_Z = 0.75
CUP_TOP = TABLE_Z + 0.06
CLEAR = 0.01            # pen's lowest point this far above the cup rim: out of the cup
SIDE = 10.0             # swing: tilt on each side of rest [deg]
SWING = 30.0            # swing: total tilt range [deg]
SETTLE, SETTLE_S = 5.0, 0.5   # back within 5 deg of rest for 0.5 s after the swing
ROLL = 30.0             # roll: range of twist about the pen's own axis [deg]


def _twist(R0, R):
    """Rotation from R0 to R about the body's own z, radians in (-pi, pi]."""
    q = Rotation.from_matrix(R0.T @ R).as_quat()
    return float((2 * np.arctan2(q[2], q[3]) + np.pi) % (2 * np.pi) - np.pi)


def _tilt(a):
    """Angle of an axis in the palm x-z plane (rotation about palm +y), radians."""
    return float(np.arctan2(a[2], a[0]))


def _wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.kin = Kinematics()
        self.arm_idx = list(sim.scene.handles.arms["right"].arm_coords)
        self.p0, R0 = self._pen()
        # Centre-to-tip length of the capsule: it starts standing on its tip in the cup.
        self.reach = self.p0[2] - TABLE_Z
        self.max_lift = 0.0
        self.out = False          # the pen has left the cup
        self.rest = None          # in-palm pen axis when it first cleared the cup
        self.R_last = None
        self.twist = 0.0
        self.twist_lo = self.twist_hi = 0.0
        self.tilt_lo = self.tilt_hi = 0.0
        self.calm = 0             # consecutive steps back near rest after both swing sides
        self.settled = False
        self.bottom = -1.0

    def _pen(self):
        p = np.array(self.sim["pen"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def _in_palm(self):
        P, Rp = self.kin.palm_pose(self.sim.state.joint_q.numpy()[self.arm_idx])
        p, R = self._pen()
        return p, R, Rp.T @ R

    def step(self):
        p, Rw, Rr = self._in_palm()
        self.max_lift = max(self.max_lift, p[2] - self.p0[2])
        self.bottom = p[2] - self.reach * abs(Rw[2, 2])      # lowest point of the pen
        if not self.out and self.bottom > CUP_TOP + CLEAR:
            self.out = True
            self.rest = Rr[:, 2].copy()
            self.R_last = Rr
        if not self.out:
            return
        if self.bottom > CUP_TOP:     # still up in the hand
            self.twist += _twist(self.R_last, Rr)
            self.twist_lo, self.twist_hi = min(self.twist_lo, self.twist), max(self.twist_hi, self.twist)
            t = np.degrees(_wrap(_tilt(Rr[:, 2]) - _tilt(self.rest)))
            self.tilt_lo, self.tilt_hi = min(self.tilt_lo, t), max(self.tilt_hi, t)
            if self.tilt_hi >= SIDE and self.tilt_lo <= -SIDE and abs(t) < SETTLE:
                self.calm += 1
                self.settled |= self.calm >= SETTLE_S * 50
            else:
                self.calm = 0
        self.R_last = Rr

    def result(self):
        swing = self.tilt_hi - self.tilt_lo
        roll = np.degrees(self.twist_hi - self.twist_lo)
        held = self.out and self.bottom > CUP_TOP + CLEAR
        m = {"max_lift_mm": round(1000 * self.max_lift, 1), "left_cup": bool(self.out),
             "swing_range_deg": round(float(swing), 1),
             "swing_min_max_deg": [round(float(self.tilt_lo), 1), round(float(self.tilt_hi), 1)],
             "settled_after_swing": bool(self.settled),
             "roll_range_deg": round(float(roll), 1),
             "final_clearance_above_cup_mm": round(1000 * (self.bottom - CUP_TOP), 1), "held_at_end": bool(held)}
        ok = (held and swing >= SWING and self.tilt_hi >= SIDE and self.tilt_lo <= -SIDE
              and self.settled and roll >= ROLL)
        return {"success": bool(ok), "metrics": m}
