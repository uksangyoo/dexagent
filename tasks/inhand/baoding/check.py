"""Scorer for baoding, independent of the policy. Reads simulator state only:
the two ball poses and the arm joint state (the palm frame by forward kinematics)."""
import numpy as np
from dexagent.workcell import Kinematics

OBJECTS = ("ball_a", "ball_b")
GOAL_DEG = 180.0          # the balls have exchanged places
PALM_UP_DEG = 30.0        # palm normal within this of world up to count / to finish
COUNT_FROM_S = 1.0        # the balls drop in and roll to rest first; that settling is not rotation


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.kin = Kinematics()
        self.arm_idx = list(sim.scene.handles.arms["right"].arm_coords)
        self.last = None
        self.total = 0.0          # signed cumulative rotation of a->b about the palm normal [rad]
        self.max_abs = 0.0
        self.counted_steps = 0
        self.steps = 0
        self.ever_dropped = False
        self.max_tilt = 0.0

    def _palm(self):
        return self.kin.palm_pose(self.sim.state.joint_q.numpy()[self.arm_idx])

    def _in_palm(self):
        P, Rp = self._palm()
        out = []
        for n in OBJECTS:
            p = np.array(self.sim[n].pose, float)[:3]
            out.append(Rp.T @ (p - P))
        return out, Rp

    @staticmethod
    def _held(p):
        # Within the palm's footprint and within 75 mm of the palm face (palm frame:
        # +x out of the face, +y toward the thumb, +z toward the fingers).
        return -0.01 < p[0] < 0.075 and abs(p[1]) < 0.06 and -0.03 < p[2] < 0.15

    def _state(self):
        (a, b), Rp = self._in_palm()
        tilt = float(np.degrees(np.arccos(np.clip(Rp[2, 0], -1, 1))))   # palm +x vs world +z
        held = self._held(a) and self._held(b)
        return a, b, tilt, held

    def step(self):
        self.steps += 1
        a, b, tilt, held = self._state()
        self.max_tilt = max(self.max_tilt, tilt)
        if not held:
            self.ever_dropped = True
        d = b - a
        ang = float(np.arctan2(d[1], d[2]))
        # Rotation only counts while both balls are in the hand and the palm faces up.
        if held and tilt < PALM_UP_DEG and self.last is not None and float(self.sim.time) > COUNT_FROM_S:
            self.total += (ang - self.last + np.pi) % (2 * np.pi) - np.pi
            self.max_abs = max(self.max_abs, abs(self.total))
            self.counted_steps += 1
        self.last = ang if held else None

    def result(self):
        a, b, tilt, held = self._state()
        swap = float(np.degrees(abs(self.total)))
        m = {"swap_deg": round(swap, 1), "max_swap_deg": round(float(np.degrees(self.max_abs)), 1),
             "both_in_hand_end": bool(held), "ever_dropped": bool(self.ever_dropped),
             "final_palm_tilt_deg": round(tilt, 1), "max_palm_tilt_deg": round(self.max_tilt, 1),
             "ball_centre_dist_end_mm": round(1000 * float(np.linalg.norm(b - a)), 1)}
        ok = swap >= GOAL_DEG and held and not self.ever_dropped and tilt < PALM_UP_DEG
        return {"success": bool(ok), "metrics": m}
