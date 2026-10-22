"""Scorer for cube_reorient, independent of the policy. Reads simulator state only:
the cube pose, the arm joint state (for the measured palm frame) and contact forces.

All cube quantities are measured in the measured palm frame (+x out of the palm
face, +y toward the thumb, +z along the fingers), relative to the cube's own start.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.hand import Controller

OBJECTS = ("cube",)
GOAL = np.radians(90)
TOL = np.radians(10)        # final yaw within 80-100 deg of the start, either sense
FLAT = 0.985                # some cube face within ~10 deg of lying flat on the palm
HEIGHT_TOL = 0.005          # cube centre height above the palm within 5 mm of its start
FINGER_N = 0.5              # at the end the fingers together press the cube less than this
PALM_MOVE = 0.03            # the palm stays put: < 30 mm ...
PALM_TURN = np.radians(15)  # ... and < 15 deg from its start pose
SETTLE_STEPS = 10           # contact forces averaged over the last 0.2 s
REF_STEPS = (5, 25)         # resting-height reference: lowest palm-frame height over 0.1-0.5 s,
                            # once the arm's initial sag and the cube's drop onto the palm have settled


def yaw_of(R):
    """Cube yaw about the palm normal (palm x), modulo 90 deg, and how flat it lies."""
    flat = float(np.abs(R[0]).max())
    k = int(np.argmax(np.abs(R[0])))
    a = [R[:, j] for j in range(3) if j != k][0]
    return float(np.arctan2(a[1], a[2]) % (np.pi / 2)), flat


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.c = Controller(sim)    # read-only use: measured palm frame, contact forces
        self.P0, self.Rp0 = self.c.measured_palm()
        self.p0, R0 = self.c.in_palm("cube")
        self.last, _ = yaw_of(R0)
        self.total = 0.0
        self.max_abs_yaw = 0.0
        self.min_flat = 1.0
        self.left_palm = False
        self.max_palm_move = 0.0
        self.max_palm_turn = 0.0
        self.forces = []
        self.k = 0
        self.h_ref = None

    def step(self):
        p, R = self.c.in_palm("cube")
        self.k += 1
        if REF_STEPS[0] <= self.k <= REF_STEPS[1]:
            self.h_ref = p[0] if self.h_ref is None else min(self.h_ref, p[0])
        y, flat = yaw_of(R)
        self.total += (y - self.last + np.pi / 4) % (np.pi / 2) - np.pi / 4
        self.last = y
        self.min_flat = min(self.min_flat, flat)
        if flat > 0.9:
            self.max_abs_yaw = max(self.max_abs_yaw, abs(self.total))
        if p[0] > 0.09 or abs(p[1]) > 0.07 or not (-0.02 < p[2] < 0.16):
            self.left_palm = True
        P, Rp = self.c.measured_palm()
        self.max_palm_move = max(self.max_palm_move, float(np.linalg.norm(P - self.P0)))
        self.max_palm_turn = max(self.max_palm_turn, float(Rotation.from_matrix(self.Rp0.T @ Rp).magnitude()))
        self.forces.append(self.c.contact_forces("cube"))
        self.forces = self.forces[-SETTLE_STEPS:]

    def result(self):
        p, R = self.c.in_palm("cube")
        _, flat = yaw_of(R)
        f = np.mean(self.forces, axis=0) if self.forces else self.c.contact_forces("cube")
        dh = float(p[0] - (self.p0[0] if self.h_ref is None else self.h_ref))
        yaw_err = abs(abs(self.total) - GOAL)
        resting = abs(dh) < HEIGHT_TOL and abs(p[1]) < 0.045 and f[5] > 0.02 and f[:5].sum() < FINGER_N
        palm_still = self.max_palm_move < PALM_MOVE and self.max_palm_turn < PALM_TURN
        m = {"yaw_deg": round(float(np.degrees(self.total)), 1),
             "yaw_error_deg": round(float(np.degrees(yaw_err)), 1),
             "max_abs_yaw_deg": round(float(np.degrees(self.max_abs_yaw)), 1),
             "flat": round(flat, 3), "min_flat": round(self.min_flat, 3),
             "height_change_mm": round(1000 * dh, 1),
             "drift_mm": round(1000 * float(np.linalg.norm(p[1:] - self.p0[1:])), 1),
             "cube_in_palm_mm": np.round(1000 * p, 1).tolist(),
             "finger_force_N": round(float(f[:5].sum()), 2), "palm_force_N": round(float(f[5]), 2),
             "resting_on_palm": bool(resting), "left_palm": bool(self.left_palm),
             "palm_move_mm": round(1000 * self.max_palm_move, 1),
             "palm_turn_deg": round(float(np.degrees(self.max_palm_turn)), 1)}
        ok = yaw_err < TOL and flat > FLAT and resting and not self.left_palm and palm_still
        return {"success": bool(ok), "metrics": m}
