"""Scorer for cube_pinch. Reads cube pose and hand–cube contact forces.

Success is pick + turn + place with a real fingertip hold: opposition contact
(thumb and at least one other finger) must be present while the cube is clear
of the table for lift and yaw credit. A flick or free tumble does not count.
"""
import numpy as np
from scipy.spatial.transform import Rotation

OBJECTS = ("cube",)
DT = 0.02
TOUCH_N = 0.15          # finger counts as touching above this force [N]
CLEAR_M = 0.04          # cube centre this far above start: clear of the table
LIFT_M = 0.06           # must reach this height while held
TURN_DEG = 20.0         # yaw change (cube 90° symmetry) while held and clear
HOLD_S = 0.4            # cumulative held-and-clear time required
PLACE_M = 0.04          # final xy distance from start [m]
ON_TABLE_M = 0.008      # final height within this of start height
RELEASE_N = 0.1         # hand has let go: total finger+palm force below this


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        labels = list(sim.model.shape_label)
        hand_shapes = list(sim.scene.handles.arms["right"].shapes)
        # thumb, index, middle, ring, little, palm — same grouping as Controller.contact_forces
        self.groups = [[s for s in hand_shapes if key in labels[s] and not labels[s].endswith("_visual")]
                       for key in [f"/link_{i}_" for i in range(1, 6)] + ["/palm/palm_"]]
        self.cube_shapes = list(sim.scene.handles.objects["cube"].shapes)
        self.p0, R0 = self._pose()
        self.yaw0 = self._yaw(R0)
        self.max_lift_held = 0.0
        self.max_turn_held = 0.0
        self.held_clear_steps = 0
        self.max_held_fingers = 0

    def _pose(self):
        p = np.array(self.sim["cube"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    @staticmethod
    def _yaw(R):
        # Cube yaw modulo 90 deg (4-fold symmetry about z when upright).
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _forces(self):
        c = self.sim.contacts
        c.update(self.sim.step_count)
        return np.array([float(np.linalg.norm(c.pairwise_force(g, self.cube_shapes))) if g else 0.0
                         for g in self.groups])

    def _held(self, f):
        """Opposition fingertip grasp: thumb plus at least one other finger on the cube."""
        fingers = f[:5]
        n_touch = int(np.sum(fingers > TOUCH_N))
        return bool(fingers[0] > TOUCH_N and fingers[1:].max() > TOUCH_N), n_touch

    def step(self):
        p, R = self._pose()
        lift = float(p[2] - self.p0[2])
        f = self._forces()
        held, n_touch = self._held(f)
        self.max_held_fingers = max(self.max_held_fingers, n_touch if held else 0)
        if held and lift > CLEAR_M:
            self.held_clear_steps += 1
            self.max_lift_held = max(self.max_lift_held, lift)
            d = (self._yaw(R) - self.yaw0 + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self.max_turn_held = max(self.max_turn_held, abs(d))

    def result(self):
        p, R = self._pose()
        f = self._forces()
        off = float(np.linalg.norm(p[:2] - self.p0[:2]))
        upright = abs(R[2, 2]) > 0.99 or max(abs(R[2, 0]), abs(R[2, 1])) > 0.99
        on_table = abs(p[2] - self.p0[2]) < ON_TABLE_M
        released = float(f.sum()) < RELEASE_N
        held_clear_s = self.held_clear_steps * DT
        turn_deg = float(np.degrees(self.max_turn_held))
        m = {
            "max_lift_held_mm": round(1000 * self.max_lift_held, 1),
            "max_turn_while_held_deg": round(turn_deg, 1),
            "held_clear_s": round(held_clear_s, 2),
            "max_held_fingers": int(self.max_held_fingers),
            "final_offset_mm": round(1000 * off, 1),
            "resting_flat": bool(upright),
            "on_table": bool(on_table),
            "released": bool(released),
            "final_hand_force_N": round(float(f.sum()), 3),
        }
        ok = (
            held_clear_s >= HOLD_S - 1e-9
            and self.max_lift_held > LIFT_M
            and turn_deg > TURN_DEG
            and off < PLACE_M
            and upright
            and on_table
            and released
        )
        return {"success": bool(ok), "metrics": m}
