"""Scorer for cube_palm, independent of the policy. Reads simulator state only:
the cube pose, the arm joint angles (for the palm frame) and contact forces."""
import numpy as np
from scipy.spatial.transform import Rotation
from dexagent.workcell import Kinematics

OBJECTS = ("cube",)
DT = 0.02
ON_PALM_S = 1.0          # cumulative time the open palm alone must carry the cube
CARRY_HEIGHT = 0.10      # cube centre at least this far above its start height while palm-carried
PLACE_RADIUS = 0.08      # final xy distance from the start position


class Monitor:
    def __init__(self, sim):
        self.sim = sim
        self.kin = Kinematics()
        self.arm_idx = list(sim.scene.handles.arms["right"].arm_coords)
        model = sim.model
        labels = list(model.shape_label)
        hand_shapes = list(sim.scene.handles.arms["right"].shapes)
        self.groups = [[s for s in hand_shapes if key in labels[s] and not labels[s].endswith("_visual")]
                       for key in [f"/link_{i}_" for i in range(1, 6)] + ["/palm/palm_"]]
        self.cube_shapes = list(sim.scene.handles.objects["cube"].shapes)
        self.p0, _ = self._pose()
        self.max_lift = 0.0
        self.max_palm_up = -1.0
        self.palm_steps = 0
        self.palm_streak = 0
        self.best_streak = 0
        self.max_palm_force_lifted = 0.0
        self.steps = 0

    def _pose(self):
        p = np.array(self.sim["cube"].pose, float)
        return p[:3], Rotation.from_quat(p[3:]).as_matrix()

    def _forces(self):
        """Contact-force magnitude [N] on the cube from thumb, index, middle, ring, little, palm."""
        c = self.sim.contacts
        c.update(self.sim.step_count)
        return np.array([float(np.linalg.norm(c.pairwise_force(g, self.cube_shapes))) if g else 0.0
                         for g in self.groups])

    def _on_palm(self, p, f):
        P, Rp = self.kin.palm_pose(self.sim.state.joint_q.numpy()[self.arm_idx])
        palm_up = float(Rp[2, 0])  # palm normal (+x of the palm frame) against world up
        rel = Rp.T @ (p - P)
        lifted = p[2] - self.p0[2] > CARRY_HEIGHT
        ok = (palm_up > 0.7 and lifted and f[5] > 0.02 and f[:5].sum() < 0.5
              and rel[0] < 0.06 and abs(rel[1]) < 0.045)
        return ok, palm_up, lifted

    def step(self):
        self.steps += 1
        p, _ = self._pose()
        self.max_lift = max(self.max_lift, p[2] - self.p0[2])
        f = self._forces()
        ok, palm_up, lifted = self._on_palm(p, f)
        if lifted:
            self.max_palm_up = max(self.max_palm_up, palm_up)
            if palm_up > 0.7:
                self.max_palm_force_lifted = max(self.max_palm_force_lifted, f[5])
        self.palm_steps += ok
        self.palm_streak = self.palm_streak + 1 if ok else 0
        self.best_streak = max(self.best_streak, self.palm_streak)
        self.final_hand_force = float(f.sum())

    def result(self):
        p, R = self._pose()
        off = float(np.linalg.norm(p[:2] - self.p0[:2]))
        flat = bool(np.abs(R[2]).max() > 0.99)        # some face down
        on_table = bool(abs(p[2] - self.p0[2]) < 0.005)
        released = getattr(self, "final_hand_force", 1.0) < 0.05
        on_palm_s = self.palm_steps * DT
        m = {"max_lift_mm": round(1000 * self.max_lift, 1),
             "max_palm_up_while_lifted": round(self.max_palm_up, 3),
             "max_palm_force_lifted_N": round(self.max_palm_force_lifted, 3),
             "on_open_palm_s": round(on_palm_s, 2),
             "longest_on_open_palm_s": round(self.best_streak * DT, 2),
             "final_offset_mm": round(1000 * off, 1),
             "final_height_err_mm": round(1000 * (p[2] - self.p0[2]), 1),
             "resting_flat": flat, "on_table": on_table, "released": bool(released)}
        ok = on_palm_s >= ON_PALM_S - 1e-9 and flat and on_table and released and off < PLACE_RADIUS
        return {"success": bool(ok), "metrics": m}
