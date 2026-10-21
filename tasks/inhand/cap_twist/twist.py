"""Rim-gait helpers for the cap-twist scene: lid readings and fingertip rim targets."""
import numpy as np
from dexagent.hand import Controller

# Disk grasp measured on this hand (palm frame, metres and radians): the pads sit on a
# circle about CENTRE in the palm's y-z plane, at these angles from the finger axis
# (+z toward +y). Thumb, index and middle stay reachable over a +-35 deg twist.
CENTRE = np.array([0.08, 0.02, 0.13])
PHI = {0: np.radians(160), 1: np.radians(35), 2: np.radians(-10)}


def rim_targets(c, lid, R, palm, twist, radius, lift=0.0):
    """20 joint angles placing thumb/index/middle pads on the rim, turned `twist` rad CCW from above,
    `lift` metres above the lid's mid-height."""
    targets = {}
    for f, phi in PHI.items():
        d = R @ np.array([0, np.sin(phi), np.cos(phi)])
        psi = np.arctan2(d[1], d[0]) + twist
        p = lid + radius * np.array([np.cos(psi), np.sin(psi), 0.0]) + [0, 0, lift]
        targets[f] = R.T @ (p - palm)
    return c.fingertips(targets, _tucked(c))


def _tucked(c):
    from dexagent.hand import OPEN, CURL
    q = OPEN.copy()
    q[12:16] = CURL
    q[16:20] = [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    q[:12] = c.hand[:12]
    return q


class CapController(Controller):
    """Controller that also logs the lid hinge angle and pad forces on the lid."""

    def __init__(self, sim, session=None):
        super().__init__(sim, session)
        model = sim.model
        labels = list(model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("lid_twist"))
        self.lid_q = int(model.joint_q_start.numpy()[j])
        shapes = list(model.shape_label)
        self.lid_shapes = [i for i, l in enumerate(shapes) if "/lid/" in l or l.endswith("/lid")]
        self.peak_force = np.zeros(6)
        self.yaws = []

    def lid_angle(self):
        return float(self.sim.state.joint_q.numpy()[self.lid_q])

    def palm_yaw(self):
        _, R = self.measured_palm()
        z = R[:, 2]  # finger direction
        return float(np.arctan2(z[1], z[0]))

    def max_yaw_change(self, yaw0):
        return max([abs((y - yaw0 + np.pi) % (2 * np.pi) - np.pi) for y in self.yaws] or [0.0])

    def record(self):
        super().record()
        self.sim.contacts.update(self.sim.step_count)
        cont = self.sim.contacts
        forces = np.array([float(np.linalg.norm(cont.pairwise_force(g, self.lid_shapes))) if len(g) else 0.
                           for g in self._finger_shapes()])
        self.peak_force = np.maximum(self.peak_force, forces)
        self.yaws.append(self.palm_yaw())
        row = self.trace[-1]
        row["lid_deg"] = round(float(np.degrees(self.lid_angle())), 2)
        row["lid_force"] = forces.round(3).tolist()
