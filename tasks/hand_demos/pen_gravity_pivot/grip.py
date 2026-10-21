"""Fingertip IK with a pointing constraint, for pinching a pen with 16-20 mm thick fingertips.

The shared `Kinematics.solve_finger` places the pad centre only. Pinching a
7.5 mm cylinder also needs the distal links pointed: the rounded fingertip
ends, tilted inward, meet the pen where flat pads cannot. Used by
pen_gravity_pivot and pen_pickup_pivot.
"""
import numpy as np
from scipy.optimize import least_squares
from dexagent.rss.hand import CURL, OPEN




def pad(kin, i):
    """True pad centre of finger i in the world, from the current kinematics state."""
    return kin.data.site_xpos[kin.tip_site[i]].copy()


def pads_world(c):
    """Pad centres (5, 3) from the controller's measured joints."""
    q = c.sim.state.joint_q.numpy()
    c.kin.set(arm=q[c.arm_idx], hand=q[c.hand_idx])
    return np.array([pad(c.kin, i) for i in range(5)])


def solve(kin, i, target, direction=None, seed=None, weight=0.02):
    m, d = kin.model, kin.data
    ids = [m.joint(f"joint_{i + 1}_{j}").id for j in range(1, 5)]
    lim = m.jnt_range[ids]
    idx = kin.finger_q[i]
    body = m.body(f"link_{i + 1}_4").id
    seed = np.clip(np.zeros(4) if seed is None else seed, lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)

    def residual(q):
        d.qpos[idx] = q
        kin.mj.mj_kinematics(m, d)
        P, R = d.xpos[kin.palm], d.xmat[kin.palm].reshape(3, 3)
        r = [R.T @ (pad(kin, i) - P) - target]
        if direction is not None:
            r.append(weight * (R.T @ d.xmat[body].reshape(3, 3)[:, 0] - direction))
        r.append(0.001 * (q - seed))
        return np.concatenate(r)
    fit = least_squares(residual, seed, bounds=(lim[:, 0], lim[:, 1]), max_nfev=300)
    res = residual(fit.x)
    return fit.x, float(np.linalg.norm(res[:3])), (float(np.linalg.norm(res[3:6]) / weight) if direction is not None else 0.0)


TIP_END = 0.009   # pad-centre site to the fingertip's end, along the distal link (tip mesh)


def tweezer(kin, centre, axis, down, half, beta=np.radians(50), base=None, tuck=(2, 3, 4)):
    """Thumb and index end caps at centre +/- half*axis, each tilted `beta` from `down` toward the other.

    For a thin cylinder lying on a table: the rounded ends of two inward-tilted
    fingertips meet it above its equator but inside the friction cone, and the
    distal links diverge upward so they do not collide with each other.
    """
    q = (OPEN if base is None else np.asarray(base, float)).copy()
    for i in tuck:
        q[4 * i:4 * i + 4] = CURL if i != 4 else [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    errs = []
    for i, sign in ((0, 1.0), (1, -1.0)):
        d = np.cos(beta) * down - sign * np.sin(beta) * axis
        end = centre + sign * half * axis
        q[4 * i:4 * i + 4], e, a = solve(kin, i, end - TIP_END * d, d, q[4 * i:4 * i + 4], 0.02)
        errs.append((e, a))
    return q, errs



def tweezer3(kin, centre, axis, down, half, spread, beta=np.radians(50), beta_mid=None, base=None, mid_lift=0.0,
             mid_back=0.0):
    """Thumb end at centre + half*axis; index and middle ends at centre - half*axis -/+ spread along palm y.

    Two contacts spaced along a pen on one side and the thumb between them on
    the other: friction at the spaced pair holds the pen against turning about
    the pinch axis, which a two-finger pinch cannot do without torsional
    friction. `mid_lift` raises the middle finger's end and `mid_back` draws it
    back across the pinch, away from the object, to release it.
    """
    q = (OPEN if base is None else np.asarray(base, float)).copy()
    for i in (3, 4):
        q[4 * i:4 * i + 4] = CURL if i != 4 else [0.3, CURL[1] - .2, CURL[2], CURL[3]]
    y = np.array([0.0, 1.0, 0.0])
    errs = []
    for i, sign, lateral, b, lift, back in ((0, 1.0, 0.0, beta, 0.0, 0.0), (1, -1.0, spread, beta, 0.0, 0.0),
                                            (2, -1.0, -spread, beta if beta_mid is None else beta_mid, mid_lift, mid_back)):
        d = np.cos(b) * down - sign * np.sin(b) * axis
        end = centre + sign * (half + back) * axis + lateral * y - lift * down
        q[4 * i:4 * i + 4], e, a = solve(kin, i, end - TIP_END * d, d, q[4 * i:4 * i + 4], 0.02)
        errs.append((e, a))
    return q, errs
