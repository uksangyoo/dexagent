"""Thumb-index grasp synthesis for a pen lying on the table (CPU MuJoCo, geometry only).

The finger pads are 16 x 20 mm rounded shells and the pen is 7.5 mm thick, so
a pinch only lifts the pen if both pads meet it no higher than ~35 deg above
its equator (friction 0.9) while neither fingertip goes through the table.
This searches the palm offset over the pen and the thumb/index joints for
exactly that, on the hand's own collision meshes, and returns joint targets
plus the palm offset. Nothing here touches the simulation.
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from dexagent.workcell import ROBOT_XML, hand_down
from dexagent.objects import PEN_RADIUS, PEN_HALF

_CACHE = {}


def _model(yaw_R):
    import mujoco
    key = tuple(np.round(yaw_R, 6).ravel())
    if key in _CACHE:
        return _CACHE[key]
    root = ET.parse(ROBOT_XML).getroot()
    palm = copy.deepcopy(root.find(".//body[@name='palm']"))
    world = root.find("worldbody")
    for b in list(world):
        world.remove(b)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(yaw_R).as_quat()
    palm.set("pos", "0 0 0"); palm.set("quat", f"{q[3]} {q[0]} {q[1]} {q[2]}")
    world.append(palm)
    # Pen along world y'(the palm's +y when the palm faces down), table plane below.
    pen = ET.SubElement(world, "body", name="pen", mocap="true")
    ET.SubElement(pen, "geom", name="pen", type="capsule", size=f"{PEN_RADIUS} {PEN_HALF}", contype="0", conaffinity="0")
    table = ET.SubElement(world, "body", name="table", mocap="true")
    ET.SubElement(table, "geom", name="table", type="box", size="0.3 0.3 0.01", contype="0", conaffinity="0")
    act = root.find("actuator")
    if act is not None:
        for a in list(act):
            if a.get("joint", "") in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"):
                act.remove(a)
    root.find("contact").clear()
    xml = ET.tostring(root, encoding="unicode").replace('file="', f'file="{ROBOT_XML.parent}/')
    m = mujoco.MjModel.from_xml_string(xml)
    _CACHE[key] = m
    return m


def synthesize(yaw_R=None, pen_dir=None, lateral=0.0, seed=None, height_deg=25.0, iters=150, w_axis=0.3, max_tilt=12.0, verbose=False):
    """Palm offset (pen-centre frame, world axes) and 20-joint hand for a thumb/index pinch.

    Returns (palm_minus_pen, q_contact, info). Thumb/index joints touch the pen;
    other fingers are curled. `lateral` shifts the pinch along the pen.
    """
    import mujoco
    R = hand_down() if yaw_R is None else yaw_R
    m = _model(R)
    d = mujoco.MjData(m)
    pen_dir = R[:, 1] if pen_dir is None else np.asarray(pen_dir, float)
    from scipy.spatial.transform import Rotation
    axis = np.cross([0, 0, 1.], pen_dir); s = np.linalg.norm(axis)
    pq = Rotation.from_rotvec(axis / s * np.arctan2(s, pen_dir[2])).as_quat()
    d.mocap_quat[0] = [pq[3], pq[0], pq[1], pq[2]]
    jt = [m.joint(f"joint_1_{k}").qposadr[0] for k in range(1, 5)]
    ji = [m.joint(f"joint_2_{k}").qposadr[0] for k in range(1, 5)]
    lim_t = m.jnt_range[[m.joint(f"joint_1_{k}").id for k in range(1, 5)]]
    lim_i = m.jnt_range[[m.joint(f"joint_2_{k}").id for k in range(1, 5)]]
    curl = np.array([0, 1.6, 1.3, 1.1])
    for f in (3, 4, 5):
        c = curl if f != 5 else [0.3, 1.4, 1.3, 1.1]
        for k in range(4):
            d.qpos[m.joint(f"joint_{f}_{k + 1}").qposadr[0]] = c[k]
    tip_geoms = {f: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"link_{f}_4").id and m.geom_group[g] == 3] for f in (1, 2)}
    finger_geoms = {f: [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.body(m.geom_bodyid[g]).name in (f"link_{f}_3", f"link_{f}_4")] for f in (1, 2)}
    hand_geoms = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.body(m.geom_bodyid[g]).name.startswith(("link_", "palm"))]
    pen_g, table_g = m.geom("pen").id, m.geom("table").id
    fromto = np.zeros(6)

    def place(x):
        off = x[:3]  # palm position relative to the pen centre
        d.mocap_pos[0] = -off
        d.mocap_pos[1] = -off + [0, 0, -PEN_RADIUS - 0.01]
        d.qpos[jt] = x[3:7]; d.qpos[ji] = x[7:11]
        mujoco.mj_forward(m, d)

    def dist(g1, g2):
        return mujoco.mj_geomDistance(m, d, g1, g2, 0.05, fromto), fromto.copy()

    def residual(x):
        place(x)
        r = []
        centre = d.mocap_pos[0]
        for f in (1, 2):
            dd, ft = min((dist(g, pen_g) for g in tip_geoms[f]), key=lambda t: t[0])
            r.append(200 * (dd + 0.0003))                      # touching
            rel = ft[3:] - centre                              # contact point on the pen
            elev = np.degrees(np.arcsin(np.clip(rel[2] / PEN_RADIUS, -1, 1)))
            r.append(0.5 * max(0, elev - height_deg))          # not over the top
            r.append(300 * (ft[3:] - centre) @ pen_dir - 300 * lateral)  # centred along the pen
            distal = d.xmat[m.body(f"link_{f}_4").id].reshape(3, 3)[:, 0]
            r.append(w_axis * max(0, np.degrees(np.arccos(np.clip(-distal[2], -1, 1))) - max_tilt))  # pointing into the table
        for g1 in finger_geoms[1]:
            for g2 in finger_geoms[2]:
                r.append(300 * max(0, 0.0015 - dist(g1, g2)[0]))  # thumb and index clear of each other
        for g in hand_geoms:
            dt, _ = dist(g, table_g)
            r.append(300 * max(0, 0.0015 - dt))               # clear of the table
        r.extend(0.02 * (x[3:] - x0[3:]))
        return np.array(r)

    if seed is None:
        seed = np.r_[-R @ np.array([0.105, 0.03, 0.064]), [1.4, -1.2, -1.1, 1.5], [0.14, 1.9, 0.56, -0.8]]
    x0 = np.asarray(seed, float)
    lo = np.r_[[-0.2, -0.2, 0.0], lim_t[:, 0], lim_i[:, 0]]
    hi = np.r_[[0.2, 0.2, 0.2], lim_t[:, 1], lim_i[:, 1]]
    from scipy.optimize import differential_evolution
    lo_s = np.maximum(lo, np.r_[x0[:3] - 0.04, lo[3:]])
    hi_s = np.minimum(hi, np.r_[x0[:3] + 0.04, hi[3:]])
    best = differential_evolution(lambda x: float(np.sum(residual(x) ** 2)), list(zip(lo_s, hi_s)), x0=np.clip(x0, lo_s, hi_s),
                                  maxiter=iters, popsize=16, tol=1e-10, seed=0, polish=False, init="sobol")
    fit = least_squares(residual, best.x, bounds=(lo_s, hi_s), max_nfev=200, diff_step=1e-3, x_scale="jac")
    if np.sum(fit.fun ** 2) > best.fun:
        fit.x = best.x
    place(fit.x)
    info = {}
    for f in (1, 2):
        dd, ft = min((dist(g, pen_g) for g in tip_geoms[f]), key=lambda t: t[0])
        info[f"d{f}_mm"] = round(1000 * dd, 2)
        info[f"tilt{f}_deg"] = round(float(np.degrees(np.arccos(-d.xmat[m.body(f"link_{f}_4").id].reshape(3, 3)[2, 0]))), 1)
        info[f"elev{f}_deg"] = round(float(np.degrees(np.arcsin(np.clip((ft[5] - d.mocap_pos[0][2]) / PEN_RADIUS, -1, 1)))), 1)
    info["fingers_mm"] = round(1000 * min(dist(a, b)[0] for a in finger_geoms[1] for b in finger_geoms[2]), 2)
    info["table_mm"] = round(1000 * min(dist(g, table_g)[0] for g in hand_geoms), 2)
    q = np.zeros(20)
    for f in range(1, 6):
        for k in range(4):
            q[4 * (f - 1) + k] = d.qpos[m.joint(f"joint_{f}_{k + 1}").qposadr[0]]
    def gap(qh, f):
        x = fit.x.copy(); x[3:7] = qh[0:4]; x[7:11] = qh[4:8]
        place(x)
        return min(dist(g, pen_g)[0] for g in tip_geoms[f])

    def shift(qh, delta):
        """Thumb and index moved `delta` metres along their own pen-distance gradient (+ opens)."""
        qh = np.array(qh, float)
        for f, sl in ((1, slice(0, 4)), (2, slice(4, 8))):
            target = gap(qh, f) + delta
            for _ in range(8):
                g0 = gap(qh, f)
                J = np.array([(gap(np.where(np.arange(20) == sl.start + k, qh + 1e-3, qh), f) - g0) / 1e-3 for k in range(4)])
                if abs(target - g0) < 2e-4 or J @ J < 1e-12:
                    break
                qh[sl] = qh[sl] + J * (target - g0) / (J @ J)
        place(fit.x)
        return qh
    info["shift"] = shift
    if verbose:
        print({k: v for k, v in info.items() if k != "shift"}, fit.x.round(3))
        print("open", shift(q, 0.012)[:8].round(3), "squeeze", shift(q, -0.003)[:8].round(3))
    return fit.x[:3].copy(), q, info


if __name__ == "__main__":
    synthesize(verbose=True)


def tip_clearance(c, fingers=(0, 1)):
    """Lowest point of the given fingertip meshes above the table [m], from measured joints (CPU kinematics)."""
    from dexagent.workcell import TABLE_Z
    kin = c.kin
    q = c.sim.state.joint_q.numpy()
    kin.set(arm=q[c.arm_idx], hand=q[c.hand_idx])
    m, d = kin.model, kin.data
    if not hasattr(kin, "_tip_meshes"):
        kin._tip_meshes = {}
        for f in range(5):
            body = m.body(f"link_{f + 1}_4").id
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_type[g] == 7 and m.geom_group[g] == 3]
            kin._tip_meshes[f] = [(g, m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]) for g in gs]
    low = np.inf
    for f in fingers:
        for g, v in kin._tip_meshes[f]:
            z = (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2]
            low = min(low, float(z.min()))
    return low - TABLE_Z
