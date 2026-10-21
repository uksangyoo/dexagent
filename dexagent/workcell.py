"""Single-arm UR5e + right DG5F-S table workcell for RoboSimStudio/Newton.

`build_assets()` writes `assets/ur5e_dg5fs_right.xml` with mesh paths relative
to that file, so the MJCF is portable. `register()` adds the embodiment to the
RoboSimStudio registries. `Kinematics` is a CPU MuJoCo copy of the same file,
used only for inverse kinematics; it never writes simulated state.
"""
from pathlib import Path
import copy
import os
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
ROBOT_XML = ASSETS / "ur5e_dg5fs_right.xml"
ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HAND = tuple(f"joint_{i}_{j}" for i in range(1, 6) for j in range(1, 5))
FINGERS = ("thumb", "index", "middle", "ring", "little")
TABLE_Z = 0.75
# Robot base on the table's right edge, so the arm enters the phone view from
# the bottom right as the person's arm does; the elbow stays above the frame.
BASE = (0.30, -0.58, TABLE_Z)
BASE_YAW = np.pi / 2
# Default palm yaw: fingers point 40 deg left of the table's long axis, as in the clips.
YAW = 0.7
# Finger motor bounds: kp 3, kd .12, |tau| <= .65 N m per finger joint.
HAND_KP, HAND_KV, HAND_TAU = 3.0, 0.12, 0.65


def _rel(path):
    return os.path.relpath(path, ASSETS)


def _tip_centres():
    """Pad centre of each distal link (link_i_4 frame), from the tip mesh and its manufacturer mount."""
    import trimesh
    src = ROOT / "third-party/tesollo_model/dg5fs/dg5fs_right.xml"
    world = ET.parse(src).getroot()
    out = {}
    for i in range(1, 6):
        mount = next(g for g in world.iter("geom") if g.get("mesh") == f"link_{i}_tip")
        mesh = trimesh.load(src.parent / f"meshes/dg5fs_right/link_{i}_tip.STL", force="mesh")
        # Thumb and little tips mount at x = .0289, index/middle/ring at .018.
        out[i] = np.array(mesh.bounding_box.centroid) + np.array([float(v) for v in mount.get("pos").split()])
    return out


def build_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model="ur5e_dg5fs_right")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", cone="elliptic")
    defaults = ET.SubElement(root, "default")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    acts = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")

    src = ROOT / "third-party/universal_robots_ur5e/ur5e.xml"
    arm = ET.parse(src).getroot()
    for mesh in arm.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", _rel(src.parent / "assets" / mesh.get("file")))
    for item in arm.find("default"):
        defaults.append(item)
    for item in arm.find("asset"):
        asset.append(item)
    base = arm.find("worldbody/body")
    base.set("pos", " ".join(map(str, BASE)))
    # Menagerie base quat is (1 0 0 1): 90 deg yaw. Compose with BASE_YAW.
    yaw = np.pi / 2 + BASE_YAW
    base.set("quat", f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}")
    world.append(base)
    for item in arm.find("actuator"):
        acts.append(item)

    src = ROOT / "third-party/tesollo_model/dg5fs/dg5fs_right.xml"
    hand = ET.parse(src).getroot()
    for mesh in hand.findall("asset/mesh"):
        mesh.set("file", _rel(src.parent / "meshes" / mesh.get("file")))
    for item in hand.find("asset"):
        asset.append(item)
    wrist = base.find(".//body[@name='wrist_3_link']")
    palm = ET.SubElement(wrist, "body", name="palm", pos="0 0.1 0", quat="-1 1 1 1", childclass="hand")
    inertial = ET.parse(src.with_suffix(".urdf")).find("link/inertial")
    ii = inertial.find("inertia")
    ET.SubElement(palm, "inertial", mass=inertial.find("mass").get("value"), pos=inertial.find("origin").get("xyz"),
                  fullinertia=" ".join(ii.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
    hd = ET.SubElement(defaults, "default", {"class": "hand"})
    ET.SubElement(hd, "joint", damping="0.015", armature="0.00002")
    # condim 4 adds torsional friction (.005 m patch) to every hand contact: without
    # it a two-point pinch is a free hinge about the pinch axis.
    ET.SubElement(hd, "geom", condim="4", friction="0.9 0.005 0.0001", rgba="0.12 0.13 0.14 1", solref="0.004 1", solimp="0.95 0.99 0.0005")
    for item in hand.find("worldbody"):
        palm.append(item)
    tips = _tip_centres()
    for i in range(1, 6):
        distal = palm.find(f".//body[@name='link_{i}_4']")
        ET.SubElement(distal, "site", name=f"tip_{i}", pos=" ".join(f"{v:.5f}" for v in tips[i]), size="0.004", rgba="1 .3 .1 1", group="4")
    # Pale fingertip pads, as on the physical hand.
    for parent in palm.iter("body"):
        for geom in list(parent.findall("geom")):
            visual = copy.deepcopy(geom)
            visual.attrib.update(contype="0", conaffinity="0", group="2")
            if geom.get("mesh", "").endswith("_tip"):
                visual.set("rgba", "0.78 0.78 0.76 1")
            geom.set("group", "3")
            parent.append(visual)
    for joint in palm.findall(".//joint"):
        joint.set("actuatorfrcrange", f"{-HAND_TAU} {HAND_TAU}")
        ET.SubElement(acts, "position", name=joint.get("name"), joint=joint.get("name"),
                      kp=str(HAND_KP), kv=str(HAND_KV), forcerange=f"{-HAND_TAU} {HAND_TAU}")
    for parent in base.iter("body"):
        for child in parent.findall("body"):
            ET.SubElement(contact, "exclude", body1=parent.get("name"), body2=child.get("name"))
    # The thumb swivel housing sits inside the palm CAD; the distal fingers still collide.
    ET.SubElement(contact, "exclude", body1="palm", body2="link_1_2")
    ET.SubElement(contact, "exclude", body1="wrist_3_link", body2="link_1_1")
    ET.indent(root)
    ET.ElementTree(root).write(ROBOT_XML, encoding="unicode")
    return ROBOT_XML


def hand_down(yaw=YAW):
    """Palm rotation with fingers forward (+x after yaw) and the palm facing down."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    # Palm frame: +z along the fingers, +x out of the palm face (see `palm_axes`).
    return Rz @ np.array([[0, 0, 1.], [0, 1, 0], [-1, 0, 0]])


def hand_up(yaw=YAW, tilt=0.0):
    """Palm facing up, fingers pointing `yaw` from +x, raised by `tilt` (the palm slopes toward the wrist)."""
    from scipy.spatial.transform import Rotation
    return hand_down(yaw) @ np.diag([-1.0, -1.0, 1.0]) @ Rotation.from_rotvec([0, tilt, 0]).as_matrix()


class Kinematics:
    """CPU kinematics of the generated MJCF. No simulated object state is touched."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        if not ROBOT_XML.exists():
            build_assets()
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_q = np.array([m.joint(j).qposadr[0] for j in ARM])
        self.arm_ids = [m.joint(j).id for j in ARM]
        self.palm = m.body("palm").id
        self.finger_q = [np.array([m.joint(f"joint_{i}_{j}").qposadr[0] for j in range(1, 5)]) for i in range(1, 6)]
        self.tip_site = [m.site(f"tip_{i}").id for i in range(1, 6)]

    def set(self, arm=None, hand=None):
        if arm is not None:
            self.data.qpos[self.arm_q] = arm
        if hand is not None:
            for i in range(5):
                self.data.qpos[self.finger_q[i]] = hand[4 * i:4 * i + 4]
        self.mj.mj_kinematics(self.model, self.data)

    def palm_pose(self, arm):
        self.set(arm=arm)
        return self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy()

    def solve_arm(self, xyz, rotation, seed, weight=0.2, tol=0.003):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        lim = self.model.jnt_range[self.arm_ids]

        def residual(q):
            p, R = self.palm_pose(q)
            return np.r_[p - xyz, weight * Rotation.from_matrix(rotation @ R.T).as_rotvec()]
        seed = np.clip(seed, lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)
        fit = least_squares(residual, seed, bounds=(lim[:, 0], lim[:, 1]), max_nfev=200)
        err = np.linalg.norm(residual(fit.x)[:3])
        if err > tol:
            raise ValueError(f"arm IK position residual {err * 1000:.1f} mm for {xyz}")
        return fit.x

    def tips_in_palm(self, hand):
        """Fingertip pad centres in the palm frame, (5, 3)."""
        self.set(hand=hand)
        p, R = self.data.xpos[self.palm], self.data.xmat[self.palm].reshape(3, 3)
        return np.array([R.T @ (self.data.site_xpos[s] - p) for s in self.tip_site])

    def solve_finger(self, i, target, seed, posture=0.002, direction=None, weight=0.02):
        """Joint angles that put finger i's pad centre at `target` (palm frame).

        With `direction`, also point the distal link's x axis (toward the
        fingertip end) along that palm-frame unit vector, weighted by `weight`.
        """
        from scipy.optimize import least_squares
        m, d = self.model, self.data
        ids = [m.joint(f"joint_{i + 1}_{j}").id for j in range(1, 5)]
        lim = m.jnt_range[ids]
        idx = self.finger_q[i]
        body = m.body(f"link_{i + 1}_4").id

        def residual(q):
            d.qpos[idx] = q
            self.mj.mj_kinematics(m, d)
            P, Rm = d.xpos[self.palm], d.xmat[self.palm].reshape(3, 3)
            r = [Rm.T @ (d.site_xpos[self.tip_site[i]] - P) - target]
            if direction is not None:
                r.append(weight * (Rm.T @ d.xmat[body].reshape(3, 3)[:, 0] - direction))
            r.append(posture * (q - seed))
            return np.concatenate(r)
        seed = np.clip(seed, lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)
        fit = least_squares(residual, seed, bounds=(lim[:, 0], lim[:, 1]), max_nfev=300)
        return fit.x, float(np.linalg.norm(residual(fit.x)[:3]))

def register(name="ur5e_dg5fs_right", home_xyz=(0.45, -0.30, TABLE_Z + 0.30), home_R=None, hand_q=None,
             seed=(0, -1.2, 1.5, -1.9, -1.57, 0)):
    """Register an embodiment over the workcell MJCF. Returns its name.

    `home_xyz`, `home_R` (default `hand_down()`) and `hand_q` (20 finger
    angles, default zeros) are the pose the scene is built in, so a scene can
    start palm-up with an object already resting in the hand. Each distinct
    start pose needs its own `name`.
    """
    from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec, GripperSpec
    from robosimstudio.robots.registry import EMBODIMENTS, GRIPPERS
    if not ROBOT_XML.exists():
        build_assets()
    kin = Kinematics()
    R = hand_down() if home_R is None else np.asarray(home_R, float)
    home = tuple(kin.solve_arm(np.array(home_xyz), R, np.array(seed, float)))
    q = tuple(float(v) for v in (np.zeros(20) if hand_q is None else hand_q))
    g = GripperSpec(name=f"{name}_hand", finger_names=HAND, finger_open_q=q, finger_closed_q=q,
                    linear_travel=False, tcp_body="palm",
                    finger_bodies=tuple((f"link_{i}_4",) for i in range(1, 6)))
    (GRIPPERS.replace if g.name in GRIPPERS.names() else GRIPPERS.register)(g.name, g)
    spec = EmbodimentSpec(name=name, asset=AssetSpec(path=str(ROBOT_XML)), collapse_fixed_joints=False,
                          enable_self_collisions=True, show_collision_shapes=False,
                          arms=(ArmSpec(name="right", joint_names=ARM, base_body_name="base", ee_body_name="palm",
                                        gripper=g.name, home_q=home),))
    (EMBODIMENTS.replace if name in EMBODIMENTS.names() else EMBODIMENTS.register)(name, spec)
    return name


if __name__ == "__main__":
    print(build_assets())
