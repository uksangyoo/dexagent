"""52 independently actuated joints; manufacturer left/right hand geometry."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec, GripperSpec
from robosimstudio.robots.registry import EMBODIMENTS, GRIPPERS

HERE = Path(__file__).resolve().parent
ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HAND = tuple(f"joint_{i}_{j}" for i in range(1,6) for j in range(1,5))


class Kinematics:
    """CPU kinematics only. No alternate physics or simulated object writes."""
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(HERE / "assets/bimanual_ur5e_tesollo.xml"))
        self.data = mujoco.MjData(self.model)
        self.arm_q = {s: np.array([self.model.joint(f"{s}_{j}").qposadr[0] for j in ARM]) for s in ("left","right")}

    def solve(self, side, xyz, rotation, seed):
        idx = self.arm_q[side]
        bid = self.model.body(f"{side}_palm").id
        def residual(q):
            self.data.qpos[idx] = q
            mujoco.mj_kinematics(self.model,self.data)
            r = self.data.xmat[bid].reshape(3,3)
            return np.r_[self.data.xpos[bid]-xyz, .18*Rotation.from_matrix(rotation@r.T).as_rotvec()]
        joint_ids=[self.model.joint(f"{side}_{j}").id for j in ARM]
        limits=self.model.jnt_range[joint_ids]
        result = least_squares(residual,seed, bounds=(limits[:,0],limits[:,1]), max_nfev=150)
        if np.linalg.norm(residual(result.x)) > .003:
            raise ValueError(f"{side} IK residual {residual(result.x)}")
        return result.x


def hand_rotation(side):
    s = 1 if side == "left" else -1
    return np.array([[0,-s,0],[-s,0,0],[0,0,-1.]])


def register():
    kin = Kinematics()
    homes = {}
    for side, s in (("left",1),("right",-1)):
        homes[side] = tuple(kin.solve(side,np.array([1.41,s*.38,1.15]),hand_rotation(side),[1.57,-1.4,1.5,-1.5,-1.57,0]))
    g = GripperSpec(name="tesollo_crate",finger_names=tuple("{arm}_"+n for n in HAND),
                    finger_open_q=(0.,)*20, finger_closed_q=(0.,)*20, linear_travel=False,
                    tcp_body="{arm}_palm", finger_bodies=tuple((f"{{arm}}_link_{i}_4",) for i in range(1,6)))
    if g.name in GRIPPERS.names(): GRIPPERS.replace(g.name,g)
    else: GRIPPERS.register(g.name,g)
    spec = EmbodimentSpec(name="ur5e_tesollo_bimanual_crate",asset=AssetSpec(path=str(HERE/"assets/bimanual_ur5e_tesollo.xml")),
                         collapse_fixed_joints=False, enable_self_collisions=True, show_collision_shapes=False,
                         arms=tuple(ArmSpec(name=s,joint_names=tuple(f"{s}_{n}" for n in ARM),
                                            base_body_name=f"{s}_base",ee_body_name=f"{s}_palm",
                                            gripper=g.name,home_q=homes[s]) for s in ("left","right")))
    if spec.name in EMBODIMENTS.names(): EMBODIMENTS.replace(spec.name,spec)
    else: EMBODIMENTS.register(spec.name,spec)
    return spec.name
