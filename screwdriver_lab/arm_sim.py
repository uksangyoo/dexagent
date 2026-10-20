"""Same hand/controller observations with an articulated UR5e, no Cartesian stages."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.full_motion import FullSim
from screwdriver_lab.arm_scene import attach_arm,ArmIK
from screwdriver_lab.screw_scene import R
class ArmFullSim(FullSim):
 def __init__(self,case=None):
  self.arm_ready=False
  super().__init__(case)
  oldm,oldd=self.m,self.d;bid=oldm.body('palm').id;pos=oldd.xpos[bid].copy();rot=oldd.xmat[bid].reshape(3,3).copy()
  self.xml=attach_arm(self.xml);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m)
  for j in range(oldm.njnt):
   name=oldm.joint(j).name;k=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,name)
   if k<0:continue
   a,b=oldm.jnt_qposadr[j],self.m.jnt_qposadr[k];n=7 if oldm.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1;self.d.qpos[b:b+n]=oldd.qpos[a:a+n]
  self.arm=ArmIK(self.m);self.d.qpos[self.arm.qi]=self.arm.solve(pos,rot,exact=True)
  self.jids=np.array([self.m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)]);self.qi=self.m.jnt_qposadr[self.jids];self.vi=self.m.jnt_dofadr[self.jids]
  for attr,name in [('tool','screwdriver'),('cap','swivel_cap'),('screw','screw')]:setattr(self,attr,self.m.body(name).id)
  self.depthq=self.m.joint('screw_depth').qposadr[0];self.angleq=self.m.joint('screw_angle').qposadr[0];self.anglev=self.m.joint('screw_angle').dofadr[0];self.toq=self.m.joint('tool_free').qposadr[0]
  self.tipids=[self.m.geom(f'finger_tip_{i}').id for i in range(1,6)];self.bids=[self.m.body(f'link_{i}_4').id for i in range(1,6)]
  self.d.ctrl[:20]=self.openq;self.d.ctrl[20:]=self.d.qpos[self.arm.qi];self.arm_ready=True;mujoco.mj_forward(self.m,self.d);self.measure()
  err=np.linalg.norm(self.d.xpos[self.m.body('palm').id]-pos)
  if err>.001:raise RuntimeError(f'UR5e initial IK error {err}')
 def cart_state(self):
  if not self.arm_ready:return super().cart_state()
  b=self.m.body('palm').id;offset=self.d.xpos[b]-self.p0;rot=self.d.xmat[b].reshape(3,3)@R.T
  return np.r_[-offset[2],offset[0],offset[1],np.arctan2(rot[0,2],rot[2,2])]
 def apply_control(self,q,cart):
  pos=self.p0+np.array([cart[1],cart[2],-cart[0]]);rot=getattr(self,'arm_orientation_target',Rot.from_euler('y',cart[3]).as_matrix()@R)
  self.d.ctrl[:20]=self.bound(q);self.d.ctrl[20:]=self.arm.solve(pos,rot,seed=self.d.qpos[self.arm.qi])
  self.arm_goal_pos=pos.copy();self.arm_goal_rot=rot.copy()
 def observe(self):
  o=super().observe()
  if self.arm_ready and hasattr(self,'arm_goal_pos'):
   b=self.m.body('palm').id;o.update(arm_pose_error_mm=float(np.linalg.norm(self.d.xpos[b]-self.arm_goal_pos)*1000),arm_rotation_error_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(self.arm_goal_rot@self.d.xmat[b].reshape(3,3).T).as_rotvec()))))
  if self.arm_ready:o.update(arm_joint_rad=self.d.qpos[self.arm.qi].tolist(),arm_torque_Nm=self.d.actuator_force[20:].tolist(),embodiment='UR5e + Tesollo DG5F-S')
  return o
