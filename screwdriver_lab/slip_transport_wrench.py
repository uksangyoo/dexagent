"""Robot-only Cartesian contact servo for transport; no object constraints."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
class TransportWrenchServo:
 def __init__(self,s):
  self.s=s;self.locals=[v.copy()for v in s.local];self.targets=None;self.velocity=np.zeros((3,3));self.use_cap=False;self.load=2.;self.cap_load=.4;self.reference=None
 def set_target(self,center,rotation,use_cap=False):
  s=self.s
  if self.reference is None:
   R=rotation;self.reference=[];self.ref_rot=[rotation.T@s.d.xmat[b].reshape(3,3) for b in s.bids[:3]]
   for slot,i in enumerate([0,1,2]):
    point=s.d.xpos[s.bids[i]]+s.d.xmat[s.bids[i]].reshape(3,3)@self.locals[i]
    if i!=1 and s.handleloads[0 if i==0 else 1]>.01:point=s.pad_points[0 if i==0 else 1].copy();self.locals[i]=s.d.xmat[s.bids[i]].reshape(3,3).T@(point-s.d.xpos[s.bids[i]])
    self.reference.append(R.T@(point-s.d.xpos[s.tool]))
  targets=np.array([center+rotation@p for p in self.reference]);targets[1]=center+rotation@np.array([0,0,.043]);self.velocity=np.zeros((3,3))if self.targets is None else .5*self.velocity+.5*(targets-self.targets)/.02;self.targets=targets;self.use_cap=use_cap;self.rot_targets=[rotation@v for v in self.ref_rot]
 def step(self,q,cart,listener):
  s=self.s;s.apply_control(q,cart);events=[]
  for _ in range(20):
   axis=s.d.xmat[s.tool].reshape(3,3)[:,2];center=s.d.xpos[s.tool]
   for i in [0,2]:
    slot=0 if i==0 else 1;b=s.bids[i];R=s.d.xmat[b].reshape(3,3);cols=s.vi[4*i:4*i+4]
    if s.handleloads[slot]>.001:self.locals[i]+=.2*(R.T@(s.pad_points[slot]-s.d.xpos[b])-self.locals[i])
    point=s.d.xpos[b]+R@self.locals[i];normal=point-center;normal-=axis*np.dot(normal,axis);normal/=max(1e-9,np.linalg.norm(normal));J=np.zeros((3,s.m.nv));Jr=J.copy();mujoco.mj_jac(s.m,s.d,J,Jr,point,b);v=J@s.d.qvel-self.velocity[i];e=self.targets[i]-point;et=e-normal*np.dot(e,normal);vt=v-normal*np.dot(v,normal);F=-normal*np.clip(self.load+5*np.dot(v,normal),.1,4)+400*et-5*vt
    orientation_error=Rot.from_matrix(self.rot_targets[i]@R.T).as_rotvec();moment=normal*np.clip(.06*np.dot(orientation_error,normal)-.002*np.dot(Jr@s.d.qvel,normal),-.02,.02);tau=J[:,cols].T@F+Jr[:,cols].T@moment;kp=s.m.actuator_gainprm[4*i:4*i+4,0];kv=-s.m.actuator_biasprm[4*i:4*i+4,2];qt=s.d.qpos[s.qi[4*i:4*i+4]]+(tau+kv*s.d.qvel[cols])/kp;s.d.ctrl[4*i:4*i+4]=np.clip(qt,s.m.jnt_range[s.jids[4*i:4*i+4],0]+1e-5,s.m.jnt_range[s.jids[4*i:4*i+4],1]-1e-5)
   mujoco.mj_step(s.m,s.d);mujoco.mj_forward(s.m,s.d);s.measure();o=s.observe();listener(o)
   if s.peak>12:events.append('force_limit')
   if s.pen>.0006:events.append('penetration')
   if s.forbidden>.3:events.append('shaft_contact')
   if events:break
  return events
