"""Keep an already acquired index-cap contact force limited during transport/spin."""
import numpy as np
import mujoco
class CapLoadServo:
 def __init__(self,s,load=.6):self.s=s;self.load=load;self.local=s.local[1].copy()
 def command(self):
  s=self.s;b=s.bids[1];R=s.d.xmat[b].reshape(3,3);points=[];weights=[]
  for k,c in enumerate(s.d.contact):
   if {s.m.geom(c.geom1).name,s.m.geom(c.geom2).name}=={'cap','finger_tip_2'}:
    f=np.zeros(6);mujoco.mj_contactForce(s.m,s.d,k,f)
    if f[0]>.001:points.append(c.pos.copy());weights.append(f[0])
  if weights:self.local+=.2*(R.T@(np.average(points,axis=0,weights=weights)-s.d.xpos[b])-self.local)
  point=s.d.xpos[b]+R@self.local;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];target=s.d.xpos[s.tool]+axis*.043
  J=np.zeros((3,s.m.nv));K=J.copy();mujoco.mj_jac(s.m,s.d,J,None,point,b);mujoco.mj_jac(s.m,s.d,K,None,point,s.cap);v=(J-K)@s.d.qvel;e=target-point;et=e-axis*np.dot(e,axis);vt=v-axis*np.dot(v,axis)
  fn=np.clip(self.load+.4*(self.load-s.index_cap_load)+5*np.dot(v,axis),.02,1.5);F=150*et-3*vt-axis*fn;cols=s.vi[4:8];tau=J[:,cols].T@F;kp=s.m.actuator_gainprm[4:8,0];kv=-s.m.actuator_biasprm[4:8,2]
  q=s.d.qpos[s.qi[4:8]]+(tau+kv*s.d.qvel[cols])/kp;s.d.ctrl[4:8]=np.clip(q,s.m.jnt_range[s.jids[4:8],0]+1e-5,s.m.jnt_range[s.jids[4:8],1]-1e-5)
 def step(self,q,cart,listener):
  s=self.s;s.apply_control(q,cart);events=[]
  for _ in range(20):
   self.command();mujoco.mj_step(s.m,s.d);mujoco.mj_forward(s.m,s.d);s.measure();listener(s.observe())
   if s.peak>12:events.append('force_limit')
   if s.pen>.0006:events.append('penetration')
   if s.forbidden>.3:events.append('shaft_contact')
   if s.rigid_tool_load>.2:events.append('rigid_link_contact')
   if events:break
  return events
from screwdriver_lab.pivot_spin_controller import CapturedSlipController
class FullContactController(CapturedSlipController):
 def __init__(self,s,cap_servo,**options):super().__init__(s,**options);self.cap_servo=cap_servo
 def torques(self):super().torques();self.cap_servo.command()
