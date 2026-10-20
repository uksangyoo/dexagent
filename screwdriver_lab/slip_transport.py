"""Force-limited index support during articulated pickup and reorientation."""
import numpy as np,mujoco
class CapTransportServo:
 def __init__(self,s):self.s=s;self.local=s.local[1].copy();self.target=None;self.load=.4;self.velocity=np.zeros(3)
 def set_target(self,p):
  if self.target is not None:self.velocity=.5*self.velocity+.5*(p-self.target)/.02
  self.target=p.copy()
 def step(self,q,cart,listener):
  s=self.s;s.apply_control(q,cart);events=[];cols=s.vi[4:8];b=s.bids[1]
  for _ in range(20):
   points=[];weights=[]
   for k,c in enumerate(s.d.contact):
    names=[s.m.geom(g).name for g in [c.geom1,c.geom2]]
    if set(names)=={'cap','finger_tip_2'}:
     f=np.zeros(6);mujoco.mj_contactForce(s.m,s.d,k,f)
     if f[0]>.001:points.append(c.pos.copy());weights.append(f[0])
   R=s.d.xmat[b].reshape(3,3)
   if weights:self.local+=.2*(R.T@(np.average(points,axis=0,weights=weights)-s.d.xpos[b])-self.local)
   point=s.d.xpos[b]+R@self.local;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];J=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,J,None,point,b);v=J@s.d.qvel;K=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,K,None,point,s.tool);relative=v-K@s.d.qvel;e=self.target-point;et=e-axis*np.dot(e,axis);tracking=v-self.velocity;vt=tracking-axis*np.dot(tracking,axis);fn=np.clip(self.load+5*np.dot(relative,axis),.05,1.)
   F=300*et-5*vt-axis*fn;tau=J[:,cols].T@F;kp=s.m.actuator_gainprm[4:8,0];kv=-s.m.actuator_biasprm[4:8,2];qt=s.d.qpos[s.qi[4:8]]+(tau+kv*s.d.qvel[cols])/kp;s.d.ctrl[4:8]=np.clip(qt,s.m.jnt_range[s.jids[4:8],0]+1e-5,s.m.jnt_range[s.jids[4:8],1]-1e-5)
   mujoco.mj_step(s.m,s.d);mujoco.mj_forward(s.m,s.d);s.measure();o=s.observe();listener(o)
   if s.peak>12:events.append('force_limit')
   if s.pen>.0006:events.append('penetration')
   if s.forbidden>.3:events.append('shaft_contact')
   if events:break
  return events
