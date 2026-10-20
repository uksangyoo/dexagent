"""Regulate circumferential motion while allowing compliant axial pad sliding."""
import numpy as np,mujoco
from screwdriver_lab.pivot_spin_controller import CapturedSlipController
class SurfaceSlipController(CapturedSlipController):
 def torques(self):
  s=self.s;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];center=s.d.xpos[s.tool];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex);radius=float(s.m.geom('handle').size[0])
  if not hasattr(self,'live_local'):self.live_local=[v.copy()for v in self.local]
  for slot,i in enumerate([0,2]):
   b=s.bids[i];R=s.d.xmat[b].reshape(3,3)
   if s.handleloads[slot]>.001:self.live_local[slot]+=.2*(R.T@(s.pad_points[slot]-s.d.xpos[b])-self.live_local[slot])
   point=s.d.xpos[b]+R@self.live_local[slot];radial=point-center;radial-=axis*np.dot(radial,axis);n=radial/max(1e-9,np.linalg.norm(radial));tangent=-np.cross(axis,n)
   angle=self.angle
   if hasattr(self,'return_angles')and self.phase in ['unload','slide_return','seat']:angle=self.return_angles[slot]*self.angle/self.amplitude
   phi=self.phi[slot]-np.radians(angle);desired=center+radius*(ex*np.cos(phi)+ey*np.sin(phi))+self.height[slot]*axis
   J=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,J,None,point,b);cols=s.vi[4*i:4*i+4];Ji=J[:,cols];v=J@s.d.qvel;error=desired-point;target=float(self.force_targets[slot]);fn=np.clip(target+self.normal_d*np.dot(v,n),.2*target,1.)
   F=-n*fn+tangent*(150*np.dot(error,tangent)-2*np.dot(v,tangent))+axis*(20*np.dot(error,axis)-2*np.dot(v,axis))
   N=np.eye(4)-Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),Ji);tau=Ji.T@F+N@(-.01*s.d.qvel[cols]);kp=s.m.actuator_gainprm[4*i:4*i+4,0];kv=-s.m.actuator_biasprm[4*i:4*i+4,2];q=s.d.qpos[s.qi[4*i:4*i+4]]+(tau+kv*s.d.qvel[cols])/kp;s.d.ctrl[4*i:4*i+4]=np.clip(q,s.m.jnt_range[s.jids[4*i:4*i+4],0]+1e-5,s.m.jnt_range[s.jids[4*i:4*i+4],1]-1e-5)
