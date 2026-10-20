"""A velocity-loop refinement: reject persistent tangential contact load."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,mujoco
import screwdriver_lab.controlled_slip as cs
from screwdriver_lab.slip_compare import BASE,METHODS
class VelocityIntegral(cs.SlipPairedController):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.tangent_integral=np.zeros(2);self.integral_gain=500.
 def transition(self,phase,o):
  super().transition(phase,o)
  if phase in ['unload','acquire']:self.tangent_integral[:]=0
 def torques(self):
  super().torques();s=self.s;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];center=s.d.xpos[s.tool];radius=float(s.m.geom('handle').size[0])
  for slot,i in enumerate([0,2]):
   b=s.bids[i];point=s.d.xpos[b]+s.d.xmat[b].reshape(3,3)@self.live_local[slot];radial=point-center;radial-=axis*np.dot(radial,axis);radial/=max(1e-9,np.linalg.norm(radial));tangent=-np.cross(axis,radial);J=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,J,None,point,b);error=np.radians(self.command_speed[slot])*radius-np.dot(J@s.d.qvel,tangent)
   self.tangent_integral[slot]=np.clip(self.tangent_integral[slot]+self.integral_gain*error*.001,-.3,.3)if self.phase in ['drive','slide_return']else 0.
   cols=s.vi[4*i:4*i+4];s.d.ctrl[4*i:4*i+4]+=J[:,cols].T@(tangent*self.tangent_integral[slot])/2.;s.d.ctrl[4*i:4*i+4]=np.clip(s.d.ctrl[4*i:4*i+4],s.m.jnt_range[s.jids[4*i:4*i+4],0]+1e-5,s.m.jnt_range[s.jids[4*i:4*i+4],1]-1e-5)
if __name__=='__main__':
 cs.SlipPairedController=VelocityIntegral;r=cs.run('slip-velocity-integral',**{**BASE,**METHODS['velocity']});p=Path(__file__);sha=hashlib.sha256(p.read_bytes()).hexdigest();r['source_sha256'][p.name]=sha;(cs.OUT/'source_snapshots'/(sha+'_'+p.name)).write_bytes(p.read_bytes());r['parameters'].update(controller_class='VelocityIntegral',tangential_integral_gain=500,integral_force_limit_N=.3);(cs.OUT/(r['id']+'.json')).write_text(json.dumps(r))
