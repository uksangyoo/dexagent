"""Guarded thumb/middle rolling stroke with index support through compact resets."""
import numpy as np
class ReferenceCycle:
 def __init__(self,amplitude=35,release=.0005,return_load_limit=.1):
  self.return_load_limit=return_load_limit;self.amplitude=amplitude;self.release=release;self.clearance=release;self.phase='drive';self.start=None;self.angle=0.;self.rad=np.array([-.0006,-.0006]);self.phase_angle=0.;self.turn_start=0.;self.cycles=0;self.trace=[]
 def tick(self,t,o):
  if self.start is None:self.start=t;self.turn_start=o['screw_angle_deg']
  dt=t-self.start;loads=np.array(o['handle_forces_N']);old=self.phase
  if self.phase=='drive':
   self.angle=min(self.amplitude,dt*self.amplitude/1.2);self.rad=np.clip(self.rad-.0005*(.8-loads)*.02,-.004,.003);radial=self.rad.copy()
   if o['screw_angle_deg']-self.turn_start>=25 or dt>=1.4:self.phase='unload';self.phase_angle=self.angle;self.clearance=self.release
  elif self.phase=='unload':
   self.angle=self.phase_angle;self.clearance=float(np.clip(self.clearance+(.004*.02 if max(loads)>self.return_load_limit else 0.),self.release,.003));radial=np.array([self.clearance]*2)
   if max(loads)<self.return_load_limit and dt>=.08:self.phase='return'
   elif dt>2.:self.phase='blocked'
  elif self.phase=='return':
   self.clearance=float(np.clip(self.clearance+(.004*.02 if max(loads)>self.return_load_limit else 0.),self.release,.003));radial=np.array([self.clearance]*2)
   if max(loads)<self.return_load_limit:self.angle=max(0,self.angle-self.amplitude/.4*.02)
   if self.angle<=0:self.phase='reacquire'
   elif dt>3.:self.phase='blocked'
  elif self.phase=='blocked':
   radial=np.array([self.clearance]*2)
  else:
   self.angle=0.;radial=np.array([self.clearance*(1-min(1,dt/.2))-.0012*min(1,dt/.2)]*2)
   if (dt>.2 and min(loads)>.1)or dt>.8:self.phase='drive';self.turn_start=o['screw_angle_deg'];self.rad=radial.copy();self.cycles+=1
  if self.phase!=old:self.trace.append(dict(t=t,from_node=old,to=self.phase,handle_loads_N=loads.tolist(),index_cap_N=o['index_cap_load_N']));self.start=t
  return self.angle,radial,old

class ReferenceRollingCycle:
 """Overlapping drive strokes: one side finger carries torque during the other's light return."""
 def __init__(self,amplitude=30,period=2.4):
  self.amplitude=amplitude;self.period=period;self.start=None;self.bias=np.zeros((2,2));self.phase='rolling';self.cycles=0;self.trace=[];self.last_modes=None
 def tick(self,t,o):
  if self.start is None:self.start=t
  angles=[];radial=[];modes=[];loads=o['handle_forces_N']
  for i in range(2):
   p=((t-self.start)/self.period+.325+i*.5)%1;drive=p<.65;angle=-self.amplitude/2+self.amplitude*p/.65 if drive else self.amplitude/2-self.amplitude*(p-.65)/.35;target=.8 if drive else .04;slot=0 if drive else 1
   self.bias[i,slot]=np.clip(self.bias[i,slot]-.001*(target-loads[i])*.02,-.004,.003);radial.append(float((-.001 if drive else .001)+self.bias[i,slot]));angles.append(float(angle));modes.append('drive'if drive else'light_return')
  self.cycles=int((t-self.start)/self.period)
  if modes!=self.last_modes:self.trace.append(dict(t=t,modes=modes,handle_loads_N=list(loads),index_cap_N=o['index_cap_load_N']));self.last_modes=modes.copy()
  return angles,np.array(radial),modes
