"""Executable graph-as-policy skill; adapters implement its contact commands."""
class SlipPolicyGraph:
 def __init__(self):
  self.routes={'acquire':{'ready':'drive'},'drive':{'stroke_complete':'unload'},'unload':{'light_contact':'slide_return'},'slide_return':{'reset_complete':'acquire'},'seat':{}}
  self.schema={'version':3,'meta':{'name':'loaded_slip_screw_skill','policy':'Executable Python nodes with observation-conditioned exits','inputs':['pad_loads','relative_slip','screw_rotation','screw_depth','cap_load']},'edges':[['START','acquire']],'nodes':{n:{'type':'code','ref':'SlipPolicyGraph.'+n}for n in self.routes},'conditional_edges':{n:{'router_field':'exit','mapping':dict(r,seated='seat',guard_failed='abort')}for n,r in self.routes.items()}}
 def advance(self,c,o):
  s=c.s;loads=getattr(c,"guard_loads",s.handleloads);radius=float(s.m.geom('handle').size[0]);mu=float(s.m.geom_friction[s.tipids[0],0]);resistance=float(s.m.dof_frictionloss[s.anglev]);drag=mu*radius*sum(loads)
  if getattr(s,'thread_parameters',None):
   import math
   p=s.thread_parameters;lead=p['pitch_m']/(2*math.pi*p['radius_m']);friction=p['friction']/math.cos(math.radians(p['half_flank_angle_deg']))
   # Conservative lowering/backdrive capacity of the declared thread.
   # Ignore screwdriver weight, which the side pads may partly support.
   resistance=max(0.,s.index_cap_load+.008*9.81)*p['radius_m']*max(0.,friction-lead)/(1+friction*lead)
  state=dict(loads=loads,drag=drag,resistance=resistance,t=o['t']);exit=getattr(self,c.phase)(c,state)
  if exit:c.transition(self.routes[c.phase][exit],o)
  return drag,resistance
 def acquire(self,c,x):
  if x['t']-c.start>.5 and min(x['loads'])>.1 and max(abs(v-c.drive_load)for v in x['loads'])<.15 and c.s.index_cap_load>.1:return 'ready'
 def drive(self,c,x):
  c.angle=min(c.amplitude,c.angle+c.speed*.02)
  if c.angle>=c.amplitude:return 'stroke_complete'
 def unload(self,c,x):
  c.ready=c.ready+.02 if min(x['loads'])>.005 and x['drag']<.8*x['resistance']else 0.
  if c.ready>.1:return 'light_contact'
 def slide_return(self,c,x):
  if min(x['loads'])>.005 and x['drag']<x['resistance']:c.angle=max(0,c.angle-c.speed*.02)
  if c.angle<=0:c.cycles+=1;return 'reset_complete'
 def seat(self,c,x):return None
