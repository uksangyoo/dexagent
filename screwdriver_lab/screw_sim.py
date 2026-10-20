"""Contact-driven screw insertion. Only fingers and axial carriage are actuated."""
import json,hashlib,time,argparse
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from screwdriver_lab.screw_scene import ROOT,OUT,build,CENTER,DEPTH,PITCH

class ScrewSim:
 def __init__(self,case=None):
  self.case=case or {};g=json.loads((OUT/'precision_grasp.json').read_text());self.xml=build(self.case,g['palm']);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m)
  self.jids=np.array([self.m.joint(f'joint_{i}_{j}').id for i in range(1,6) for j in range(1,5)]);self.qi=self.m.jnt_qposadr[self.jids];self.vi=self.m.jnt_dofadr[self.jids]
  self.d.qpos[self.qi]=g['q'];self.target=np.array(g['q']);self.d.ctrl[:20]=self.target;self.d.ctrl[20]=0
  self.tool=self.m.body('screwdriver').id;self.cap=self.m.body('swivel_cap').id;self.screw=self.m.body('screw').id
  self.depthq=self.m.joint('screw_depth').qposadr[0];self.angleq=self.m.joint('screw_angle').qposadr[0];self.anglev=self.m.joint('screw_angle').dofadr[0]
  self.tipids=[self.m.geom(f'finger_tip_{i}').id for i in range(1,6)];self.bids=[self.m.body(f'link_{i}_4').id for i in range(1,6)]
  self.loads=np.zeros(5);self.axial=0;self.slot_force=0;self.pen=0;self.forbidden=0;self.peak=0;self.peakpen=0;self.last_yaw=0;self.tool_angle=0
  mujoco.mj_forward(self.m,self.d);self.capture();self.measure();self.home=self.target.copy();self.feed=0
 def capture(self):
  self.local=[];self.homepatch=[];self.normals=[]
  for i in range(3):
   g=self.m.geom('cap' if i==0 else 'handle').id;line=np.zeros(6);mujoco.mj_geomDistance(self.m,self.d,self.tipids[i],g,.3,line)
   b=self.bids[i];self.local.append(self.d.xmat[b].reshape(3,3).T@(line[:3]-self.d.xpos[b]));self.homepatch.append(line[3:]-self.d.xpos[self.tool])
   n=np.array([0.,0.,1.]) if i==0 else np.r_[line[3:5]-self.d.xpos[self.tool,:2],0.];n/=np.linalg.norm(n);self.normals.append(n)
 def ik(self,angle=0,release=0,compression=.0007):
  rot=Rotation.from_euler('z',-angle,degrees=True).as_matrix();q=self.d.qpos[self.qi].copy();center=CENTER.copy();center[2]-=self.d.qpos[self.depthq]
  for i in range(3):
   b=self.bids[i];point=self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@self.local[i]
   if i==0:target=center+self.homepatch[i]-self.normals[i]*.0018
   else:target=center+rot@(self.homepatch[i]+self.normals[i]*(release-compression))
   jac=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jac,None,point,b);J=jac[:,self.vi[i*4:i*4+4]];e=target-point;e*=min(1.,.0015/max(1e-12,np.linalg.norm(e)))
   dq=J.T@np.linalg.solve(J@J.T+2e-6*np.eye(3),e);q[i*4:i*4+4]+=np.clip(.6*dq,-.03,.03)
  return self.bound(q)
 def bound(self,q):return np.clip(np.clip(q,self.d.qpos[self.qi]-.18,self.d.qpos[self.qi]+.18),self.m.jnt_range[self.jids,0]+1e-5,self.m.jnt_range[self.jids,1]-1e-5)
 def measure(self):
  self.loads[:]=0;self.axial=0;self.slot_force=0;self.pen=0;self.forbidden=0;self.seat=0
  for k,c in enumerate(self.d.contact):
   ids=[c.geom1,c.geom2];bs=[int(self.m.geom_bodyid[x]) for x in ids];fn=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,fn);load=max(0,float(fn[0]));self.pen=max(self.pen,-c.dist)
   if self.tool in bs or self.cap in bs:
    target=self.tool if self.tool in bs else self.cap;other=1-bs.index(target);body=self.m.body(bs[other]).name;gn=self.m.geom(ids[1-other]).name
    if body.startswith('link_'):
     finger=int(body.split('_')[1])-1;self.loads[finger]+=load
     allowed=gn in ('cap','handle')
     if not allowed:self.forbidden+=load
     if gn=='cap':
      world_force=c.frame.reshape(3,3).T@fn[:3];on_tool=world_force if bs[1]==target else -world_force;self.axial+=max(0.,-float(on_tool[2]))
    if self.screw in bs:self.slot_force+=load
   if self.screw in bs and any(self.m.geom(g).name.startswith('workpiece_') for g in ids):self.seat+=load
  R=self.d.xmat[self.tool].reshape(3,3);yaw=-np.arctan2(R[1,0],R[0,0]);self.tool_angle+=float(np.arctan2(np.sin(yaw-self.last_yaw),np.cos(yaw-self.last_yaw)));self.last_yaw=yaw
  self.peak=max(self.peak,max(self.loads));self.peakpen=max(self.peakpen,self.pen)
 def observe(self):
  R=self.d.xmat[self.tool].reshape(3,3);tilt=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))));bit=self.d.xpos[self.tool]+R@np.array([0,0,-.123]);head=self.d.xpos[self.screw]
  lateral=float(np.linalg.norm(bit[:2]-head[:2]));height=float(bit[2]-head[2]);engaged=lateral<.0012 and -.0005<height<.003 and tilt<8 and self.slot_force>.05
  return dict(t=float(self.d.time),screw_angle_deg=float(np.degrees(self.d.qpos[self.angleq])),tool_angle_deg=float(np.degrees(self.tool_angle)),depth_mm=float(self.d.qpos[self.depthq]*1000),axial_force_N=self.axial,slot_force_N=self.slot_force,seat_force_N=self.seat,finger_forces=self.loads.tolist(),tilt_deg=tilt,bit_lateral_mm=lateral*1000,bit_height_mm=height*1000,engaged=bool(engaged),penetration_mm=self.pen*1000,forbidden_N=self.forbidden,screw_speed_rad_s=float(abs(self.d.qvel[self.anglev])),feed_mm=self.d.qpos[0]*1000)
 def step(self,q,listener=None,force_target=.6):
  self.target=self.bound(q);self.d.ctrl[:20]=self.target
  # Common force-controlled axial feed; no wrist rotation is available.
  self.feed=float(np.clip(self.feed+.00025*(force_target-self.axial)*.02,-.001,.008));self.d.ctrl[20]=self.feed
  events=[]
  for _ in range(20):
   mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);self.measure();o=self.observe()
   if self.peak>8:events.append('force_limit')
   if self.pen>.0005:events.append('penetration')
   if self.forbidden>.2:events.append('unintended_contact')
   if o['tilt_deg']>8:events.append('alignment_lost')
   if o['bit_lateral_mm']>1.2 or o['bit_height_mm']>3:events.append('engagement_lost')
   if listener:listener(o)
   if events:break
  return events

def trial(method='contact',case=None,verbose=True,timeout=35):
 from screwdriver_lab.screw_control import Adapter
 s=ScrewSim(case);adapter=Adapter(method);frames=[];qs=[];trace=[];phase='press';nextframe=0.;held=0;stage_start=0.;phase_angle=0.;angle=0.;release=0.;cycle=0;cycle_start=0.;startangle=0.;outcome='timeout';lastprogress=0.;bestdepth=0.
 def capture(o,force=False):
  nonlocal nextframe,held
  if force or o['t']+1e-9>=nextframe:
   if not frames or o['t']>frames[-1]['t']+1e-10:frames.append(dict(o,phase=phase));qs.append(s.d.qpos.copy())
   nextframe+=1/30
  ok=o['depth_mm']>=.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and .3<o['axial_force_N']<5 and o['seat_force_N']>.1
  held=held+.001 if ok else 0
 capture(s.observe(),True)
 for k in range(int(timeout/.02)):
  t=float(s.d.time);o=s.observe()
  if o['depth_mm']>bestdepth+.01:bestdepth=o['depth_mm'];lastprogress=t
  if o['depth_mm']>=.985 and phase!='seat':phase='seat';stage_start=t
  if phase=='press':
   angle=0;release=0
   if t>=1.5:
    if not o['engaged'] or min(o['finger_forces'][1:3])<.02 or o['axial_force_N']<.1:outcome='acquisition_failed';break
    phase='turn';stage_start=t;startangle=o['screw_angle_deg'];cycle_start=t;lastprogress=t
  elif phase=='turn':
   angle=min(30,(t-stage_start)*35);release=0
   if method=='object':angle=float(np.clip(angle+.5*(min(360,(t-stage_start)*35)-(o['screw_angle_deg']-startangle)),0,40))
   if method=='contact' and (o['screw_angle_deg']-startangle>=22 or t-stage_start>1.4):phase='release';stage_start=t;phase_angle=angle
   elif method!='contact' and t-lastprogress>2.:outcome='stalled';break
  elif phase=='release':
   angle=phase_angle;release=min(.004,(t-stage_start)*.02)
   if t-stage_start>=.25:phase='return';stage_start=t
  elif phase=='return':
   angle=phase_angle*max(0,1-(t-stage_start)/.35);release=.004
   if t-stage_start>=.35:phase='regrasp';stage_start=t
  elif phase=='regrasp':
   angle=0;release=.004*max(0,1-(t-stage_start)/.35)
   if t-stage_start>=.45:
    cycle+=1;phase='turn';stage_start=t;startangle=o['screw_angle_deg'];cycle_start=t
  elif phase=='seat':release=0
  if k%10==0:trace.append(dict(t=t,phase=phase,target_roll_deg=angle,release_mm=release*1000,cycle=cycle,reason={'press':'Maintain cap pressure and establish bit engagement','turn':'Turn handle clockwise while monitoring screw advance','release':'Release side fingers; cap and engaged bit support the tool','return':'Return side fingers clear of the handle','regrasp':'Close on fresh handle patches','seat':'Stop turning and verify seating under axial load'}[phase]))
  q=adapter.command(s,angle,release,phase)
  events=s.step(q,capture)
  if verbose and k%50==0:print(method,round(t,2),phase,{key:round(val,3) if isinstance(val,float) else val for key,val in s.observe().items() if key in ['screw_angle_deg','depth_mm','axial_force_N','engaged','finger_forces','tilt_deg']},flush=True)
  if events:outcome=events[0];break
  if held>=.4:outcome='seated';break
 capture(s.observe(),True)
 r=dict(id=f"{(case or {}).get('id','nominal')}-{method}",task='screw_in',method=method,case=(case or {}).get('id','nominal'),outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,trace=trace,metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=s.peakpen*1000,cycles=cycle),parameters=dict(pitch_mm=1,seat_depth_mm=1,friction=(case or {}).get('friction',.9),slot_yaw_deg=(case or {}).get('slot_yaw_deg',0),thread_torque_Nm=(case or {}).get('thread_torque_Nm',.0008)),initial_state_sha256=hashlib.sha256(qs[0].tobytes()).hexdigest())
 return r,np.array(qs),s.xml

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',default='contact');p.add_argument('--timeout',type=float,default=35);a=p.parse_args()
 r,q,xml=trial(a.method,timeout=a.timeout);OUT.mkdir(exist_ok=True,parents=True);(OUT/(r['id']+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(r['id']+'.npz'),qpos=q);(OUT/(r['id']+'.xml')).write_text(xml);print('RESULT',r['outcome'],r['final'],flush=True)
