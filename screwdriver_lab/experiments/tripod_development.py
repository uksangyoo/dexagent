"""Contact-driven pickup, wrist reorientation and alternating in-hand sliding."""
import argparse,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.screw_sim import ScrewSim
from screwdriver_lab.screw_scene import CENTER
from screwdriver_lab.full_scene import OUT,PICK,TRAVEL,build_full

class FullSim(ScrewSim):
 def __init__(self,case=None):
  super().__init__(case);self.xml,g=build_full(case);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.p0=np.array(g['palm'])
  self.jids=np.array([self.m.joint(f'joint_{i}_{j}').id for i in range(1,6) for j in range(1,5)]);self.qi=self.m.jnt_qposadr[self.jids];self.vi=self.m.jnt_dofadr[self.jids];self.d.qpos[self.qi]=g['q'];self.target=np.array(g['q'])
  for attr,name in [('tool','screwdriver'),('cap','swivel_cap'),('screw','screw')]:setattr(self,attr,self.m.body(name).id)
  self.depthq=self.m.joint('screw_depth').qposadr[0];self.angleq=self.m.joint('screw_angle').qposadr[0];self.anglev=self.m.joint('screw_angle').dofadr[0]
  self.tipids=[self.m.geom(f'finger_tip_{i}').id for i in range(1,6)];self.bids=[self.m.body(f'link_{i}_4').id for i in range(1,6)]
  self.cartq=[self.m.joint(n).qposadr[0] for n in ['feed','cart_x','cart_y','wrist_tilt']];self.toq=self.m.joint('tool_free').qposadr[0];self.peak=0;self.peakpen=0;self.pen=0;self.last_yaw=0;self.tool_angle=0;self.stage='reset';self.slip=[0.,0.];self.modes=['open','open'];self.finger_roll=[0.,0.]
  mujoco.mj_forward(self.m,self.d);self.capture();self.home=self.target.copy()
  # Solve open-hand reset geometrically before simulation; never reset a rollout.
  from scipy.optimize import least_squares
  qref=self.d.qpos[self.qi].copy()
  self.grip_offset=(case or {}).get('pickup_offset',-.025)
  self.d.qpos[self.toq:self.toq+3]=CENTER-[0,0,self.grip_offset]
  self.driving_thumb=[x[0].copy()for x in [self.local,self.homepatch,self.normals]]
  pickup=json.loads((OUT/'pickup-thumb.json').read_text());self.local[0]=np.array(pickup['local']);self.homepatch[0]=np.array(pickup['patch']);self.normals[0]=np.array(pickup['normal']);qref[:4]=pickup['q'];self.pickup_thumb=True
  targets=[CENTER+self.homepatch[i]+self.normals[i]*.006 for i in range(3)]
  def residual(q):
   self.d.qpos[self.qi]=q;mujoco.mj_forward(self.m,self.d)
   pts=[self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@self.local[i] for i,b in enumerate(self.bids[:3])]
   pen=sum(max(0,-c.dist-.00001)**2 for c in self.d.contact)
   return np.r_[(np.array(pts)-targets).ravel()*400,np.sqrt(pen)*12000,.03*(q-qref)]
  fit=least_squares(residual,qref,bounds=(self.m.jnt_range[self.jids,0]+1e-5,self.m.jnt_range[self.jids,1]-1e-5),max_nfev=120)
  self.d.qpos[self.qi]=fit.x;mujoco.mj_forward(self.m,self.d)
  self.openq=self.d.qpos[self.qi].copy();self.d.qpos[self.toq:self.toq+3]=PICK;self.d.qpos[self.toq+3:self.toq+7]=Rot.from_euler('y',-90,degrees=True).as_quat(scalar_first=True)
  cart=self.cart_pose(PICK+np.array([-self.grip_offset,0,.05]),-np.pi/2);self.d.qpos[self.cartq]=cart;self.d.ctrl[:20]=self.openq;self.d.ctrl[20:]=cart;self.target=self.openq.copy();mujoco.mj_forward(self.m,self.d);self.peak=0;self.peakpen=0;self.measure()
 def cart_pose(self,center,tilt):
  offset=np.array(center)+Rot.from_euler('y',tilt).apply(self.p0-CENTER)-self.p0
  return np.array([-offset[2],offset[0],offset[1],tilt])
 def ik_pose(self,center,tilt,angles,radial,cap_offset=-.0018):
  q=self.d.qpos[self.qi].copy();R=Rot.from_euler('y',tilt).as_matrix()
  for i in range(3):
   b=self.bids[i];point=self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@self.local[i]
   patch=self.homepatch[i]+self.normals[i]*(cap_offset if i==0 else radial[i-1])
   if i>0:patch=Rot.from_euler('z',-angles[i-1],degrees=True).apply(patch)
   target=np.asarray(center)+R@patch
   if i==0 and self.stage=='shift_grip':target[0]+=getattr(self,'thumb_outside',0.)
   jac=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jac,None,point,b);J=jac[:,self.vi[i*4:i*4+4]];e=target-point;e*=min(1.,.004/max(1e-12,np.linalg.norm(e)))
   q[i*4:i*4+4]+=np.clip(.9*J.T@np.linalg.solve(J@J.T+1e-6*np.eye(3),e),-.06,.06)
  return self.bound(q)
 def observe(self):
  o=super().observe();o.update(feed_mm=float(self.cart_state()[0]*1000),stage=self.stage,tool_center=self.d.xpos[self.tool].tolist(),wrist_tilt_deg=float(np.degrees(self.cart_state()[-1])),slip_mm_s=self.slip.copy(),finger_modes=self.modes.copy(),finger_roll_deg=self.finger_roll.copy())
  return o
 def measure(self):
  if getattr(self,'hydro',None) is not None:self.hydro.measure();return
  super().measure()
  self.cap_axial=self.axial;net=0.
  for k,c in enumerate(self.d.contact):
   bs=[int(self.m.geom_bodyid[g])for g in [c.geom1,c.geom2]]
   if self.tool not in bs:continue
   idx=bs.index(self.tool);other=bs[1-idx]
   if self.m.body(other).name.startswith('link_'):
    f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);world=c.frame.reshape(3,3).T@f[:3];net+=float((world if idx==1 else -world)[2])
  self.axial=max(0.,self.axial-net)
  if not hasattr(self,'slip'):return
  self.slip=[0.,0.];counts=[0,0]
  for c in self.d.contact:
   g=[c.geom1,c.geom2];bs=[int(self.m.geom_bodyid[x]) for x in g]
   if self.tool not in bs:continue
   other=1-bs.index(self.tool);name=self.m.body(bs[other]).name
   if name.startswith(('link_2_','link_3_')) and self.m.geom(g[1-other]).name=='handle':
    i=int(name.split('_')[1])-2;J1=np.zeros((3,self.m.nv));J2=J1.copy();mujoco.mj_jac(self.m,self.d,J1,None,c.pos,bs[other]);mujoco.mj_jac(self.m,self.d,J2,None,c.pos,self.tool);vel=(J1-J2)@self.d.qvel;n=c.frame[:3];self.slip[i]+=float(np.linalg.norm(vel-n*np.dot(vel,n)))*1000;counts[i]+=1
  self.slip=[v/max(1,n) for v,n in zip(self.slip,counts)]
 def cart_state(self):return self.d.qpos[self.cartq].copy()
 def apply_control(self,q,cart):
  self.d.ctrl[:20]=self.bound(q);self.d.ctrl[20:]=cart
 def step_full(self,q,cart,listener):
  self.apply_control(q,cart)
  events=[]
  if getattr(self,'hydro',None) is not None:
   guard=self.hydro.step_batch();o=self.observe();listener(o)
   if guard[0]>12:events.append('force_limit')
   if guard[1]>.001:events.append('penetration')
   if self.forbidden>.3:events.append('shaft_contact')
   return events
  for _ in range(20):
   if getattr(self,'hydro',None) is not None:self.hydro.step()
   else:mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d)
   self.measure();o=self.observe()
   if self.peak>12:events.append('force_limit')
   if self.pen>.0006:events.append('penetration')
   if self.forbidden>.3:events.append('shaft_contact')
   if self.stage in ['turn','seat'] and (o['tilt_deg']>8 or o['bit_lateral_mm']>1.2 or o['bit_height_mm']>3):events.append('engagement_lost')
   listener(o)
   if events:break
  return events

def smooth(v):v=np.clip(v,0,1);return v*v*(3-2*v)
def trial(mode='slide',timeout=70,verbose=True,arm=False,hydro=False):
 from screwdriver_lab.arm_sim import ArmFullSim
 s=ArmFullSim() if arm else FullSim()
 if hydro:
  from screwdriver_lab.hydro_backend import HydroBackend
  s.hydro=HydroBackend(s)
 from screwdriver_lab.graph_policy import GraphPolicy
 policy=GraphPolicy()
 frames=[];qs=[];trace=[];nextframe=0.;phase=policy.current;start=0.;center=PICK.copy();tilt=-np.pi/2;angles=[0.,0.];radial=[.006,.006];cap=.006;outcome='timeout';held=0;rad=np.array([-.0015,-.0015]);turnstart=0.;seatangles=None
 def capture(o,force=False):
  nonlocal nextframe,held
  if force or o['t']>=nextframe-1e-9:
   frames.append(dict(o,phase=phase));qs.append(s.d.qpos.copy());nextframe+=1/30
  ok=o['depth_mm']>=1.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and .15<o['axial_force_N']<5 and o['seat_force_N']>.1
  held=held+.001 if ok else 0
 capture(s.observe(),True)
 for k in range(int(timeout/.02)):
  t=float(s.d.time);o=s.observe();elapsed=t-start;s.stage=phase
  if phase=='approach':
   center=PICK+np.array([0,0,.05*(1-smooth(elapsed/1.5))]);radial=[.006,.006];cap=.006
   if elapsed>=1.8:phase=policy.route('complete',t,o,'Open-hand approach completed');start=t
  elif phase=='grasp':
   radial=[float(np.clip(r-.001*(2.5-f)*.02,-.008,.008)) for r,f in zip(radial,s.loads[1:3])];cap=float(np.clip(cap-.001*(2.-s.loads[0])*.02,-.006,.006))
   if elapsed>3.5:
    if min(s.loads[1:3])<.3:outcome='grasp_failed';break
    phase=policy.route('complete',t,o,'Both opposed contacts carry load');start=t;transport_q=s.d.ctrl[:20].copy();transport_start_R=s.d.xmat[s.tool].reshape(3,3).copy()
  elif phase=='lift':
   center=PICK+np.array([0,0,.075*smooth(elapsed/1.5)]);radial=[-.002]*2;cap=.003
   if elapsed>1.8:
    if s.d.xpos[s.tool][2]<PICK[2]+.06 or min(s.loads[1:3])<.2:outcome='lift_not_retained';break
    phase=policy.route('complete',t,o,'Lift clears the cradle');start=t
  elif phase=='reorient':
   a=smooth(elapsed/6.);center=(1-a)*(PICK+[0,0,.075])+a*(CENTER+[0,0,.055]);tilt=(1-a)*-np.pi/2;radial=[-.002]*2;cap=.002
   if elapsed>6.5:
    if o['tilt_deg']>2 or np.linalg.norm(s.d.xpos[s.tool]-(CENTER+[0,0,.055]))>.005:outcome='reorientation_failed';break
    phase=policy.route('complete',t,o,'Tool is upright and retained');start=t
  elif phase=='align':
   center=CENTER+np.array([0,0,.055*(1-smooth(elapsed/4.))]);tilt=0.;radial=[-.0015]*2;cap=.0005
   center[:2]+=(s.d.xmat[s.tool].reshape(3,3)@np.array([0,0,.123]))[:2]
   if elapsed>4.5:
    if o['bit_lateral_mm']>.3 or o['bit_height_mm']>.5:outcome='alignment_failed';break
    phase=policy.route('complete',t,o,'Bit aligned above slot');start=t
  elif phase=='press':
   center=CENTER.copy();center[2]-=s.d.qpos[s.depthq]+.0003*smooth(elapsed/.7);radial=[-.001]*2;cap=-.0018
   if elapsed>1.5:
    if not o['engaged']:outcome='engagement_failed';break
    s.pickup_thumb=True
    phase=policy.route('complete',t,o,'Measured bit engagement');start=t;shift_cart=s.cart_state();shift_rot=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();shift_rad=np.array(radial)
    for i in [1,2]:
     line=np.zeros(6);mujoco.mj_geomDistance(s.m,s.d,s.tipids[i],s.m.geom('handle').id,.3,line);b=s.bids[i];s.local[i]=s.d.xmat[b].reshape(3,3).T@(line[:3]-s.d.xpos[b]);s.homepatch[i]=line[3:]-s.d.xpos[s.tool]
    shift_rad=np.zeros(2);radial=[0.,0.];cap=0.;center=s.d.xpos[s.tool].copy();drive_center=center.copy();thumb_offset=0.;feed_offset=0.
  elif phase=='shift_grip':
   center=s.d.xpos[s.tool].copy();cap=0.;radial=[0.,0.]
   if elapsed>1.0:
    phase=policy.route('complete',t,o,'Tripod support retained after bit engagement');start=t;turnstart=t;thumb_offset=0.;rad=np.zeros(2);radial=[0.,0.];center=s.d.xpos[s.tool].copy()
    for i in range(3):
     line=np.zeros(6);mujoco.mj_geomDistance(s.m,s.d,s.tipids[i],s.m.geom('handle').id,.3,line);b=s.bids[i];s.local[i]=s.d.xmat[b].reshape(3,3).T@(line[:3]-s.d.xpos[b]);s.homepatch[i]=line[3:]-s.d.xpos[s.tool];normal=np.r_[s.homepatch[i][:2],0.];s.normals[i]=normal/np.linalg.norm(normal)
  elif phase=='turn':
   center=CENTER.copy();center[2]-=s.d.qpos[s.depthq];thumb_offset=np.clip(thumb_offset-.0003*(.4-s.loads[0])*.02,-.004,.008);cap=thumb_offset
   for i in range(2):
    p=((t-turnstart)/2.4+.325+i*.5)%1
    driving=p<.65;angles[i]=-22+44*(p/.65) if driving else 22-44*((p-.65)/.35)
    target=1.0 if driving else .08
    rad[i]=np.clip(rad[i]-.0003*(target-s.loads[i+1])*.02,-.004,.003)
    radial[i]=float(rad[i]);s.modes[i]='drive' if driving else 'slide_return'
   if mode=='regrasp':
    period=3.2;p=(t-turnstart)%period
    thumb_target=.7 if p<1.4 else 2.;thumb_offset=np.clip(thumb_offset-.0003*(thumb_target-s.loads[0])*.02,-.004,.008);cap=thumb_offset
    if p<1.4:angles=[min(28,p*20)]*2;radial=[-.001]*2;s.modes=['drive','drive']
    elif p<1.8:angles=[28.]*2;radial=[.004*smooth((p-1.4)/.4)]*2;s.modes=['release','release']
    elif p<2.5:angles=[28*(1-smooth((p-1.8)/.7))]*2;radial=[.004]*2;s.modes=['return','return']
    else:angles=[0.]*2;radial=[.004*(1-smooth((p-2.5)/.7))-.001]*2;s.modes=['regrasp','regrasp']
   if o['depth_mm']>=1.985:phase=policy.route('complete',t,o,'Target insertion depth reached');start=t;seatangles=angles.copy()
  elif phase=='seat':
   center=CENTER.copy();center[2]-=s.d.qpos[s.depthq];angles=seatangles;cap=-.0018
  elapsed=t-start;s.stage=phase
  grip_offset=s.grip_offset if phase in ['approach','grasp','lift','reorient','align','press'] else 0.*(1-smooth(elapsed/2.)) if phase=='shift_grip' else 0.
  adjusted_center=center+Rot.from_euler('y',tilt).apply([0,0,grip_offset])
  adjusted_cap=cap if s.pickup_thumb else cap-grip_offset
  s.thumb_outside=0.
  s.finger_roll=angles.copy();q=s.ik_pose(adjusted_center,tilt,angles,radial,adjusted_cap);cart=s.cart_pose(adjusted_center,tilt)
  if phase=='shift_grip':
   q=transport_q;cart=shift_cart.copy()
   if arm:s.arm_orientation_target=shift_rot.copy()
  if phase in ['turn','seat']:
   feed_offset=float(np.clip(feed_offset+.0003*(.7-s.axial)*.02,-.003,.01));cart=shift_cart.copy();cart[0]+=s.d.qpos[s.depthq]+feed_offset
   if arm and mode!='regrasp':
    actual=s.d.xmat[s.tool].reshape(3,3);yaw=np.arctan2(actual[1,0],actual[0,0]);change=Rot.from_euler('z',yaw).as_matrix()@actual.T
    palm=s.d.xpos[s.m.body('palm').id];bit=s.d.xpos[s.tool]+actual@np.array([0,0,-.123]);desired_bit=bit.copy();desired_bit[:2]=0
    change=Rot.from_rotvec(Rot.from_matrix(change).as_rotvec()*.1).as_matrix()
    desired_palm=bit+.1*(desired_bit-bit)+change@(palm-bit);offset=desired_palm-s.p0
    cart[:3]=[-offset[2],offset[0],offset[1]];s.arm_orientation_target=change@s.d.xmat[s.m.body('palm').id].reshape(3,3)
  if phase in ['lift','reorient','align','press']:
   for i in [0,1,2]:
    b=s.bids[i];point=s.d.xpos[b]+s.d.xmat[b].reshape(3,3)@s.local[i];J=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,J,None,point,b);J=J[:,s.vi[4*i:4*i+4]]
    normal=s.d.xmat[s.tool].reshape(3,3)@s.normals[i];desired_load=2. if i==0 or (phase=='shift_grip' and s.axial>.2) else 3.;move=-normal*.0004*(desired_load-s.loads[i])*.02
    transport_q[4*i:4*i+4]+=np.clip(J.T@np.linalg.solve(J@J.T+1e-6*np.eye(3),move),-.006,.006)
   q=transport_q
   actual=s.d.xmat[s.tool].reshape(3,3);actual_tilt=np.arctan2(actual[0,2],actual[2,2]);delta=tilt-actual_tilt
   palm=s.d.xpos[s.m.body('palm').id];delta_center=center-s.d.xpos[s.tool]
   change=Rot.from_euler('y',delta).as_matrix()
   if arm:
    from scipy.spatial.transform import Slerp
    desired_R=transport_start_R if phase=='lift' else Slerp([0,1],Rot.from_matrix([transport_start_R,np.eye(3)]))([smooth(elapsed/6.)]).as_matrix()[0] if phase=='reorient' else np.eye(3)
    change=desired_R@actual.T
    s.arm_orientation_target=change@s.d.xmat[s.m.body('palm').id].reshape(3,3)
   desired_palm=s.d.xpos[s.tool]+delta_center+change@(palm-s.d.xpos[s.tool])
   offset=desired_palm-s.p0;cart=np.r_[-offset[2],offset[0],offset[1],s.cart_state()[-1]+delta]
  events=s.step_full(q,cart,capture)
  if k%10==0:trace.append(dict(t=t,phase=phase,angles=angles.copy(),radial_mm=(np.array(radial)*1000).tolist(),modes=s.modes.copy()))
  if verbose and k%50==0:print(round(t,2),phase,'depth',round(s.observe()['depth_mm'],3),'tool',np.round(s.d.xpos[s.tool],3),'loads',np.round(s.loads,2),'tilt',round(s.observe()['tilt_deg'],1),flush=True)
  if events:outcome=events[0];break
  if held>=.4:outcome='seated';break
 capture(s.observe(),True)
 if outcome=='seated':policy.route('complete',float(s.d.time),s.observe(),'Depth and seating held under axial load')
 else:policy.route('failed',float(s.d.time),s.observe(),outcome)
 r=dict(id=('arm-' if arm else 'stage-')+('hydro-' if hydro else 'point-')+mode,method=mode,case='nominal',outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,trace=trace,metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=float(s.peakpen*1000)),target_depth_mm=2,graph_trace=policy.transitions)
 if hydro:r['hydro']=s.hydro.stats
 return r,np.array(qs),s.xml
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--timeout',type=float,default=70);p.add_argument('--arm',action='store_true');p.add_argument('--hydro',action='store_true');p.add_argument('--name');p.add_argument('--mode',default='slide');a=p.parse_args();r,q,xml=trial(mode=a.mode,timeout=a.timeout,arm=a.arm,hydro=a.hydro);r['id']=a.name or r['id'];OUT.mkdir(exist_ok=True,parents=True);(OUT/(r['id']+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(r['id']+'.npz'),qpos=q);(OUT/(r['id']+'.xml')).write_text(xml);print('RESULT',r['outcome'],r['final'],flush=True)
