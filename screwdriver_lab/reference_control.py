"""Reference-inspired index-cap, thumb/middle rolling controller with a real UR5e.

Uses the same screw mechanism as the earlier study to isolate strategy changes.
Free tool; only the 26 robot joints are actuated. No post-reset tool pose edits.
"""
import json,argparse,hashlib,xml.etree.ElementTree as E
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.arm_sim import ArmFullSim
from screwdriver_lab.arm_scene import attach_arm,ArmIK
from screwdriver_lab.full_scene import build_full,OUT
from screwdriver_lab.screw_scene import CENTER as BASE_CENTER,ROOT
FIXTURE_LIFT=.12
CENTER=BASE_CENTER+np.array([0,0,FIXTURE_LIFT])
from screwdriver_lab.screw_sim import ScrewSim
REF=ROOT/'screwdriver_lab/artifacts/reference_strategy'

class ReferenceSim(ArmFullSim):
 def __init__(self,case=None):
  self.case=case or {};g=json.loads((REF/'grasp.json').read_text());self.grasp_calibration=g;self.p0=np.array(g['palm'])+[0,0,FIXTURE_LIFT];self.R0=np.array(g['rotation']);root=E.fromstring(attach_arm(build_full(self.case)[0]));world=root.find('worldbody')
  for name in ['screwdriver','screw_slide']:
   b=world.find(f"body[@name='{name}']");pos=np.fromstring(b.get('pos'),sep=' ');pos[2]+=FIXTURE_LIFT;b.set('pos',' '.join(map(str,pos)))
  for geom in list(world.findall('geom')):
   if geom.get('name','').startswith('cradle_'):world.remove(geom);continue
   if geom.get('name','').startswith('workpiece_'):
    pos=np.fromstring(geom.get('pos'),sep=' ');size=np.fromstring(geom.get('size'),sep=' ');pos[2]+=FIXTURE_LIFT/2;size[2]+=FIXTURE_LIFT/2;axis=int(geom.get('name').split('_')[1]);size[axis]=.021;pos[axis]=np.sign(pos[axis])*.0243;geom.set('pos',' '.join(map(str,pos)));geom.set('size',' '.join(map(str,size)))
  E.SubElement(world,'geom',name='pickup_surface',type='box',pos=' '.join(map(str,self.case.get('pickup_position',[-.09,.15,.20482]))),size='.14 .045 .012',rgba='.36 .22 .12 1')
  self.xml=E.tostring(root,encoding='unicode');self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m)
  self.jids=np.array([self.m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)]);self.qi=self.m.jnt_qposadr[self.jids];self.vi=self.m.jnt_dofadr[self.jids];self.bids=[self.m.body(f'link_{i}_4').id for i in range(1,6)];self.tipids=[self.m.geom(f'finger_tip_{i}').id for i in range(1,6)]
  for attr,name in [('tool','screwdriver'),('cap','swivel_cap'),('screw','screw')]:setattr(self,attr,self.m.body(name).id)
  self.depthq=self.m.joint('screw_depth').qposadr[0];self.angleq=self.m.joint('screw_angle').qposadr[0];self.anglev=self.m.joint('screw_angle').dofadr[0];self.toq=self.m.joint('tool_free').qposadr[0]
  self.arm=ArmIK(self.m);self.d.qpos[self.arm.qi]=g['arm_q']if g.get('arm_fixture_lift')==FIXTURE_LIFT else self.arm.solve(self.p0,self.R0,exact=True);self.d.qpos[self.qi]=g['q'];self.d.ctrl[:20]=g['q'];self.d.ctrl[20:]=self.d.qpos[self.arm.qi];self.arm_ready=True;self.target=np.array(g['q']);self.openq=self.target.copy();self.home=self.target.copy();self.arm_orientation_target=self.R0.copy()
  self.loads=np.zeros(5);self.axial=self.slot_force=self.pen=self.forbidden=self.peak=self.peakpen=self.last_yaw=self.tool_angle=self.seat=0.;self.slip=[0.,0.];self.handleloads=[0.,0.];self.stage='press';self.modes=['press','press'];self.finger_roll=[0.,0.];self.feed=0.;mujoco.mj_forward(self.m,self.d)
  self.local=[np.array(v)for v in g['local']];self.homepatch=[np.array(v)for v in g['patches']];self.normals=[np.array(v)for v in g['normals']];self.pad_normals_local=[self.d.xmat[self.bids[i]].reshape(3,3).T@(-self.normals[i])for i in range(3)];self.measure();self.set_hand_impedance(2.,.08)
 def cart_state(self):
  offset=self.d.xpos[self.m.body('palm').id]-self.p0;return np.r_[-offset[2],offset[:2],0.]
 def apply_control(self,q,cart):
  pos=self.p0+np.array([cart[1],cart[2],-cart[0]]);rot=self.arm_orientation_target
  self.d.ctrl[:20]=self.bound(q);self.d.ctrl[20:]=self.arm.solve(pos,rot,seed=self.d.qpos[self.arm.qi]);self.arm_goal_pos=pos.copy();self.arm_goal_rot=rot.copy()
 def measure(self):
  ScrewSim.measure(self);self.slip=[0.,0.];self.handleloads=[0.,0.];self.index_cap_load=0.
  for k,c in enumerate(self.d.contact):
   bs=[int(self.m.geom_bodyid[g])for g in [c.geom1,c.geom2]];gn=[self.m.geom(g).name for g in [c.geom1,c.geom2]]
   if 'cap'in gn:
    other=bs[1-gn.index('cap')]
    if self.m.body(other).name.startswith('link_2_'):
     f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);self.index_cap_load+=max(0.,float(f[0]))
   if 'handle'not in gn:continue
   other=bs[1-gn.index('handle')];name=self.m.body(other).name
   if not name.startswith(('link_1_','link_3_')):continue
   i=0 if name.startswith('link_1_')else 1;f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);load=max(0.,float(f[0]));self.handleloads[i]+=load
   if load<=.001:continue
   J=np.zeros((3,self.m.nv));K=J.copy();mujoco.mj_jac(self.m,self.d,J,None,c.pos,other);mujoco.mj_jac(self.m,self.d,K,None,c.pos,self.tool);v=(J-K)@self.d.qvel;n=c.frame[:3];self.slip[i]+=load*np.linalg.norm(v-n*np.dot(v,n))*1000
  self.slip=[v/max(1e-9,w)for v,w in zip(self.slip,self.handleloads)]
 def observe(self):
  o=super().observe();o.update(index_cap_load_N=self.index_cap_load,contact_roles={'cap':'index','handle':['thumb','middle']});return o
 def capture_reference(self,preserve_index=False):
  index=(self.local[1].copy(),self.homepatch[1].copy(),self.normals[1].copy())
  self.local=[];self.homepatch=[];self.normals=[]
  for i in range(3):
   if i==1 and preserve_index:
    self.local.append(index[0]);self.homepatch.append(index[1]);self.normals.append(index[2]);continue
   line=np.zeros(6);mujoco.mj_geomDistance(self.m,self.d,self.tipids[i],self.m.geom('cap'if i==1 else'handle').id,.3,line);b=self.bids[i];self.local.append(self.d.xmat[b].reshape(3,3).T@(line[:3]-self.d.xpos[b]));self.homepatch.append(line[3:]-self.d.xpos[self.tool]);n=np.array([0.,0.,1.])if i==1 else np.r_[line[3:5]-self.d.xpos[self.tool,:2],0];self.normals.append(n/np.linalg.norm(n))
  self.pad_normals_local=[self.d.xmat[self.bids[i]].reshape(3,3).T@(-self.normals[i])if i!=1 or not preserve_index else self.pad_normals_local[i]for i in range(3)]
 def contact_ik(self,angle,radial,cap=-.0012,center=None,tool_R=None,active=(0,1,2)):
  center=CENTER-np.array([0,0,self.d.qpos[self.depthq]])if center is None else np.asarray(center);tool_R=np.eye(3)if tool_R is None else tool_R
  q=self.d.qpos[self.qi].copy()
  for i in active:
   slot=0 if i==0 else 1;patch=self.homepatch[i]+self.normals[i]*(cap if i==1 else radial[slot]);patch=patch if i==1 else Rot.from_euler('z',-(angle[slot]if np.ndim(angle)else angle),degrees=True).apply(patch)
   target=center+tool_R@patch;b=self.bids[i];point=self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@self.local[i];J=np.zeros((3,self.m.nv));Jr=J.copy();mujoco.mj_jac(self.m,self.d,J,Jr,point,b);J=J[:,self.vi[4*i:4*i+4]];e=target-point;e*=min(1.,.002/max(1e-9,np.linalg.norm(e)))
   weight=self.case.get('pad_normal_weight',0.)
   if weight:
    normal=self.d.xmat[b].reshape(3,3)@self.pad_normals_local[i];rot=Rot.from_euler('z',-(angle[slot]if np.ndim(angle)else angle),degrees=True).as_matrix()if i!=1 else np.eye(3);normal_target=-tool_R@rot@self.normals[i];x,y,z=normal;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);J=np.vstack([J,-weight*skew@Jr[:,self.vi[4*i:4*i+4]]]);e=np.r_[e,weight*(normal_target-normal)]
   dq=np.linalg.solve(J.T@J+1e-6*np.eye(4),J.T@e);q[4*i:4*i+4]+=np.clip(.8*dq,-.04,.04)
  return self.bound(q)

 def velocity_ik(self,velocities,loads,cap):
  q=self.contact_ik(0,[0,0],cap=cap,active=(1,));axis=np.array([0.,0.,1.])
  for slot,i in enumerate([0,2]):
   line=np.zeros(6);mujoco.mj_geomDistance(self.m,self.d,self.tipids[i],self.m.geom('handle').id,.3,line);point=line[:3];r=line[3:]-self.d.xpos[self.tool];r[2]=0;n=r/max(1e-9,np.linalg.norm(r));J=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,J,None,point,self.bids[i]);cols=self.vi[4*i:4*i+4];Ji=J[:,cols];v=J@self.d.qvel;desired=-np.radians(velocities[slot])*np.cross(axis,r);vt=v-axis*np.dot(v,axis)-n*np.dot(v,n);F=-n*np.clip(loads[slot]+1.5*(loads[slot]-self.handleloads[slot]),-.8,1.5)+25*(desired-vt)
   tau=Ji.T@F+.08*self.d.qvel[cols];q[4*i:4*i+4]=self.d.qpos[self.qi[4*i:4*i+4]]+np.clip(tau/2.,-.15,.15)
  return self.bound(q)

def run(timeout=15,mode='compact',period=1.4,amplitude=22,release=.0005,pad_weight=0.,case=None):
 if mode not in ['compact','guarded','rolling','velocity','hold']:raise ValueError(mode)
 from screwdriver_lab.reference_cycle import ReferenceCycle,ReferenceRollingCycle
 cycle=ReferenceRollingCycle()if mode in ['rolling','velocity']else ReferenceCycle(amplitude,release)
 s=ReferenceSim({**(case or {}),'pad_normal_weight':pad_weight});frames=[];qs=[];commands=[];nextframe=0.;held=0.;outcome='timeout';rad=np.array([-.0006,-.0006]);phase='press';cycles=0;last_phase='press';angle=0.;cap_offset=-.0003
 def capture(o):
  nonlocal nextframe,held
  if o['t']>=nextframe-1e-9:frames.append(dict(o,phase=phase));qs.append(s.d.qpos.copy());nextframe+=1/30
  ok=o['depth_mm']>=1.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and .15<o['axial_force_N']<5 and o['seat_force_N']>.1;held=held+.001 if ok else 0.
 capture(s.observe())
 for k in range(int(timeout/.02)):
  t=s.d.time;o=s.observe();phase='press';radial=rad.copy()
  if t>=1.5:
   p=((t-1.5)/period)%1
   if p<.70:phase='drive';angle=amplitude*p/.70;target=.8;rad=np.clip(rad-.0005*(target-s.loads[[0,2]])*.02,-.003,.003);radial=rad.copy()
   else:
    phase='compact_return';angle=amplitude*(1-(p-.70)/.30);radial=np.array([release,release])
   if phase=='drive'and last_phase=='compact_return':cycles+=1
  if mode=='guarded'and t>=1.5:angle,radial,phase=cycle.tick(float(t),o);cycles=cycle.cycles
  if mode in ['rolling','velocity']and t>=1.5:angle,radial,rolling_modes=cycle.tick(float(t),o);phase='rolling';cycles=cycle.cycles
  if mode=='hold':angle=0;radial=[-.0007,-.0007];phase='hold'
  if o['depth_mm']>=1.98 and o['seat_force_N']>.1:phase='seat';angle=s.finger_roll[0];radial=rad.copy()
  s.modes=rolling_modes if mode in ['rolling','velocity']and t>=1.5 else[phase,phase];s.finger_roll=angle if isinstance(angle,list)else[angle,angle];cap_offset=float(np.clip(cap_offset+.001*(s.axial-.6)*.02,-.007,.004));q=s.contact_ik(angle,radial,cap=cap_offset)
  if mode=='velocity'and t>=1.5:q=s.velocity_ik([30/(2.4*.65)if x=='drive'else-30/(2.4*.35)for x in rolling_modes],[.8 if x=='drive'else .02 for x in rolling_modes],cap_offset)
  s.feed=float(s.d.qpos[s.depthq]);s.stage='turn'if t>=1.5 else'press';events=s.step_full(q,np.array([s.feed,0,0,0.]),capture);commands.append(dict(t=float(t),phase=phase,q=q.tolist(),angle=angle if isinstance(angle,list)else float(angle),radial_mm=(np.array(radial)*1000).tolist()));last_phase=phase
  if k%50==0:print(round(t,2),phase,round(s.observe()['depth_mm'],3),np.round(s.loads,2),'cap',round(s.index_cap_load,2),'pen',round(s.pen*1000,3),flush=True)
  if events:outcome=events[0];break
  if mode=='guarded'and cycle.phase=='blocked':outcome='contact_release_failed';break
  if held>=.4:outcome='seated';break
 frames.append(dict(s.observe(),phase=phase));qs.append(s.d.qpos.copy());r=dict(id='reference-'+mode,method='reference strategy',case=(case or {}).get('id','aligned'),scope='corrected finger roles, drive only',outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,commands=commands,trace=commands,metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=float(s.peakpen*1000),cycles=cycles),target_depth_mm=2,parameters=dict(period=period,amplitude_deg=amplitude,return_clearance_mm=release*1000,pad_normal_weight=pad_weight,case=case or {},budget_s=timeout),initial_state_sha256=hashlib.sha256(qs[0].tobytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/'screwdriver_lab/reference_control.py',ROOT/'screwdriver_lab/reference_cycle.py']},grasp_calibration=s.grasp_calibration,cycle_trace=cycle.trace)
 return r,np.array(qs),s.xml
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--timeout',type=float,default=15);p.add_argument('--mode',default='compact');p.add_argument('--name');p.add_argument('--period',type=float,default=1.4);p.add_argument('--amplitude',type=float,default=22);p.add_argument('--release',type=float,default=.0005);p.add_argument('--pad-weight',type=float,default=0.);a=p.parse_args();r,q,xml=run(a.timeout,a.mode,a.period,a.amplitude,a.release,a.pad_weight);r['id']=a.name or r['id'];(OUT/(r['id']+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(r['id']+'.npz'),qpos=q);(OUT/(r['id']+'.xml')).write_text(xml);print('RESULT',r['outcome'],r['final'])
