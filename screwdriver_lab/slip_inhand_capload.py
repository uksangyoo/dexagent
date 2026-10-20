"""Observed strategy order: thumb/middle pickup, reorient, index cap, lower, roll."""
import argparse,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot,Slerp
from scipy.optimize import least_squares
from screwdriver_lab.reference_control import REF,CENTER,OUT
from screwdriver_lab.controlled_slip import SlipSim,SlipPairedController
from screwdriver_lab.slip_metrics import SlipMetrics
from screwdriver_lab.pivot_spin_controller import CapturedSlipController
from screwdriver_lab.slip_transport import CapTransportServo
from screwdriver_lab.pivot_cap_servo import CapLoadServo,FullContactController
from screwdriver_lab.full_scene import PICK as OLD_PICK
PICK=np.array([-.22,.10,.35])
from screwdriver_lab.full_motion import smooth

def axis_frame(R):
 z=R[:,2];y=np.array([0.,1.,0.]);y-=z*np.dot(y,z);y/=np.linalg.norm(y);return np.column_stack([np.cross(y,z),y,z])

PHASES=['approach','grasp','lift','reorient','cap_support','align','turn','seat']
def graph():
 return {'version':3,'meta':{'name':'reference_index_cap_strategy','source':'human upper-left panel; manually interpreted contact strategy','finger_roles':{'index':'axial cap support','thumb':'handle roll','middle':'opposed handle roll'}},'edges':[['START','approach']],'nodes':{p:{'type':'subgraph','ref':p+'_skill'}for p in PHASES},'conditional_edges':{p:{'router_field':'exit','mapping':{'complete':PHASES[i+1]if i+1<len(PHASES)else'done','failed':'abort'}}for i,p in enumerate(PHASES)},'subgraphs':{p+'_skill':{'inputs':['observation','contact_roles'],'exits':['running','complete','failed']}for p in PHASES}}

def clear_arm(s,pos,rot):
 bid=s.m.body('palm').id;qi=s.arm.qi;low=s.m.jnt_range[s.arm.jids,0]+1e-4;high=s.m.jnt_range[s.arm.jids,1]-1e-4;rng=np.random.default_rng(12);best=None
 for k in range(24):
  seed=s.d.qpos[qi].copy()if k==0 else rng.uniform(-np.pi,np.pi,6)
  def fun(q):
   s.d.qpos[qi]=q;mujoco.mj_forward(s.m,s.d);return np.r_[(s.d.xpos[bid]-pos)*5,Rot.from_matrix(rot@s.d.xmat[bid].reshape(3,3).T).as_rotvec()]
  f=least_squares(fun,seed,bounds=(low,high),max_nfev=160);err=np.linalg.norm(fun(f.x));pen=max([max(0,-c.dist)for c in s.d.contact if any(s.m.geom(g).name.startswith('arm_collision')for g in [c.geom1,c.geom2])]+[0]);score=err+pen*100
  if best is None or score<best[0]:best=(score,f.x.copy(),err,pen)
  if err<1e-5 and pen<1e-5:break
 if best[2]>1e-4 or best[3]>.0001:raise RuntimeError(f'No clear initial UR5e pose: {best}')
 s.d.qpos[qi]=best[1];s.arm.last=best[1].copy();return best[1]

def pickup_supports(s):
 import xml.etree.ElementTree as E
 from screwdriver_lab.arm_scene import ArmIK
 root=E.fromstring(s.xml);world=root.find('worldbody');world.remove(world.find("geom[@name='pickup_surface']"))
 for name,dx,rad in [('cap',.043,float(s.m.geom('cap').size[0])),('shaft',-.070,float(s.m.geom('shaft').size[0]))]:
  top=PICK[2]-rad;bottom=.12
  E.SubElement(world,'geom',name='pickup_'+name,type='box',pos=' '.join(map(str,[PICK[0]+dx,PICK[1],(top+bottom)/2])),size=' '.join(map(str,[.002,.025,(top-bottom)/2])),rgba='.36 .22 .12 1')
 q=s.d.qpos.copy();ctrl=s.d.ctrl.copy();s.xml=E.tostring(root,encoding='unicode');s.m=mujoco.MjModel.from_xml_string(s.xml);s.d=mujoco.MjData(s.m);s.d.qpos[:]=q;s.d.ctrl[:]=ctrl;s.arm=ArmIK(s.m);s.arm.last=s.d.qpos[s.arm.qi].copy();mujoco.mj_forward(s.m,s.d)

def run(timeout=500,period=1.4,amplitude=24,release=0,strategy='controlled_slip',recovery=False,transport_torsion=.0005):
 from screwdriver_lab.reference_cycle import ReferenceCycle,ReferenceRollingCycle
 s=SlipSim('screwdriver_lab/artifacts/controlled_slip_tip',.015,transport_torsion,8,.003,0,pickup_position=[-.22,.25,.20482]);pickup_supports(s);slip=None;evidence=SlipMetrics();cap_servo=CapTransportServo(s);cap_load_servo=CapLoadServo(s);
 cycle=ReferenceRollingCycle()if strategy=='rolling'else ReferenceCycle(35,.0005);g=graph();g['nodes']['recover']={'type':'subgraph','ref':'recover_skill'};g['conditional_edges']['turn']['mapping']['reacquire']='recover';g['conditional_edges']['recover']={'router_field':'exit','mapping':{'complete':'turn','failed':'abort'}};g['subgraphs']['recover_skill']={'inputs':['observed_handle_loads','index_cap_load'],'exits':['running','complete','failed']};(REF/'controlled-slip-workflow.json').write_text(json.dumps(g,indent=2));phase=g['edges'][0][1];transitions=[];start=0.;outcome='timeout';tilt=np.pi/2;pickup_roll=0.;Rpick=Rot.from_euler('y',tilt).as_matrix()@Rot.from_euler('z',pickup_roll).as_matrix();offset=s.p0-CENTER
 # Solve an open hand at reset, then move the whole articulated robot/tool to the pickup initial condition.
 qref=s.d.qpos[s.qi].copy();targets=[CENTER+s.homepatch[i]+s.normals[i]*(.010 if i==1 else .006)for i in range(3)]
 def open_res(q):
  s.d.qpos[s.qi]=q;mujoco.mj_forward(s.m,s.d);points=[s.d.xpos[s.bids[i]]+s.d.xmat[s.bids[i]].reshape(3,3)@s.local[i]for i in range(3)];pen=np.zeros(s.m.ngeom)
  for c in s.d.contact:
   if any(s.m.body(int(s.m.geom_bodyid[g])).name.startswith('link_')for g in [c.geom1,c.geom2]):pen[c.geom1]=max(pen[c.geom1],-c.dist-.00001)
  return np.r_[(np.array(points)-targets).ravel()*400,pen*12000,.05*(q-qref)]
 fit=least_squares(open_res,qref,bounds=(s.m.jnt_range[s.jids,0]+1e-5,s.m.jnt_range[s.jids,1]-1e-5),max_nfev=250,diff_step=1e-4);s.d.qpos[s.qi]=fit.x;mujoco.mj_forward(s.m,s.d)
 openq=s.d.qpos[s.qi].copy();s.d.qpos[s.toq:s.toq+3]=PICK;s.d.qpos[s.toq+3:s.toq+7]=Rot.from_matrix(Rpick).as_quat(scalar_first=True);s.d.qpos[s.depthq]=0;s.d.qpos[s.angleq]=0;s.d.qpos[s.m.joint('cap_bearing').qposadr[0]]=0
 pos=PICK+offset+[0,0,.05];rotation=s.R0.copy();clear_arm(s,pos,rotation);s.d.qvel[:]=0;s.d.ctrl[:20]=openq;s.d.ctrl[20:]=s.d.qpos[s.arm.qi];s.arm_orientation_target=rotation.copy();mujoco.mj_forward(s.m,s.d);s.last_yaw=0;s.tool_angle=0;s.peak=0;s.peakpen=0;s.measure();s.openq=openq.copy();s.set_hand_impedance(8.,.18)
 initial_clearance={}
 for name in ['handle','cap','shaft']:
  target=s.m.geom(name).id;initial_clearance[name]=min(mujoco.mj_geomDistance(s.m,s.d,i,target,.5,None)for i in range(s.m.ngeom)if s.m.body(int(s.m.geom_bodyid[i])).name.startswith(('link_','palm')))*1000
 assert min(initial_clearance.values())>1,initial_clearance
 frames=[];qs=[];commands=[];nextframe=0.;spin_angle=0.;spin_time=0.;last_spin_rate=0.;held=0.;rad=np.array([.006,.006]);cap=.010;angle=0.;cycles=0;lastmode='press';drive_start=0.;transport=None;lost_for=0.;recoveries=0
 def capture(o,force=False):
  nonlocal nextframe,held,spin_angle,spin_time,last_spin_rate
  velocity=np.zeros(6);mujoco.mj_objectVelocity(s.m,s.d,mujoco.mjtObj.mjOBJ_BODY,s.tool,velocity,0);spin_rate=-float(np.dot(velocity[:3],s.d.xmat[s.tool].reshape(3,3)[:,2]));spin_angle+=(o['t']-spin_time)*.5*(spin_rate+last_spin_rate);spin_time=o['t'];last_spin_rate=spin_rate;o['tool_axial_angle_deg']=float(np.degrees(spin_angle))
  if slip is not None:evidence.add(o,s.modes)
  if force or o['t']>=nextframe-1e-9:frames.append(dict(o,phase=phase,tool_axis=s.d.xmat[s.tool].reshape(3,3)[:,2].tolist(),palm_rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).tolist(),contact_phase=slip.phase if slip else phase));qs.append(s.d.qpos.copy());nextframe+=1/30
  ok=o['depth_mm']>=1.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and .15<o['axial_force_N']<5 and o['seat_force_N']>.1;held=held+.001 if ok else 0.
 def route(exit,reason):
  nonlocal phase,start
  t=float(s.d.time);n=g['conditional_edges'][phase]['mapping'][exit];transitions.append(dict(t=t,from_node=phase,exit=exit,to=n,reason=reason,evidence=s.observe()));phase=n;start=t
 capture(s.observe(),True)
 for k in range(int(timeout/.02)):
  phase_at_tick=phase;t=float(s.d.time);dt=t-start;o=s.observe();twist=pickup_roll if phase in ['approach','grasp','lift']else pickup_roll*(1-smooth(dt/3.))if phase=='reorient'else 0.;actual=axis_frame(s.d.xmat[s.tool].reshape(3,3))@Rot.from_euler('z',twist).as_matrix();tool=s.d.xpos[s.tool].copy();q=s.d.ctrl[:20].copy();mode=phase;center=tool.copy();desired_R=actual.copy()
  if phase=='approach':
   center=PICK+[0,0,.05*(1-smooth(dt/1.3))];desired_R=np.eye(3);q=openq.copy();pos=center+desired_R@offset;rotation=desired_R@s.R0
   if dt>=1.5:grasp_center=tool.copy();grasp_R=np.eye(3);route('complete','Open hand reaches horizontal tool without contact')
  elif phase=='grasp':
   center=tool;desired_R=actual;rad=np.full(2,.006*(1-smooth(dt/2.)))if dt<2. else np.clip(rad-.003*(4.-np.array(s.handleloads))*.02,-.012,.008);q=s.contact_ik(0,rad,cap=.010,center=grasp_center,tool_R=grasp_R);pos=grasp_center+grasp_R@offset;rotation=grasp_R@s.R0
   if dt>=3.0 and min(np.array(s.handleloads))>1.3:transport=q.copy();lift_tool=tool.copy();lift_R=np.eye(3);hand_offset=s.d.xpos[s.m.body('palm').id]-tool;hand_rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();route('complete','Thumb and middle oppose on handle')
   elif dt>12:outcome='pickup_grasp_failed';break
  elif phase=='cap_pickup':
   center=tool;desired_R=actual;pos=grasp_center+grasp_R@offset;rotation=grasp_R@s.R0;q=transport.copy()
   cap=float(np.clip(cap+.003*(s.index_cap_load-.4)*.02,-.008,.010));indexq=s.contact_ik(0,[0,0],cap=cap,center=tool,tool_R=actual,active=(1,));q[4:8]=indexq[4:8]
   if dt>.5 and s.index_cap_load>.15:
    transport=q.copy();lift_tool=tool.copy();lift_R=np.eye(3);hand_offset=s.d.xpos[s.m.body('palm').id]-tool;hand_rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();route('complete','Robot acquires index cap support on the bench for stable transport')
   elif dt>12:outcome='pickup_cap_failed';break
  elif phase in ['lift','reorient','cap_support','align']:
   for i in [0,2]:
    b=s.bids[i];point=s.d.xpos[b]+s.d.xmat[b].reshape(3,3)@s.local[i];J=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,J,None,point,b);J=J[:,s.vi[4*i:4*i+4]];radial=point-tool;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];radial-=axis*np.dot(radial,axis);radial/=max(1e-9,np.linalg.norm(radial));move=-radial*.002*((5. if phase=='lift'else .65 if phase=='reorient'else 3.)-s.handleloads[0 if i==0 else 1])*.02;transport[4*i:4*i+4]+=np.clip(J.T@np.linalg.solve(J@J.T+1e-6*np.eye(3),move),-.01,.01)
   q=transport.copy()
   if False:
    cap=float(np.clip(cap+.001*(s.index_cap_load-.4)*.02,-.008,.010));indexq=s.contact_ik(0,[0,0],cap=cap,center=tool,tool_R=actual,active=(1,));q[4:8]=indexq[4:8];transport[4:8]=q[4:8]
   if phase=='lift':
    center=lift_tool+[0,0,.18*smooth(dt/2.4)];desired_R=lift_R
    if dt>=2.7:
     if tool[2]<PICK[2]+.16 or min(np.array(s.handleloads))<.1:outcome='lift_failed';break
     reorient_start=tool.copy();reorient_R=np.eye(3);hand_offset=s.d.xpos[s.m.body('palm').id]-tool;hand_rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();route('complete','Lift retained by thumb-middle pinch')
   elif phase=='reorient':
    center=reorient_start.copy();desired_R=np.eye(3)
    if dt>=.4 and o['tilt_deg']<20 and min(s.handleloads)>.1:
     cap_start=tool.copy();route('complete','Gravity-assisted in-hand pivot with wrist orientation retained')
    elif dt>10:outcome='pivot_failed';break
   elif phase=='cap_support':
    center=cap_start;desired_R=np.eye(3);cap=max(-.003,cap-.006*.02)if s.index_cap_load<.4 else cap;indexq=s.contact_ik(0,[0,0],cap=cap,center=tool,tool_R=actual,active=(1,));q[4:8]=indexq[4:8];transport[4:8]=q[4:8]
    if dt>1 and s.index_cap_load>.15 and o['tilt_deg']<5:lower_start=tool.copy();lower_started=None;raw=s.d.xmat[s.tool].reshape(3,3);yaw=np.arctan2(raw[1,0],raw[0,0]);align_target_R=Rot.from_euler('z',round(yaw/(np.pi/2))*np.pi/2).as_matrix();route('complete','Index cap support acquired before lowering')
    elif dt>9:outcome='index_cap_acquisition_failed';break
   else:
    raw=s.d.xmat[s.tool].reshape(3,3);rot_error=np.linalg.norm(Rot.from_matrix(align_target_R@raw.T).as_rotvec())
    if lower_started is None and dt>4 and rot_error<np.radians(2) and o['bit_lateral_mm']<.5:lower_started=t
    lower_dt=0. if lower_started is None else t-lower_started
    center=(1-smooth(dt/4.))*lower_start+smooth(dt/4.)*(CENTER+[0,0,.04]);center[2]-=.0403*smooth(lower_dt/3.)
    if o['slot_force_N']>.5 and not o['engaged'] and o['bit_height_mm']>1.5:
     if lower_started is not None:lower_started+=.02
     center[2]=tool[2]+.0001
    desired_R=np.eye(3);cap=np.clip(cap+.001*(s.axial-.6)*.02,-.007,.003);indexq=s.contact_ik(0,[0,0],cap=cap,center=tool,tool_R=actual,active=(1,));q[4:8]=indexq[4:8];transport[4:8]=q[4:8]
    if lower_started is not None and lower_dt>3.4 and o['engaged']:
     drive_pos=s.d.xpos[s.m.body('palm').id].copy();drive_R=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();drive_depth=s.d.qpos[s.depthq];drive_anchor=tool.copy();rad=np.zeros(2);s.set_hand_impedance(2.,.08);s.m.actuator_gainprm[4:8,0]=8.;s.m.actuator_biasprm[4:8,1]=-8.;s.m.actuator_biasprm[4:8,2]=-.18;route('complete','Bit engaged with index cap support');drive_start=t;slip=FullContactController(s,cap_servo=cap_load_servo,cap_offset=cap,amplitude=24,speed=12,drive_load=.4,slide_load=.09,force_p=0,force_i=0,normal_d=5,orientation_gain=0,posture_gain=0);slip.start=t
   if phase_at_tick in ['cap_support','align'] and phase==phase_at_tick:
    correction_R=desired_R@actual.T if phase=='cap_support'else align_target_R@s.d.xmat[s.tool].reshape(3,3).T
    correction=Rot.from_rotvec(Rot.from_matrix(correction_R).as_rotvec()*.04).as_matrix();hand_offset=desired_R.T@correction@desired_R@hand_offset+desired_R.T@(center-tool)*.04;hand_rotation=desired_R.T@correction@desired_R@hand_rotation
    if phase=='align'and dt>14:outcome='bit_engagement_failed';break
   pos=center+desired_R@hand_offset;rotation=desired_R@hand_rotation
  elif phase=='recover':
   mode='reacquire';cap=float(np.clip(cap+.001*(s.axial-.6)*.02,-.007,.004))
   if recover_q is None:
    rad=np.clip(rad-.001*(.9-np.array(s.handleloads))*.02,-.006,.004);q=s.contact_ik(0,rad,cap=cap,center=recover_anchor)
    if min(s.handleloads)>.15:recover_q=q.copy()
   else:
    q=recover_q.copy();indexq=s.contact_ik(0,[0,0],cap=cap,center=tool,tool_R=actual,active=(1,));q[4:8]=indexq[4:8];raw=s.d.xmat[s.tool].reshape(3,3);yaw=np.arctan2(raw[1,0],raw[0,0]);correction=Rot.from_rotvec(Rot.from_matrix(Rot.from_euler('z',yaw).as_matrix()@raw.T).as_rotvec()*.04).as_matrix();bit=tool+raw@np.array([0,0,-.123]);recover_pos=bit+correction@(recover_pos-bit);recover_pos[:2]-=bit[:2]*.04;recover_R=correction@recover_R
   pos=recover_pos;rotation=recover_R
   if dt>.5 and o['tilt_deg']<1.5 and min(s.handleloads)>.1 and s.index_cap_load>.15:
    drive_pos=s.d.xpos[s.m.body('palm').id].copy();drive_R=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();drive_depth=s.d.qpos[s.depthq];drive_anchor=tool.copy();s.capture_reference(preserve_index=True);cycle=ReferenceRollingCycle()if strategy=='rolling'else ReferenceCycle(35,.0005);lost_for=0.;route('complete','Handle reacquired and tool axis realigned');drive_start=t
   elif dt>8:outcome='reacquisition_failed';break
  elif phase in ['turn','seat']:
   if phase=='seat'and slip.phase!='seat':slip.transition('seat',o)
   q,low_command=slip.tick(o);cycles=slip.cycles;mode=s.modes.copy();angle=s.finger_roll.copy()
   pos=drive_pos-[0,0,s.d.qpos[s.depthq]-drive_depth];rotation=drive_R
   if phase=='turn'and o['depth_mm']>=1.98 and o['seat_force_N']>.1:route('complete','Screw reaches seating depth through controlled slip cycles')
  else:break
  if phase!=phase_at_tick:continue
  s.stage='turn'if phase in ['turn','seat']else phase;s.modes=mode if isinstance(mode,list)else[mode,mode];s.finger_roll=angle if isinstance(angle,list)else[angle,angle];s.arm_orientation_target=rotation;delta=pos-s.p0;cart=np.r_[-delta[2],delta[:2],0.];cap_servo.set_target(s.d.xpos[s.tool]+s.d.xmat[s.tool].reshape(3,3)@np.array([0,0,.043]));cap_servo.load=.6 if phase=='align'else .4;events=(slip.step if phase in ['turn','seat']else cap_load_servo.step if phase=='align'else s.step_full)(q,cart,capture);commands.append(dict(t=t,phase=phase,q=q.tolist(),arm=s.d.ctrl[20:].tolist(),finger_mode=mode,angle=angle));lastmode=mode
  if k%50==0:print(round(t,2),phase,round(s.observe()['depth_mm'],3),np.round(s.loads,2),'cap',round(s.index_cap_load,2),'tilt',round(s.observe()['tilt_deg'],1),flush=True)
  if events:outcome=events[0];break
  if held>=.4:outcome='seated';break
 capture(s.observe(),True)
 if phase in g['conditional_edges']:route('complete'if outcome=='seated'else'failed',outcome)
 r=dict(id='controlled-slip-full',method='controlled-slip contact wrench',case='nominal',scope='pickup through screw seating; index cap, thumb/middle controlled slipping',outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,commands=commands,trace=commands,graph_trace=transitions,metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=float(s.peakpen*1000),cycles=cycles,recoveries=recoveries,controlled_slip=evidence.report()),initial_hand_clearance_mm=initial_clearance,initial_state_sha256=hashlib.sha256(qs[0].tobytes()).hexdigest(),target_depth_mm=2,return_strategy=strategy,recovery_enabled=recovery,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [__import__('pathlib').Path(__file__),REF.parents[1]/'reference_control.py',REF.parents[1]/'controlled_slip.py',REF.parents[1]/'pivot_spin_controller.py',REF.parents[1]/'pivot_cap_servo.py']},grasp_calibration=s.grasp_calibration,cycle_trace=slip.trace if slip is not None else [],parameters=dict(controller='paired controlled slip',normal_drive_N=.4,normal_return_N=.09,thread_torque_Nm=.003,native_ccd=False,pad_time=.015,torsional_friction_m=transport_torsion))
 return r,np.array(qs),s.xml
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--timeout',type=float,default=500);p.add_argument('--name',default='controlled-slip-full');p.add_argument('--strategy',choices=['controlled_slip'],default='controlled_slip');p.add_argument('--experimental-recovery',action='store_true');p.add_argument('--torsion',type=float,default=.0005);a=p.parse_args();r,q,xml=run(a.timeout,strategy=a.strategy,recovery=a.experimental_recovery,transport_torsion=a.torsion);r['id']=a.name;(OUT/(a.name+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(a.name+'.npz'),qpos=q);(OUT/(a.name+'.xml')).write_text(xml);print('RESULT',r['outcome'],r['final'])
