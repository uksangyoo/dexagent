"""Motor-only fixture pickup and in-hand cube rotation with three hand APIs."""
import argparse,json,time,hashlib,signal
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from tasks.cube.model import build,OUT
from tasks.cube.policy import build as build_policy,Policy
from tasks.cube.metrics import orientation,valid_hold
from dexagent.sim.drake_backend import HydroPlant
from dexagent.planning.kinematics import finger_velocity_target


def normal(point,center,R):
 r=R.T@(point-center);i=int(np.argmax(np.abs(r)));n=np.zeros(3);n[i]=1 if r[i]>=0 else -1
 return R@n

def run(name,interface='hom',duration=12,goal_deg=90,yaw=0,dt=.00025,balanced=False,edge=.02,pulp=False,grip=.6,speed=.6,resume=None,resume_time=2.2,null_kd=.01,spin_gain=.06,cartesian_kp=None,cartesian_kd=3.,axis='z',pusher_offset=0.,contact_mode='spin',initial_from=None,policy_path=None,index_load=.15,handoff='continuous',hold_mode='track',hold_damping=0.,hold_reset=False):
 if (OUT/(name+'.json')).exists():raise FileExistsError(name)
 source_files=[Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('policy.py'),Path(__file__).with_name('metrics.py'),Path('dexagent/sim/drake_backend.py'),Path('dexagent/planning/initialization.py')]
 sources={};snap=OUT/'source_snapshots';snap.mkdir(parents=True,exist_ok=True)
 for p in source_files:
  content=p.read_bytes();digest=hashlib.sha256(content).hexdigest();sources[p.name]=digest;(snap/(digest+'_'+p.name)).write_bytes(content)
 xml,q,initial=build(name,yaw,balanced,edge,pulp,initial_from)
 h=HydroPlant(xml,dt=dt,resolution=.002,explicit_steel=True,beta=0,null_posture_kp=1,null_posture_kd=null_kd,tangential_stiffness=cartesian_kp,tangential_damping=cartesian_kd)
 checkpoint=None;saved_initial=q.copy();qv=None
 if resume:
  src=OUT/(resume+'.npz');saved=np.load(src);idx=int(np.argmin(abs(saved['time']-resume_time)));q=saved['qpos'][idx].copy();qv=saved['qvel'][idx].copy()
  checkpoint=dict(source=resume,time=float(saved['time'][idx]),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),controller_reinitialized=True,full_task_credit=False)
 h.initialize_from_mujoco(q,qv);h.sync();d=h.d;m=h.m
 policy=Policy(policy_path or build_policy(interface,goal_deg,edge,axis,dict(hold_reset=hold_reset,hold_damping=hold_damping,hold_mode=hold_mode,handoff=handoff,index_load=index_load,grip=grip,speed=speed,pusher_offset=pusher_offset,spin_gain=spin_gain,contact_mode=contact_mode),name));
 if policy.graph['abstraction']!=interface or policy.graph['contract']['target_deg']!=goal_deg or policy.graph['contract']['edge_m']!=edge or f'world {axis} axis'not in policy.graph['contract']['axis']:raise ValueError('Policy and trial contract mismatch')
 hold_reset=policy.graph.get('hold_reset',hold_reset)
 if not isinstance(hold_reset,bool):raise ValueError('hold_reset must be boolean')
 hold_mode=policy.graph.get('hold_mode',hold_mode)
 if hold_mode not in ['track','freeze']:raise ValueError('Unknown hold mode')
 handoff=policy.graph.get('adapter_handoff',handoff)
 if handoff not in ['continuous','adapter_reset']:raise ValueError('Unknown controller handoff')
 bid=m.body('cube').id;palm=m.body('palm').id
 fingers=[m.body(f'link_{i}_4').id for i in [1,2,3]];local=[np.array(r['local'])for r in initial]
 R0=d.xmat[bid].reshape(3,3).copy();Rp0=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy();c0=d.xpos[bid].copy()
 arm_R_target=Rp0.copy();rel0=Rp0.T@R0;goalrel=Rp0.T@Rotation.from_euler(axis,goal_deg,degrees=True).as_matrix()
 qref=d.qpos[h.qi].copy();home=qref.copy();phase='acquire';phase_start=0.;grasp_s=0.;hold_s=0.;lost_s=0.;unexpected=0.;outcome='timeout';pgoal=p0.copy();anchor=None;center_goal=c0.copy();rotate_start=None
 if checkpoint:
  original=mujoco.MjData(m);original.qpos[:]=saved_initial;mujoco.mj_forward(m,original);c0=original.xpos[bid].copy();R0=original.xmat[bid].reshape(3,3).copy();rel0=Rp0.T@R0;phase='rotate';policy.node='rotate';center_goal=d.xpos[bid].copy();rotate_start=d.xmat[bid].reshape(3,3).copy();rotation_start_time=0.;anchor=[d.xpos[b]+d.xmat[b].reshape(3,3)@l-center_goal for b,l in zip(fingers,local)]
 samples=[];poses=[];vels=[];times=[];commands=[];refs=[];wall=time.monotonic();loaded_rotation=0.;previous_R=R0.copy();palm_shift_x=0.;rotation_pgoal=pgoal.copy()
 stopped=False
 def stop(signum,frame):
  nonlocal stopped
  stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for k in range(round(duration/.002)):
  if stopped and k>0:outcome='interrupted';break
  h.sync();R=d.xmat[bid].reshape(3,3);Rp=d.xmat[palm].reshape(3,3);center=d.xpos[bid].copy();loads=np.zeros(3);weighted=np.zeros((3,3));unexpected=0.;unexpected_pair=[]
  for c in h.contacts():
   names=set(c['names']);allowed=False
   for i in range(3):
    if names=={'cube',f'finger_tip_{i+1}'}:
     n=normal(c['point'],center,R);fn=abs(float(c['force']@n));loads[i]+=fn;weighted[i]+=fn*c['point'];allowed=True
   if not allowed and any(n.startswith(('finger_tip_','source_geometry_','arm_collision_'))for n in names):
    force=float(np.linalg.norm(c['force']))
    if force>unexpected:unexpected=force;unexpected_pair=c['names']
  points=[]
  for i,b in enumerate(fingers):
   Ri=d.xmat[b].reshape(3,3)
   if loads[i]>.002:local[i]+=.25*(Ri.T@(weighted[i]/loads[i]-d.xpos[b])-local[i])
   points.append(d.xpos[b]+Ri@local[i])
  Jobj=np.zeros((3,m.nv));Jrot=Jobj.copy();mujoco.mj_jacBody(m,d,Jobj,Jrot,bid);omega=Jrot@d.qvel
  measured=orientation(R,Rp,R0,Rp0,Rotation.from_euler(axis,goal_deg,degrees=True).as_matrix());error_deg=measured['error_deg'];palm_turn=measured['palm_turn_deg'];inhand_turn=measured['inhand_turn_deg']
  lift_mm=float((center[2]-c0[2])*1000)
  initial_error=float(np.degrees(Rotation.from_matrix(R@R0.T).magnitude()))
  grasp_s=grasp_s+.002 if min(loads[[0,2]])>.1 and (not balanced or loads[1]>.025 and initial_error<5) else 0.
  lost_s=lost_s+.002 if phase not in ['acquire']and min(loads[[0,2]])<.01 else 0.
  accepted=valid_hold(measured,lift_mm,float(np.degrees(np.linalg.norm(omega))),loads[[0,2]],unexpected)
  hold_s=hold_s+.002 if accepted else 0.
  if min(loads[[0,2]])>.05:loaded_rotation+=np.degrees(np.linalg.norm(Rotation.from_matrix(R@previous_R.T).as_rotvec()))
  previous_R=R.copy()
  failure=bool(unexpected>5 or lost_s>.15 or center[2]<.40)
  if k%25==0:
   decision=policy.step(dict(t=float(d.time),failure=failure,grasp_ready=grasp_s>.2,lift_ready=bool((lift_mm>77.5 and min(loads)>.05 if handoff=='adapter_reset'else d.time-phase_start>1.8 and lift_mm>65) and (not balanced or initial_error<8 and loads[1]>.025)),near_goal=error_deg<4,accepted=hold_s>=.3))
   command=decision.get('command')
   if command:
    args=command['arguments'];hold_damping=args.get('hold_damping',0.);index_load=args.get('index_load',.15);grip=args['grip'];speed=args['speed'];pusher_offset=args['pusher_offset'];spin_gain=args['spin_gain'];contact_mode=command['contact_mode']
    # A motor impedance choice, independent of the plant's contact stiffness.
    h.tangential_stiffness=args.get('tangent_kp',cartesian_kp)
    h.tangential_damping=args.get('tangent_kd',cartesian_kd)
    palm_shift_x=args.get('palm_shift_x',0.)
   if 'terminal'in decision:
    outcome='succeeded'if decision['terminal']=='succeeded'and hold_s>=.3 else'policy_failed';break
   if decision['action']!=phase:
    phase=decision['action'];phase_start=d.time
    if phase=='lift':anchor=[Rp.T@(p-d.xpos[palm])for p in points];home=d.qpos[h.qi].copy()
    if phase=='rotate':
     center_goal=center.copy();rotate_start=R.copy();anchor=[p-center for p in points]
     # Hand-off to in-hand motion at the achieved arm pose. Reset only motor
     # reference integrators; preserve every physical pose and velocity.
     if handoff=='adapter_reset':
      pgoal=d.xpos[palm].copy();arm_R_target=Rp.copy();qref=d.qpos[h.qi].copy();home=qref.copy()
      # Initialize a fresh contact adapter from its nominal pad coordinates,
      # then incorporate current tactile centroids. Physical state is untouched.
      for i,b in enumerate(fingers):
       local[i]=np.array(initial[i]['local']);Ri=d.xmat[b].reshape(3,3)
       if loads[i]>.002:local[i]+=.25*(Ri.T@(weighted[i]/loads[i]-d.xpos[b])-local[i])
       points[i]=d.xpos[b]+Ri@local[i]
      anchor=[p-center for p in points]
     rotation_pgoal=pgoal.copy();rotation_start_time=phase_start
    if phase=='hold':
     hold_R=R.copy() # Desired reference only; never assign the object's pose.
     if hold_reset:qref=d.qpos[h.qi].copy();home=qref.copy()
    # Hold recenters the index to remove the pusher moment and continues
    # correcting the goal orientation under increased side pressure.
  if failure:outcome='unexpected_collision'if unexpected>5 else'grasp_lost';break
  if phase=='lift':
   alpha=float(np.clip((d.time-phase_start)/1.5,0,1));alpha=alpha*alpha*(3-2*alpha);pgoal=p0+np.array([0,0,.08*alpha])
  if phase in ['rotate','hold']and palm_shift_x:
   alpha=float(np.clip((d.time-rotation_start_time)/1.5,0,1));alpha=alpha*alpha*(3-2*alpha)
   pgoal=rotation_pgoal+np.array([palm_shift_x*alpha,0,0])
  Jp=np.zeros((3,m.nv));Jr=Jp.copy();mujoco.mj_jacBody(m,d,Jp,Jr,palm)
  desired_arm=np.r_[np.clip(5*(pgoal-d.xpos[palm]),-.1,.1),5*Rotation.from_matrix(arm_R_target@Rp.T).as_rotvec()]
  A=np.vstack([Jp[:,h.vi[20:]],Jr[:,h.vi[20:]]]);vref=np.zeros(26);vref[20:]=np.clip(np.linalg.solve(A.T@A+1e-5*np.eye(6),A.T@desired_arm),-.8,.8)
  targetR=R0 if phase in ['acquire','lift']else hold_R if phase=='hold'and hold_mode=='freeze'else Rp@goalrel
  rot_error=Rotation.from_matrix(targetR@R.T).as_rotvec();omega_goal=2*rot_error-(hold_damping*omega if phase=='hold'else 0.);omega_goal*=min(1.,speed/max(1e-12,np.linalg.norm(omega_goal)))
  center_target=c0+d.xpos[palm]-p0 if phase in ['acquire','lift']else center_goal
  vcenter=Jp@d.qvel+4*(center_target-center)-.5*(Jobj@d.qvel-Jp@d.qvel)
  if phase in ['rotate','hold']and palm_shift_x:
   # Recenter the hand around the free object using motors. Hold its measured
   # world center through finger feedback while translating the palm.
   vcenter=4*(center_goal-center)-.5*(Jobj@d.qvel)
  normal_tasks=[];spin_tasks=[]
  for i,b in enumerate(fingers):
   point=points[i];n=normal(point,center,R);J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,point,b);cols=h.vi[4*i:4*i+4];qcols=h.qi[4*i:4*i+4]
   load=grip if i!=1 else (index_load if balanced else .08)
   if axis=='y'and phase=='rotate'and i!=1:
    # The speed brake is common to all adapters. The comparison should
    # isolate their motion constraints, not grant one an extra grip reflex.
    load=.4+float(np.clip(2*(np.linalg.norm(omega)-speed),0,.6))
   if phase=='acquire':load=min(load,.03+.8*max(0,d.time-.1))
   desired=vcenter+np.cross(omega_goal,point-center)
   if i==1:
    # Index approaches and stabilizes the top face; top-contact spin is free.
    offset=pusher_offset*min(1.,max(0.,(d.time-phase_start)/.5))if axis=='y'and phase=='rotate'else 0.
    target=center+R@np.array([edge/2,0,0]if balanced and edge==.02 else[offset,0,edge/2]);desired=vcenter+np.cross(omega_goal,target-center)+4*(target-point)
   if interface=='cartesian'and phase in ['rotate','hold']:
    fraction=min(1,(d.time-(phase_start if phase=='rotate'else rotation_start_time))/max(.001,np.radians(abs(goal_deg))/speed))
    rot=Rotation.from_rotvec(Rotation.from_matrix(Rp@goalrel@rotate_start.T).as_rotvec()*fraction).as_matrix()
    path_anchor=anchor[i]+(rotate_start@np.array([offset,0,0])if i==1 else 0.)
    desired=4*(center_goal+rot@path_anchor-point)
   desired-=n*np.dot(desired,n)
   rhs=finger_velocity_target(desired,J,h.vi[20:],vref[20:]);T=np.eye(3)-np.outer(n,n);A=T@J[:,cols];rhs=T@rhs
   if interface=='object':
    for direction in np.eye(3):spin_tasks.append((i,Jr,direction,float(direction@omega_goal),spin_gain))
    A=np.vstack([A,.01*Jr[:,cols]]);rhs=np.r_[rhs,.01*(omega_goal-Jr[:,h.vi[20:]]@vref[20:])]
   elif interface=='hom'and i!=1 and contact_mode=='face':
    pad_axis=np.array([.187702240,.982222897,-.00245951835])if pulp else np.array([1.,0,0]);facing=d.xmat[b].reshape(3,3)@pad_axis
    align_omega=omega_goal+4*np.cross(facing,-n)
    A=np.vstack([A,.012*T@Jr[:,cols]]);rhs=np.r_[rhs,.012*T@(align_omega-Jr[:,h.vi[20:]]@vref[20:])]
    for direction in T:
     if np.linalg.norm(direction)>.1:
      direction=direction/np.linalg.norm(direction);spin_tasks.append((i,Jr,direction,float(direction@align_omega),spin_gain))
   elif interface=='hom'and i!=1 and (contact_mode=='driven_spin'or not(axis=='y'and phase=='rotate')):
    spin_tasks.append((i,Jr,n,float(n@omega_goal),spin_gain))
    A=np.vstack([A,.01*n@Jr[:,cols]]);rhs=np.r_[rhs,.01*n@(omega_goal-Jr[:,h.vi[20:]]@vref[20:])]
   A=np.vstack([A,.002*np.eye(4)]);rhs=np.r_[rhs,.002*.3*(home[4*i:4*i+4]-d.qpos[qcols])]
   ids=[m.joint(f'joint_{i+1}_{j}').id for j in range(1,5)];lo=np.maximum(-1.,(m.jnt_range[ids,0]+1e-5-d.qpos[qcols])/.002);hi=np.minimum(1.,(m.jnt_range[ids,1]-1e-5-d.qpos[qcols])/.002)
   vref[4*i:4*i+4]=lsq_linear(A,rhs,bounds=(lo,hi)).x
   K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,point,bid)
   normal_tasks.append((J,n,load,float(loads[i]),i,float(n@(K@d.qvel))))
  if phase=='rotate':rotation_start_time=phase_start
  qref+=.002*vref;qref=np.clip(qref,d.qpos[h.qi]-.08,d.qpos[h.qi]+.08)
  for J,n,load,measured,i,_ in normal_tasks:
   sl=slice(4*i,4*i+4);Ji=J[:,h.vi[sl]];e=qref[sl]-d.qpos[h.qi[sl]];qref[sl]-=Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),n)*np.dot(n,Ji@e)
  t=float(d.time);observed_q=d.qpos.copy();observed_v=d.qvel.copy();h.motor_step(qref,vref,normal_tasks=normal_tasks,spin_tasks=spin_tasks)
  samples.append([t,error_deg,inhand_turn,palm_turn,lift_mm,*loads,float(np.degrees(np.linalg.norm(omega))),unexpected,hold_s,*h.last_effort]);refs.append(qref.copy())
  if k%10==0:times.append(t);poses.append(observed_q);vels.append(observed_v);commands.append(dict(t=t,phase=phase,orientation_error_deg=error_deg,inhand_turn_deg=inhand_turn,palm_turn_deg=palm_turn,lift_mm=lift_mm,loads_N=loads.tolist()))
  if k%100==0:
   progress=dict(t=t,phase=phase,error_deg=error_deg,inhand_turn_deg=inhand_turn,palm_turn_deg=palm_turn,lift_mm=lift_mm,loads_N=loads.tolist(),unexpected_N=unexpected,unexpected_pair=unexpected_pair,angular_speed_deg_s=float(np.degrees(np.linalg.norm(omega))),accepted_hold_s=hold_s,qpos=d.qpos.tolist(),wall_s=time.monotonic()-wall);print('PROGRESS',json.dumps(progress),flush=True);(OUT/(name+'.progress.json')).write_text(json.dumps(progress))
 report=dict(id=name,task='cube_rotation',interface=interface,outcome=outcome,success=outcome=='succeeded'and checkpoint is None,prepared_rotation_success=outcome=='succeeded'and checkpoint is not None,checkpoint=checkpoint,goal_deg=goal_deg,goal_axis=axis,graph=policy.graph,provenance=policy.manifest,agent_ticks=policy.ticks,transitions=policy.trace,commands=commands,initialization=initial,parameters=dict(hold_reset=hold_reset,hold_damping=hold_damping,hold_mode=hold_mode,adapter_handoff=handoff,initial_state_source=initial_from,dt=dt,initial_yaw_deg=yaw,edge_m=edge,mass_kg=edge**3*(400 if edge>.02 else 1000),balanced_pregrasp=balanced,pulp_contact=pulp,grip_target_N=grip,index_load_target_N=index_load,angular_speed_limit_rad_s=speed,pusher_offset_m=pusher_offset,relative_contact_mode=contact_mode,null_posture_kd=null_kd,normal_spin_motor_gain=spin_gain,tangential_stiffness_N_m=cartesian_kp,tangential_damping_N_s_m=cartesian_kd,pad_friction=.6,object_friction=.5,modulus_Pa=1.7e6,calibrated=False),final=dict(t=d.time,orientation_error_deg=error_deg,inhand_turn_deg=inhand_turn,palm_turn_deg=palm_turn,lift_mm=lift_mm,hold_s=hold_s,unexpected_N=unexpected,unexpected_pair=unexpected_pair),loaded_angular_path_deg=loaded_rotation,source_sha256=sources)
 (OUT/(name+'.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/(name+'.npz'),initial_qpos=q,qpos=poses,qvel=vels,time=times,samples=samples,qref_500hz=refs)
 print('RESULT',outcome,json.dumps(report['final']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--interface',choices=['cartesian','object','hom'],default='hom');p.add_argument('--duration',type=float,default=12);p.add_argument('--goal',type=float,default=90);p.add_argument('--yaw',type=float,default=0);p.add_argument('--dt',type=float,default=.00025);p.add_argument('--balanced',action='store_true');p.add_argument('--edge',type=float,default=.02);p.add_argument('--pulp',action='store_true');p.add_argument('--grip',type=float,default=.6);p.add_argument('--speed',type=float,default=.6);p.add_argument('--resume');p.add_argument('--resume-time',type=float,default=2.2);p.add_argument('--null-kd',type=float,default=.01);p.add_argument('--spin-gain',type=float,default=.06);p.add_argument('--cartesian-kp',type=float);p.add_argument('--cartesian-kd',type=float,default=3.);p.add_argument('--axis',choices=['x','y','z'],default='z');p.add_argument('--pusher-offset',type=float,default=0);p.add_argument('--initial-from');p.add_argument('--policy');p.add_argument('--index-load',type=float,default=.15);p.add_argument('--hold-reset',action='store_true');p.add_argument('--hold-damping',type=float,default=0.);p.add_argument('--hold-mode',choices=['track','freeze'],default='track');p.add_argument('--handoff',choices=['continuous','adapter_reset'],default='continuous');p.add_argument('--contact-mode',choices=['spin','face','driven_spin'],default='spin');a=p.parse_args();run(a.name,a.interface,a.duration,a.goal,a.yaw,a.dt,a.balanced,a.edge,a.pulp,a.grip,a.speed,a.resume,a.resume_time,a.null_kd,a.spin_gain,a.cartesian_kp,a.cartesian_kd,a.axis,a.pusher_offset,a.contact_mode,a.initial_from,a.policy,a.index_load,a.handoff,a.hold_mode,a.hold_damping,a.hold_reset)
