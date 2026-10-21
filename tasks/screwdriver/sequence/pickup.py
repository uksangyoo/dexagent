"""Actuator-only pickup / gravity-pivot development, with guarded phases."""
import json,time,hashlib,sys,signal
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
from dexagent.planning.initialization import initial_state
from dexagent.planning.geometry import ART
from dexagent.planning.kinematics import finger_velocity_target,tracked_contact_velocity
from dexagent.planning.grasp_plan import balanced_grasp
from dexagent.planning.policy import Policy
from tasks.screwdriver.sequence.assembly import AssemblyBridge
from dexagent.planning.arm_plan import solve as audit_arm_corridor
from dexagent.sim.model import OUT
from dexagent.sim.drake_backend import HydroPlant
from dexagent.sim.controllers import unit


def cap_normal(point,center,axis,radius,half_height):
    relative=point-center;height=float(relative@axis);radial=relative-axis*height
    return np.sign(height)*axis if abs(height)-half_height>np.linalg.norm(radial)-radius else unit(radial)


def run(duration=8.,name='sequence-pivot-v9',tracking='material',cap_side_load=.8,lift_load=1.3,contact_jacobian=False,grasp='tip',early_support=False,null_posture_kp=.1,align_pad=True,policy_bundle=None,assembly=False,drive_goal=90,initial_yaw_deg=0,debug_transport_from=None,transport_speed=.06,seat_clearance_mm=None,physics_dt=.00025,thread_segments=48,debug_frame=0,pickup_y=.1,pickup_x=-.22,require_clear_corridor=False,debug_stage="transport",drive_cycle="alternating",drive_load=.4,return_load=.06,drive_speed=45,reserve_margin=0.,drive_tangent_kp=None,drive_tangent_kd=3.,alignment_filter_s=0.):
    if (OUT/(name+'.json')).exists():raise FileExistsError(name)
    policy=Policy(policy_bundle)if policy_bundle else None
    interface=policy.graph['abstraction']if policy else'hom'
    sources={};snapshot=OUT/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for key,module in list(sys.modules.items()):
        if key.startswith(('dexagent.','tasks.'))and getattr(module,'__file__',None):
            content=getattr(module,'__source_bytes__',Path(module.__file__).read_bytes());digest=hashlib.sha256(content).hexdigest();filename=Path(module.__file__).name;sources[filename]=digest;(snapshot/(digest+'_'+filename)).write_bytes(content)
    content=Path(__file__).read_bytes();digest=hashlib.sha256(content).hexdigest();sources['pickup_trial.py']=digest;(snapshot/(digest+'_pickup_trial.py')).write_bytes(content)
    model_path=ART/('precision-slotted.xml'if thread_segments==48 else f'precision-slotted-s{thread_segments}.xml')
    xml,q,initial=initial_state(model_path,True,output_stem=name+'-initial',pickup_y=pickup_y,pickup_x=pickup_x)
    if seat_clearance_mm is not None:
        from dexagent.sim.seating import geometry
        mm=mujoco.MjModel.from_xml_path(str(xml));dd=mujoco.MjData(mm);dd.qpos[:]=q;mujoco.mj_forward(mm,dd)
        shift=(seat_clearance_mm-geometry(mm,dd)['flat_seat_remaining_mm'])/1000;qa=mm.joint('screw_free').qposadr[0]
        q[qa+2]+=shift;q[qa+3:qa+7]=(Rotation.from_quat(q[qa+3:qa+7],scalar_first=True)*Rotation.from_euler('z',2*np.pi*shift/.0005)).as_quat(scalar_first=True)
        dd.qpos[:]=q;mujoco.mj_forward(mm,dd)
        bad=[dict(pair=[mm.geom(c.geom1).name,mm.geom(c.geom2).name],depth_mm=-float(c.dist)*1000)for c in dd.contact if c.dist< -1e-6]
        if bad:raise RuntimeError('Reject penetrating started-screw state: '+str(bad[:5]))
        print('STARTED_SCREW_CASE',json.dumps(dict(seat_clearance_mm=seat_clearance_mm,initial_shift_mm=shift*1000,initial_rotation_deg=shift/.0005*360,penetrations=len(bad))),flush=True)
    grasp_plan=None
    if grasp=='pulp':
        from dexagent.planning.grasp_plan import balanced_grasp
        q,initial,grasp_plan=balanced_grasp(xml,q,initial,pulp=True,starts=2)
        if grasp_plan['initial_collisions']:raise RuntimeError('Reject colliding initial grasp')
        print('GRASP_PLAN',json.dumps(grasp_plan),flush=True)
    if initial_yaw_deg:
        audit_model=mujoco.MjModel.from_xml_path(str(xml));qa=audit_model.joint('tool_free').qposadr[0]
        q[qa+3:qa+7]=(Rotation.from_euler('z',initial_yaw_deg,degrees=True)*Rotation.from_quat(q[qa+3:qa+7],scalar_first=True)).as_quat(scalar_first=True)
        audit_data=mujoco.MjData(audit_model);audit_data.qpos[:]=q;mujoco.mj_forward(audit_model,audit_data)
        if any(c.dist<-.000001 for c in audit_data.contact):raise RuntimeError('Reject penetrating perturbed start')
    if require_clear_corridor:audit_arm_corridor(xml,q,ART/(name+'-arm-plan.json'),max_starts=1)
    frozen=OUT/(name+'.xml');frozen.write_bytes(xml.read_bytes());xml=frozen
    h=HydroPlant(xml,dt=physics_dt,resolution=.002,explicit_steel=True,beta=0,normal_force_limit=max(1.5,1.5*lift_load),null_posture_kp=null_posture_kp);h.initialize_from_mujoco(q);h.sync()
    bridge=AssemblyBridge(h,drive_goal,transport_speed,interface,drive_cycle,drive_load,return_load,drive_speed,reserve_margin,drive_tangent_kp,drive_tangent_kd,alignment_filter_s)if assembly else None
    d=h.d;m=h.m;tool=m.body('screwdriver').id;palm=m.body('palm').id
    cap_geom=m.geom('cap').id;cap_body=int(m.geom_bodyid[cap_geom]);cap_radius,cap_half_height=m.geom_size[cap_geom,:2]
    bids=[m.body(f'link_{i}_4').id for i in [1,2,3]]
    local=[np.array(initial[0]['local']),np.array(grasp_plan['index_local']if grasp_plan else[.0357,-.0029,.0016]),np.array(initial[1]['local'])]
    home=d.qpos[h.qi].copy();qref=home.copy();Rp=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy();pgoal=p0.copy();tool_center0=d.xpos[tool].copy()
    points=[d.xpos[b]+d.xmat[b].reshape(3,3)@l for b,l in zip(bids,local)]
    anchors=[Rp.T@(p-p0)for p in points];phase='pinch';phase_start=0.;ready=0.;trace=[];samples=[];poses=[];velocities=[];times=[];commands=[];outcome='timeout';grasp_reference=None;wall=time.monotonic();closing=False;light_ready=0.;pinch_target=.03;cap_acquired=False;index_approach_start=None;cap_lost_s=0.;support_hold=0.;lost_contact=0.;lift_hold=0.;reference_rotation=d.xmat[tool].reshape(3,3).copy();recorded_refs=[];recorded_vrefs=[]
    checkpoint_info=None
    if debug_transport_from:
        if not bridge:raise ValueError('Transport checkpoint requires --assembly')
        source=OUT/(debug_transport_from+'.npz');record=json.loads((OUT/(debug_transport_from+'.json')).read_text());saved=np.load(source)
        eligible=[i for i,c in enumerate(record['commands'])if c['phase']==debug_stage]
        if not eligible:raise ValueError('No physically recorded requested stage')
        index=eligible[debug_frame];h.initialize_from_mujoco(saved['qpos'][index],saved['qvel'][index]);h.sync()
        home=d.qpos[h.qi].copy();qref=home.copy();grasp_reference=home.copy();Rp=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy();pgoal=p0.copy();tool_center0=d.xpos[tool].copy()
        points=[d.xpos[b]+d.xmat[b].reshape(3,3)@l for b,l in zip(bids,local)];anchors=[Rp.T@(p-p0)for p in points]
        index_approach_start=-1.;closing=True;phase='transport';bridge.start()
        if debug_stage in ['drive','drive_hold']:
            # Query actual contact geometry at the recorded physical state.
            # Only controller references change; no simulator advance/reset.
            from pydrake.all import HydroelasticContactRepresentation
            query=h.scene.get_query_output_port().Eval(h.scene.GetMyContextFromRoot(h.context));ins=query.inspector()
            surfaces,_=query.ComputeContactSurfacesWithFallback(HydroelasticContactRepresentation.kTriangle)
            found=set()
            for surface in surfaces:
                names={ins.GetName(g).split('::')[-1]for g in [surface.id_M(),surface.id_N()]}
                for i,target in [(0,'handle'),(1,'cap'),(2,'handle')]:
                    if names=={f'finger_tip_{i+1}',target}:
                        points[i]=surface.centroid();local[i]=d.xmat[bids[i]].reshape(3,3).T@(points[i]-d.xpos[bids[i]]);found.add(i)
            if found!={0,1,2}:raise ValueError('Prepared drive requires three actual contact patches')
            bridge.start_drive(local,points);phase='drive'
            if debug_stage=='drive_hold':
                bridge.controller.hold_at_contact();bridge.hold_only=True;bridge.hold_reference_angles=np.zeros(2);bridge.route('drive_hold');phase='drive_hold'
        checkpoint_info=dict(source=debug_transport_from,time=float(saved['time'][index]),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),controller_reinitialized=True,full_task_credit=False)
    def route(target):
        nonlocal phase,phase_start,anchors,grasp_reference
        trace.append(dict(t=d.time,from_node=phase,to=target));phase=target;phase_start=d.time
        if target=='lift':
            anchors=[d.xmat[palm].reshape(3,3).T@(p-d.xpos[palm])for p in points];grasp_reference=d.qpos[h.qi].copy()
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for k in range(round(duration/.002)):
        if stopped and k>0:outcome='interrupted';break
        h.sync();axis=d.xmat[tool].reshape(3,3)[:,2];center=d.xpos[tool].copy();loads=np.zeros(3);cap_top_load=0.;weighted=np.zeros((3,3));unexpected=0.;unexpected_names=[]
        cap_center=d.xpos[cap_body]+d.xmat[cap_body].reshape(3,3)@m.geom_pos[cap_geom];cap_top=cap_center+cap_half_height*axis
        for c in h.contacts():
            names=set(c['names']);allowed=False
            for i,geo in [(0,'handle'),(1,'cap'),(2,'handle')]:
                if names=={f'finger_tip_{i+1}',geo}:
                    n=cap_normal(c['point'],cap_center,axis,cap_radius,cap_half_height) if i==1 else unit(c['point']-center-axis*np.dot(c['point']-center,axis));fn=abs(np.dot(c['force'],n));loads[i]+=fn;weighted[i]+=fn*c['point'];allowed=True
                    if i==1 and np.dot(n,axis)>.8:cap_top_load+=fn
            if not allowed and any(n.startswith(('finger_tip_','source_geometry_','arm_collision_'))for n in names):
                force=float(np.linalg.norm(c['force']))
                if force>unexpected:unexpected=force;unexpected_names=c['names']
        for i,b in enumerate(bids):
            R=d.xmat[b].reshape(3,3)
            if loads[i]>.002 and (i==1 or grasp_reference is None or tracking in ['frozen','object_center']):local[i]+=.25*(R.T@(weighted[i]/loads[i]-d.xpos[b])-local[i])
            points[i]=d.xpos[b]+R@local[i]
        tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
        light_ready=light_ready+.002 if min(loads[[0,2]])>.015 else 0.
        if light_ready>.1:closing=True
        pinch_target+=(1-np.exp(-.002/.15))*((lift_load if closing else .03)-pinch_target)
        ready=ready+.002 if min(loads[[0,2]])>.2 else 0.
        if policy is None and phase=='pinch' and ready>.15:
            anchors=[d.xmat[palm].reshape(3,3).T@(p-d.xpos[palm])for p in points];grasp_reference=d.qpos[h.qi].copy();route('lift')
        if phase=='lift':
            alpha=np.clip((d.time-phase_start)/2.,0,1);alpha=alpha*alpha*(3-2*alpha);pgoal=p0+np.array([0,0,.14*alpha])
            if policy is None and d.time-phase_start>2.2 and d.xpos[palm,2]-p0[2]>.135:route('release')
        if early_support and phase in ['lift','release'] and tilt<35 and (center-.123*axis)[2]>.435:route('cap_support')
        if policy is None and phase=='release' and d.time-phase_start>.4 and loads[1]<.005:route('pivot')
        if policy is None and phase=='pivot' and tilt<35:route('cap_support')
        if phase=='cap_support' and tilt<35 and index_approach_start is None:index_approach_start=d.time
        Jtool=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,Jtool,tool)
        angular=Jtool@d.qvel
        pivot_speed=np.linalg.norm(angular-axis*np.dot(axis,angular))
        target_speed=min(.5,max(.08,np.radians(tilt)*1.2))
        pivot_load=.30+np.clip(2.0*(pivot_speed-target_speed),0,1.0)
        if tilt<10:pivot_load=1.3
        if phase=='cap_support' and cap_top_load>.02:pivot_load=cap_side_load
        if interface in ['cartesian','object']:pivot_load=.3 if tilt>=10 else 1.3
        if interface in ['cartesian','object']and cap_top_load>.02:pivot_load=cap_side_load
        supported=(phase=='cap_support' and tilt<5 and cap_top_load>.025 and min(loads[[0,2]])>.1 and np.linalg.norm(angular)<np.radians(5) and unexpected<.1)
        support_hold=support_hold+.002 if supported else 0.
        lost_contact=lost_contact+.002 if phase!='pinch' and min(loads[[0,2]])<.01 else 0.
        if policy is None and support_hold>=.3 and not(bridge and bridge.active):
            if bridge:bridge.start();route('transport')
            else:outcome='pickup_cap_supported';break
        if bridge and bridge.active:
            arm_velocity,arm_omega=bridge.update(local,points)
            if bridge.outcome and (policy is None or bridge.outcome!='full_task_succeeded'):outcome=bridge.outcome;break
            if phase!=bridge.stage:route(bridge.stage)
            if bridge.controller is None:
                pgoal=d.xpos[palm]+arm_velocity/5
                Rp=Rotation.from_rotvec(arm_omega/5).as_matrix()@d.xmat[palm].reshape(3,3)
        if lost_contact>.1:outcome='grasp_lost';break
        rotation_error=float(np.degrees(Rotation.from_matrix(d.xmat[tool].reshape(3,3)@reference_rotation.T).magnitude()))
        lift_ok=(center[2]-tool_center0[2]>=.12 and rotation_error<=10 and min(loads[[0,2]])>.05 and np.linalg.norm(angular)<np.radians(5) and unexpected<.1)
        lift_hold=lift_hold+.002 if lift_ok else 0.
        if policy is not None and k%25==0:
            obs=dict(t=float(d.time),assembly_accepted=bool(bridge and bridge.outcome=='full_task_succeeded'),loaded_s=ready,lift_ready=bool(d.time-phase_start>2.2 and d.xpos[palm,2]-p0[2]>.135),release_ready=bool(d.time-phase_start>.4 and loads[1]<.005),tilt_deg=tilt,lift_accepted=lift_hold>=.3,pivot_accepted=support_hold>=.3,failure=bool(unexpected>5 or lost_contact>.1 or center[2]<.32))
            decision=policy.step(obs)
            if 'terminal'in decision:
                accepted=(bridge and bridge.outcome=='full_task_succeeded')if policy.graph['task']=='pickup_pivot_drive'else lift_hold>=.3 if policy.graph['task']=='lift_hold'else support_hold>=.3
                outcome=('full_task_succeeded'if policy.graph['task']=='pickup_pivot_drive'else'subgoal_succeeded')if decision['terminal']=='succeeded'and accepted else'policy_failed';break
            if decision['action']=='assemble':
                if bridge is None:raise ValueError('Full task graph requires --assembly')
                if not bridge.active and 'command'in decision:
                    from dexagent.planning.workflow import validate_command
                    bridge.drive_parameters=validate_command(decision['command'],interface)
                if not bridge.active:bridge.start();route('transport')
            elif phase!=decision['action']:route(decision['action'])

        vref=np.zeros(26);normal_tasks=[];spin_tasks=[];Rpal=d.xmat[palm].reshape(3,3)
        Jp=np.zeros((3,m.nv));Jr=Jp.copy();mujoco.mj_jacBody(m,d,Jp,Jr,palm)
        desired=np.r_[np.clip(5*(pgoal-d.xpos[palm]),-.15,.15),np.clip(5*Rotation.from_matrix(Rp@Rpal.T).as_rotvec(),-.3,.3)]
        A=np.vstack([Jp[:,h.vi[20:]],Jr[:,h.vi[20:]]]);vref[20:]=np.clip(np.linalg.solve(A.T@A+1e-5*np.eye(6),A.T@desired),-.8,.8)
        for i,b in enumerate(bids):
            if i==1 and phase not in ['pinch','lift','hold_horizontal','release','pivot','cap_support','transport','approach','align_slot','drive','drive_hold']:continue
            point=points[i];radial=point-center;n=(cap_normal(point,cap_center,axis,cap_radius,cap_half_height) if loads[1]>.002 else (np.array([0.,0.,1.])if phase in ['pinch','lift','hold_horizontal']else axis)) if i==1 else unit(radial-axis*np.dot(radial,axis))
            J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,point,b);cols=h.vi[4*i:4*i+4]
            if i==1:
                horizontal_support=False
                anticipate=tilt>15
                offset=float((cap_top-center)@axis)
                approach_clearance=.010 if index_approach_start is None else max(0.,.010-.015*(d.time-index_approach_start))
                cap=cap_top+axis*approach_clearance if index_approach_start is not None else center+np.array([0,0,offset+.010])
                error=cap-point;K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,center if anticipate else cap,tool);v=K@d.qvel+np.clip(6*error,-.04,.04)
                cap_lost_s=cap_lost_s+.002 if loads[1]<.005 else 0.
                if cap_lost_s>.05:cap_acquired=False
                if loads[1]>.01 and np.linalg.norm(error)<.01:cap_acquired=True
                if phase!='release' and approach_clearance<.001 and (cap_top_load>.0005 or (np.linalg.norm((point-cap_top)-axis*np.dot(point-cap_top,axis))<.003 and abs(np.dot(point-cap_top,axis))<.0005)):
                    axis_error=np.r_[center[:2]-cap_top[:2],0.]
                    Jcenter=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,Jcenter,None,tool);center_v=Jcenter@d.qvel
                    correction=8*axis_error-1.5*(K@d.qvel-center_v)+.5*error
                    v=tracked_contact_velocity(center_v,correction)if bridge and bridge.active else np.clip(center_v+correction,-.04,.04)
                    v-=n*np.dot(v,n);normal_tasks.append((J,n,float(np.clip(.035+.3*axis[2]-.04*angular[1],.005,.15)) if horizontal_support else .15,float(loads[i]),i))
            else:
                target=d.xpos[palm]+Rpal@anchors[i];K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,target,palm)
                v=6*(target-point)+K@d.qvel
                if tracking=='object_center':
                    Jcenter=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,Jcenter,None,tool)
                    center_goal=tool_center0+d.xpos[palm]-p0
                    palm_velocity=Jp@d.qvel
                    v=palm_velocity+4*(center_goal-center)-.8*(Jcenter@d.qvel-palm_velocity)
                v-=n*np.dot(v,n)
                Jforce=J
                if contact_jacobian and loads[i]>.002:
                    contact_point=weighted[i]/loads[i];Jforce=np.zeros((3,m.nv));mujoco.mj_jac(m,d,Jforce,None,contact_point,b)
                    rcontact=contact_point-center;n=unit(rcontact-axis*np.dot(rcontact,axis))
                normal_tasks.append((Jforce,n,(.8 if bridge and bridge.active else pivot_load) if phase in ['pivot','cap_support','transport','approach','align_slot','drive','drive_hold']else pinch_target,float(loads[i]),i))
            # A finger's world velocity already includes the arm contribution.
            # Solve only the remaining velocity with finger joints.
            v=finger_velocity_target(v,J,h.vi[20:],vref[20:])
            A=J[:,cols];rhs=v
            if i!=1:
                # Keep the exposed pad facing the object; free contact-normal spin during gravity pivot.
                facing=d.xmat[b].reshape(3,3)@np.array(grasp_plan['pad_normal_local']if grasp_plan else[1.,0,0]);desired_rotation=4*np.cross(facing,-n)
                tangent_rotation=np.eye(3)-np.outer(n,n)
                A=np.vstack([(np.eye(3)-np.outer(n,n))@A,.012*tangent_rotation@Jr[:,cols]])
                rhs=np.r_[(np.eye(3)-np.outer(n,n))@v,.012*desired_rotation]
                for direction in (tangent_rotation if align_pad else []):
                    if np.linalg.norm(direction)>.1:
                        direction=unit(direction);spin_tasks.append((i,Jr,direction,float(direction@desired_rotation),.06))
                if interface=='hom' and phase in ['pinch','lift','hold_horizontal','release']:spin_tasks.append((i,Jr,n,0.,.06))
                if interface=='object':
                    rigid_omega=np.zeros(3)if phase in ['pinch','lift','hold_horizontal','release']else 1.2*np.cross(axis,[0.,0.,1.])
                    for direction in np.eye(3):spin_tasks.append((i,Jr,direction,float(direction@rigid_omega),.06))
            if False and i==0 and phase in ['pivot','cap_support']:
                rotation=np.cross(axis,[0,0,1]);goal=rotation*min(2.,.35/max(1e-9,np.linalg.norm(rotation)))
                spin=float(np.dot(n,goal));spin_tasks.append((i,Jr,n,spin,.2))
                A=np.vstack([A,.01*n@Jr[:,cols]]);rhs=np.r_[rhs,.01*spin]
            A=np.vstack([A,.002*np.eye(4)]);rhs=np.r_[rhs,.002*.3*(home[4*i:4*i+4]-d.qpos[h.qi[4*i:4*i+4]])]
            ids=[m.joint(f'joint_{i+1}_{j}').id for j in range(1,5)];lo=np.maximum(-1.,(m.jnt_range[ids,0]+1e-5-d.qpos[h.qi[4*i:4*i+4]])/.002);hi=np.minimum(1.,(m.jnt_range[ids,1]-1e-5-d.qpos[h.qi[4*i:4*i+4]])/.002)
            vref[4*i:4*i+4]=lsq_linear(A,rhs,bounds=(lo,hi)).x
            if i!=1 and grasp_reference is not None and tracking=='frozen':vref[4*i:4*i+4]=0
        qref+=vref*.002
        if grasp_reference is not None and tracking=='frozen':
            qref[:4]=grasp_reference[:4];qref[8:12]=grasp_reference[8:12]
        qref=np.clip(qref,d.qpos[h.qi]-.08,d.qpos[h.qi]+.08)
        for task in normal_tasks:
            J,n,_,_,finger=task;sl=slice(4*finger,4*finger+4);Ji=J[:,h.vi[sl]];error=qref[sl]-d.qpos[h.qi[sl]]
            qref[sl]-=Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),n)*np.dot(n,Ji@error)
        observed_time=d.time;observed_q=d.qpos.copy();observed_v=d.qvel.copy()
        if bridge and bridge.controller is not None:
            qref,vref=bridge.controller.tick();h.motor_step(qref,vref,normal_tasks=bridge.controller.normal_tasks)
        else:
            if bridge and bridge.active:
                for index,task in enumerate(normal_tasks):
                    J,n,target,measured,finger=task;K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,points[finger],tool)
                    normal_tasks[index]=(*task,float(n@(K@d.qvel)))
            h.motor_step(qref,vref,normal_tasks=normal_tasks,spin_tasks=spin_tasks)
        recorded_refs.append(qref.copy());recorded_vrefs.append(vref.copy())
        assembly_values=bridge.last if bridge else {}
        slips=bridge.controller.slips.tolist()if bridge and bridge.controller else[0,0,0]
        sample=[observed_time,assembly_values.get('tool_deg',0),assembly_values.get('screw_deg',0),assembly_values.get('depth_mm',0),*loads,*slips,0,0,0,tilt,*h.last_effort,unexpected,cap_top_load,pivot_speed,support_hold,rotation_error,lift_hold];samples.append(sample)
        if k%10==0:poses.append(observed_q);velocities.append(observed_v);times.append(observed_time);commands.append(dict(phase=phase,macro_phase=phase,assembly=assembly_values.copy(),drive_command=bridge.controller.last_command if bridge and bridge.controller else None))
        if k%50==0:
            progress=dict(t=d.time,phase=phase,assembly=assembly_values,cap_top_N=cap_top_load,pivot_speed_rad_s=float(pivot_speed),support_hold_s=support_hold,loads_N=loads.tolist(),tilt_deg=tilt,tool_center=d.xpos[tool].tolist(),unexpected_N=float(unexpected),unexpected_pair=unexpected_names,wall_s=time.monotonic()-wall);print('PROGRESS',json.dumps(progress),flush=True);(OUT/(name+'.progress.json')).write_text(json.dumps(progress))
        if k>0 and k%500==0:
            partial=OUT/(name+'.partial.npz');np.savez_compressed(partial,initial_qpos=q,qpos=poses,qvel=velocities,time=times,samples=samples,qref_500hz=recorded_refs,vref_500hz=recorded_vrefs)
            (OUT/(name+'.partial.json')).write_text(json.dumps(dict(id=name,scope='incomplete physical pickup development',interface=interface,outcome='running',success=False,checkpoint=checkpoint_info,commands=commands,source_sha256=sources,assembly_trace=bridge.trace if bridge else [],assembly_final=bridge.last if bridge else None),indent=2))
        if unexpected>5:outcome='unexpected_collision';break
        if d.xpos[tool,2]<.32:outcome='tool_dropped';break
    report=dict(id=name,scope='physical pickup development; continuous assembly attempt'if assembly else'physical pickup development; no screw-in credit',checkpoint=checkpoint_info,interface=interface,policy_graph=policy.graph if policy else None,policy_provenance=policy.manifest if policy else None,agent_ticks=policy.ticks if policy else [],agent_transitions=policy.trace if policy else [],lift_hold_s=lift_hold,outcome=outcome,success=outcome=='full_task_succeeded'and checkpoint_info is None,assembly_trace=bridge.trace if bridge else [],assembly_final=bridge.last if bridge else None,drive_goal_deg=drive_goal if bridge else None,workspace_plan=bridge.workspace_plan if bridge else None,pickup_success=bool(bridge and bridge.active)or outcome=='pickup_cap_supported' or(outcome=='subgoal_succeeded'and policy.graph['task']=='controlled_pivot'),subgoal_success=outcome=='subgoal_succeeded',support_hold_s=support_hold,source_sha256=sources,initialization=initial,grasp_plan=grasp_plan,trace=trace,commands=commands,parameters=dict(drive_controller=bridge.drive_parameters if bridge else None,arm_corridor_preflight=require_clear_corridor,initial_pickup_x_m=pickup_x,initial_pickup_y_m=pickup_y,initial_seat_clearance_mm=seat_clearance_mm,transport_speed_limit_m_s=transport_speed,initial_yaw_perturbation_deg=initial_yaw_deg,initial_robot_refit_for_perturbation=False,model_revision='precision-slotted',lift_normal_N=lift_load,normal_force_limit_N=h.normal_force_limit,lift_height_m=.14,pad_normal_alignment=align_pad,null_posture_kp=null_posture_kp,index_supports_horizontal_cap_during_lift=False,index_approach='from above measured cap; top-surface contact required for pressure',side_joint_reference_frozen_after_acquisition=tracking=='frozen',side_tracking=tracking,force_jacobian_at_actual_contact=contact_jacobian,early_support_when_tip_clear=early_support,cap_alignment_gain_s=8,cap_contact_centering_gain_s=.5,cap_alignment_side_load_N=cap_side_load,subtract_arm_velocity_in_finger_ik=True,pivot_brake='pressure feedback on measured angular speed; no fingertip spin task',dt=physics_dt,thread_segments=thread_segments,pivot_normal_N=[.3,1.3],pivot_brake_gain_N_s=2.,pivot_brake_threshold_rad_s=.5,cap_normal_N=.15,cap_contact_normal='finite-cylinder signed-distance gradient',cap_surface_tracking_gain_s=4,cap_reacquire_after_s=.05,cap_approach_phase='lift: preposition above predicted upright tool until tilt below 15 deg',cap_support_phase_tilt_deg=30,cap_target='actual cylinder top from model geometry',thumb_spin_gain_Nm_s=0,middle_spin='released',sample_timing='state and observations before the 2 ms motor interval; effort applied during that interval'),final=dict(t=d.time,angle_deg=[0,0],depth_mm=0,tilt_deg=tilt,unexpected_load_N=float(unexpected),unexpected_pair=unexpected_names))
    (OUT/(name+'.xml')).write_bytes(xml.read_bytes());(OUT/(name+'.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/(name+'.npz'),initial_qpos=q,qpos=poses,qvel=velocities,time=times,samples=samples,qref_500hz=recorded_refs,vref_500hz=recorded_vrefs)
    print('RESULT',outcome,report['final'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='sequence-pivot-v9');p.add_argument('--duration',type=float,default=8);p.add_argument('--tracking',choices=['material','frozen','object_center'],default='material');p.add_argument('--cap-side-load',type=float,default=.8);p.add_argument('--lift-load',type=float,default=1.3);p.add_argument('--contact-jacobian',action='store_true');p.add_argument('--grasp',choices=['tip','pulp'],default='tip');p.add_argument('--early-support',action='store_true');p.add_argument('--null-posture-kp',type=float,default=.1);p.add_argument('--no-pad-alignment',action='store_true');p.add_argument('--policy',type=str);p.add_argument('--assembly',action='store_true');p.add_argument('--drive-goal',type=float,default=90);p.add_argument('--initial-yaw',type=float,default=0);p.add_argument('--debug-transport-from');p.add_argument('--transport-speed',type=float,default=.06);p.add_argument('--seat-clearance-mm',type=float);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--thread-segments',type=int,default=48);p.add_argument('--debug-frame',type=int,default=0);p.add_argument('--debug-stage',choices=['transport','approach','align_slot','drive','drive_hold'],default='transport');p.add_argument('--pickup-y',type=float,default=.1);p.add_argument('--pickup-x',type=float,default=-.22);p.add_argument('--require-clear-corridor',action='store_true');p.add_argument('--drive-cycle',choices=['paired','alternating'],default='alternating');p.add_argument('--drive-load',type=float,default=.4);p.add_argument('--return-load',type=float,default=.06);p.add_argument('--drive-speed',type=float,default=45);p.add_argument('--reserve-margin',type=float,default=0.);p.add_argument('--drive-tangent-kp',type=float);p.add_argument('--drive-tangent-kd',type=float,default=3.);p.add_argument('--alignment-filter',type=float,default=0.);a=p.parse_args();run(a.duration,a.name,a.tracking,a.cap_side_load,a.lift_load,a.contact_jacobian,a.grasp,a.early_support,a.null_posture_kp,not a.no_pad_alignment,a.policy,a.assembly,a.drive_goal,a.initial_yaw,a.debug_transport_from,a.transport_speed,a.seat_clearance_mm,a.physics_dt,a.thread_segments,a.debug_frame,a.pickup_y,a.pickup_x,a.require_clear_corridor,a.debug_stage,a.drive_cycle,a.drive_load,a.return_load,a.drive_speed,a.reserve_margin,a.drive_tangent_kp,a.drive_tangent_kd,a.alignment_filter)
