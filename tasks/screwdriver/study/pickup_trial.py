"""Actuator-only pickup / gravity-pivot development, with guarded phases."""
import json,time,hashlib,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
from tasks.screwdriver.study.pickup import initial_state
from dexagent.sim.model import OUT
from dexagent.sim.drake_backend import HydroPlant
from dexagent.sim.controllers import unit


def cap_normal(point,center,axis,radius,half_height):
    relative=point-center;height=float(relative@axis);radial=relative-axis*height
    return np.sign(height)*axis if abs(height)-half_height>np.linalg.norm(radial)-radius else unit(radial)


def run(duration=10.,name='pickup-trial49'):
    if (OUT/(name+'.json')).exists():raise FileExistsError(name)
    sources={};snapshot=OUT/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for key,module in list(sys.modules.items()):
        if key.startswith(('dexagent.','tasks.'))and getattr(module,'__file__',None):
            content=getattr(module,'__source_bytes__',Path(module.__file__).read_bytes());digest=hashlib.sha256(content).hexdigest();filename=Path(module.__file__).name;sources[filename]=digest;(snapshot/(digest+'_'+filename)).write_bytes(content)
    content=Path(__file__).read_bytes();digest=hashlib.sha256(content).hexdigest();sources['pickup_trial.py']=digest;(snapshot/(digest+'_pickup_trial.py')).write_bytes(content)
    xml,q,initial=initial_state();frozen=OUT/(name+'.xml');frozen.write_bytes(xml.read_bytes());xml=frozen
    h=HydroPlant(xml,dt=.00025,resolution=.002,explicit_steel=True,beta=0);h.initialize_from_mujoco(q);h.sync()
    d=h.d;m=h.m;tool=m.body('screwdriver').id;palm=m.body('palm').id
    cap_geom=m.geom('cap').id;cap_body=int(m.geom_bodyid[cap_geom]);cap_radius,cap_half_height=m.geom_size[cap_geom,:2]
    bids=[m.body(f'link_{i}_4').id for i in [1,2,3]]
    local=[np.array(initial[0]['local']),np.array([.0357,0,.0016]),np.array(initial[1]['local'])]
    home=d.qpos[h.qi].copy();qref=home.copy();Rp=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy();pgoal=p0.copy()
    points=[d.xpos[b]+d.xmat[b].reshape(3,3)@l for b,l in zip(bids,local)]
    anchors=[Rp.T@(p-p0)for p in points];phase='pinch';phase_start=0.;ready=0.;trace=[];samples=[];poses=[];velocities=[];times=[];commands=[];outcome='timeout';wall=time.monotonic();closing=False;light_ready=0.;pinch_target=.03;cap_acquired=False;cap_lost_s=0.;support_hold=0.;lost_contact=0.
    def route(target):
        nonlocal phase,phase_start
        trace.append(dict(t=d.time,from_node=phase,to=target));phase=target;phase_start=d.time
    for k in range(round(duration/.002)):
        h.sync();axis=d.xmat[tool].reshape(3,3)[:,2];center=d.xpos[tool].copy();loads=np.zeros(3);weighted=np.zeros((3,3));unexpected=0.;unexpected_names=[]
        cap_center=d.xpos[cap_body]+d.xmat[cap_body].reshape(3,3)@m.geom_pos[cap_geom];cap_top=cap_center+cap_half_height*axis
        for c in h.contacts():
            names=set(c['names']);allowed=False
            for i,geo in [(0,'handle'),(1,'cap'),(2,'handle')]:
                if names=={f'finger_tip_{i+1}',geo}:
                    n=cap_normal(c['point'],cap_center,axis,cap_radius,cap_half_height) if i==1 else unit(c['point']-center-axis*np.dot(c['point']-center,axis));fn=abs(np.dot(c['force'],n));loads[i]+=fn;weighted[i]+=fn*c['point'];allowed=True
            if not allowed and any(n.startswith(('finger_tip_','source_geometry_','arm_collision_'))for n in names):
                force=float(np.linalg.norm(c['force']))
                if force>unexpected:unexpected=force;unexpected_names=c['names']
        for i,b in enumerate(bids):
            R=d.xmat[b].reshape(3,3)
            if loads[i]>.002:local[i]+=.25*(R.T@(weighted[i]/loads[i]-d.xpos[b])-local[i])
            points[i]=d.xpos[b]+R@local[i]
        tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
        light_ready=light_ready+.002 if min(loads[[0,2]])>.015 else 0.
        if light_ready>.1:closing=True
        pinch_target+=(1-np.exp(-.002/.15))*((.35 if closing else .03)-pinch_target)
        ready=ready+.002 if min(loads[[0,2]])>.2 else 0.
        if phase=='pinch' and ready>.15:
            anchors=[d.xmat[palm].reshape(3,3).T@(p-d.xpos[palm])for p in points];route('lift')
        if phase=='lift':
            alpha=np.clip((d.time-phase_start)/1.5,0,1);alpha=alpha*alpha*(3-2*alpha);pgoal=p0+np.array([0,0,.04*alpha])
            if d.time-phase_start>1.7:route('pivot')
        if phase=='pivot' and tilt<30:route('cap_support')
        Jtool=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,Jtool,tool)
        angular=Jtool@d.qvel
        pivot_speed=np.linalg.norm(angular-axis*np.dot(axis,angular))
        pivot_load=.3+np.clip(.8*(pivot_speed-.4),0,.5)
        supported=(phase=='cap_support' and tilt<5 and loads[1]>.025 and min(loads[[0,2]])>.1 and np.linalg.norm(angular)<np.radians(5))
        support_hold=support_hold+.002 if supported else 0.
        lost_contact=lost_contact+.002 if phase!='pinch' and min(loads[[0,2]])<.01 else 0.
        if support_hold>=.2:outcome='pickup_cap_supported';break
        if lost_contact>.1:outcome='grasp_lost';break
        vref=np.zeros(26);normal_tasks=[];spin_tasks=[];Rpal=d.xmat[palm].reshape(3,3)
        Jp=np.zeros((3,m.nv));Jr=Jp.copy();mujoco.mj_jacBody(m,d,Jp,Jr,palm)
        desired=np.r_[np.clip(5*(pgoal-d.xpos[palm]),-.05,.05),np.clip(5*Rotation.from_matrix(Rp@Rpal.T).as_rotvec(),-.3,.3)]
        A=np.vstack([Jp[:,h.vi[20:]],Jr[:,h.vi[20:]]]);vref[20:]=np.clip(np.linalg.solve(A.T@A+1e-5*np.eye(6),A.T@desired),-.8,.8)
        for i,b in enumerate(bids):
            if i==1 and phase not in ['lift','pivot','cap_support']:continue
            point=points[i];radial=point-center;n=(cap_normal(point,cap_center,axis,cap_radius,cap_half_height) if loads[1]>.002 else axis) if i==1 else unit(radial-axis*np.dot(radial,axis))
            J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jac(m,d,J,Jr,point,b);cols=h.vi[4*i:4*i+4]
            if i==1:
                anticipate=tilt>15
                offset=float((cap_top-center)@axis)
                cap=center+np.array([0,0,offset]) if anticipate else cap_top
                error=cap-point;K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,center if anticipate else cap,tool);v=K@d.qvel+np.clip(6*error,-.04,.04)
                cap_lost_s=cap_lost_s+.002 if loads[1]<.005 else 0.
                if cap_lost_s>.05:cap_acquired=False
                if loads[1]>.01 and np.linalg.norm(error)<.01:cap_acquired=True
                if loads[1]>.005 or (not anticipate and (cap_acquired or np.linalg.norm(error)<.0005)):
                    axis_error=np.r_[center[:2]-cap[:2],0.];v=np.clip(3*axis_error-.5*(K@d.qvel)+4*error,-.04,.04);v-=n*np.dot(v,n);normal_tasks.append((J,n,.05,float(loads[i]),i))
            else:
                target=d.xpos[palm]+Rpal@anchors[i];K=np.zeros((3,m.nv));mujoco.mj_jac(m,d,K,None,target,palm)
                v=6*(target-point)+K@d.qvel;v-=n*np.dot(v,n)
                normal_tasks.append((J,n,pivot_load if phase in ['pivot','cap_support']else pinch_target,float(loads[i]),i))
            A=J[:,cols];rhs=v
            if i==0 and phase in ['pivot','cap_support']:
                rotation=np.cross(axis,[0,0,1]);goal=rotation*min(2.,.35/max(1e-9,np.linalg.norm(rotation)))
                spin=float(np.dot(n,goal));spin_tasks.append((i,Jr,n,spin,.2))
                A=np.vstack([A,.01*n@Jr[:,cols]]);rhs=np.r_[rhs,.01*spin]
            A=np.vstack([A,.002*np.eye(4)]);rhs=np.r_[rhs,.002*.3*(home[4*i:4*i+4]-d.qpos[h.qi[4*i:4*i+4]])]
            ids=[m.joint(f'joint_{i+1}_{j}').id for j in range(1,5)];lo=np.maximum(-1.,(m.jnt_range[ids,0]+1e-5-d.qpos[h.qi[4*i:4*i+4]])/.002);hi=np.minimum(1.,(m.jnt_range[ids,1]-1e-5-d.qpos[h.qi[4*i:4*i+4]])/.002)
            vref[4*i:4*i+4]=lsq_linear(A,rhs,bounds=(lo,hi)).x
        qref+=vref*.002;qref=np.clip(qref,d.qpos[h.qi]-.08,d.qpos[h.qi]+.08)
        for task in normal_tasks:
            J,n,_,_,finger=task;sl=slice(4*finger,4*finger+4);Ji=J[:,h.vi[sl]];error=qref[sl]-d.qpos[h.qi[sl]]
            qref[sl]-=Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),n)*np.dot(n,Ji@error)
        observed_time=d.time;observed_q=d.qpos.copy();observed_v=d.qvel.copy()
        h.motor_step(qref,vref,normal_tasks=normal_tasks,spin_tasks=spin_tasks)
        sample=[observed_time,0,0,0,*loads,0,0,0,0,0,0,tilt,*h.last_effort];samples.append(sample)
        if k%10==0:poses.append(observed_q);velocities.append(observed_v);times.append(observed_time);commands.append(dict(phase=phase,macro_phase=phase))
        if k%50==0:
            progress=dict(t=d.time,phase=phase,loads_N=loads.tolist(),tilt_deg=tilt,tool_center=d.xpos[tool].tolist(),unexpected_N=float(unexpected),unexpected_pair=unexpected_names,wall_s=time.monotonic()-wall);print('PROGRESS',json.dumps(progress),flush=True);(OUT/(name+'.progress.json')).write_text(json.dumps(progress))
        if unexpected>5:outcome='unexpected_collision';break
        if d.xpos[tool,2]<.32:outcome='tool_dropped';break
    report=dict(id=name,scope='physical pickup development; no screw-in credit',interface='hom',outcome=outcome,success=False,pickup_success=outcome=='pickup_cap_supported',support_hold_s=support_hold,source_sha256=sources,initialization=initial,trace=trace,commands=commands,parameters=dict(dt=.00025,thread_segments=48,pivot_normal_N=[.3,.8],pivot_brake_gain_N_s=.8,pivot_brake_threshold_rad_s=.4,cap_normal_N=.05,cap_contact_normal='finite-cylinder signed-distance gradient',cap_surface_tracking_gain_s=4,cap_reacquire_after_s=.05,cap_approach_phase='lift: preposition above predicted upright tool until tilt below 15 deg',cap_support_phase_tilt_deg=30,cap_target='actual cylinder top from model geometry',thumb_spin_gain_Nm_s=.2,middle_spin='released',sample_timing='state and observations before the 2 ms motor interval; effort applied during that interval'),final=dict(t=d.time,angle_deg=[0,0],depth_mm=0,tilt_deg=tilt))
    (OUT/(name+'.xml')).write_bytes(xml.read_bytes());(OUT/(name+'.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/(name+'.npz'),initial_qpos=q,qpos=poses,qvel=velocities,time=times,samples=samples)
    print('RESULT',outcome,report['final'],flush=True)


if __name__=='__main__':run()
