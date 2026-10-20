"""Exact CPU integration-state checkpoints at a physically reached handoff.

Continuation trials reuse the reached state and held motor effort. They are
not open-hand trials by themselves; a full trace must include their recorded
pickup/engagement prefix. Changing physics invalidates the shared prefix.
"""
import argparse,json,hashlib
import numpy as np,mujoco
from screwdriver_lab.physical_task import OUT,configure,CpuDevelopmentBackend

def save_checkpoint(s,backend,name,cap_offset):
    folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
    state=np.zeros(mujoco.mj_stateSize(s.m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(s.m,s.d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    motor={}
    for key,value in vars(backend.motor).items():
        if key=='s':continue
        if isinstance(value,np.ndarray):value=value.tolist()
        elif isinstance(value,list):value=[x.tolist()if isinstance(x,np.ndarray)else x for x in value]
        motor[key]=value
    metadata=dict(prefix=name,time=float(s.d.time),motor=motor,cap_offset=cap_offset,
        timestep=s.m.opt.timestep,last_yaw=s.last_yaw,tool_angle=s.tool_angle,
        fields={key:np.asarray(getattr(s,key)).tolist()for key in ['p0','R0','home','openq','local','homepatch','normals','pad_normals_local','arm_orientation_target']},
        xml_sha256=hashlib.sha256(s.xml.encode()).hexdigest())
    np.savez_compressed(folder/(name+'.npz'),integration_state=state,
        gain=s.m.actuator_gainprm,bias=s.m.actuator_biasprm,geom_friction=s.m.geom_friction)
    (folder/(name+'.json')).write_text(json.dumps(metadata));(folder/(name+'.xml')).write_text(s.xml)
    print('CHECKPOINT',name,float(s.d.time),flush=True)

def restore(name,interface='wrench'):
    from screwdriver_lab.controlled_slip import SlipSim
    from screwdriver_lab.inhand_policy import pickup_supports
    folder=OUT/'checkpoints';meta=json.loads((folder/(name+'.json')).read_text());a=np.load(folder/(name+'.npz'))
    s=SlipSim('screwdriver_lab/artifacts/controlled_slip_tip',.004,0,8,.003,0,pickup_position=[-.22,.25,.20482])
    pickup_supports(s);configure(s,True,.11,.6,'alternating',interface)
    if hashlib.sha256(s.xml.encode()).hexdigest()!=meta['xml_sha256']:
        raise RuntimeError('Checkpoint physics differs from current model; regenerate prefix')
    s.m.actuator_gainprm[:]=a['gain'];s.m.actuator_biasprm[:]=a['bias'];s.m.geom_friction[:]=a['geom_friction']
    s.m.opt.timestep=meta['timestep']
    mujoco.mj_setState(s.m,s.d,a['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    for key,value in meta['fields'].items():setattr(s,key,[np.asarray(v)for v in value]if key in ['local','homepatch','normals','pad_normals_local']else np.asarray(value))
    s.arm.last=s.d.qpos[s.arm.qi].copy();s.hydro=CpuDevelopmentBackend(s)
    for key,value in meta['motor'].items():
        old=getattr(s.hydro.motor,key,None)
        if isinstance(old,np.ndarray):value=np.asarray(value)
        elif key in ['transport_local','transport_anchors'] and value is not None:value=[np.asarray(v)for v in value]
        setattr(s.hydro.motor,key,value)
    gain=s.m.actuator_gainprm.copy();bias=s.m.actuator_biasprm.copy();ctrl=s.d.ctrl.copy()
    s.m.actuator_gainprm[:,0]=1;s.m.actuator_biasprm[:]=0;s.d.ctrl[:]=s.hydro.motor.torque
    mujoco.mj_forward(s.m,s.d)
    s.m.actuator_gainprm[:]=gain;s.m.actuator_biasprm[:]=bias;s.d.ctrl[:]=ctrl
    check=np.zeros_like(a['integration_state']);mujoco.mj_getState(s.m,s.d,check,mujoco.mjtState.mjSTATE_INTEGRATION)
    if not np.array_equal(check,a['integration_state']):raise RuntimeError('Integration-state restore mismatch')
    s.stage='turn';s.last_yaw=meta['last_yaw'];s.tool_angle=meta['tool_angle'];s.peak=0.;s.peakpen=0.;s.measure()
    return s,meta

def run(checkpoint,name,interface='wrench',duration=15,orientation=.015,cap_compensation=True,cap_orientation=0,drive_normal=.8,return_normal=.5,arm_gain=0,cycle="alternating",centering=0,cap_load=.6,tangent_k=150,tangent_d=2,side_comp=1,force_p=.5,force_i=.5,resolved_rate=False,cap_lateral=.3,return_ff=0,cap_stiffness=100,cap_damping=3,axis_recovery=False,effort_seating=False,seating_tangent=.08):
    from screwdriver_lab.alternating_contact import AlternatingContactController
    from screwdriver_lab.slip_metrics import SlipMetrics
    from pathlib import Path
    sources={};folder=OUT/'source_snapshots';folder.mkdir(exist_ok=True)
    for filename in ['drive_checkpoint.py','physical_task.py','alternating_contact.py','soft_finger.py','controlled_slip.py','pivot_spin_controller.py','slip_graph.py','resolved_rate.py']:
        data=(Path(__file__).parent/filename).read_bytes();digest=hashlib.sha256(data).hexdigest();sources[filename]=digest
        (folder/(digest+'_'+filename)).write_bytes(data)
    s,meta=restore(checkpoint,interface);s.cap_damping_compensation=cap_compensation;s.cap_orientation_gain=cap_orientation
    s.effort_seating=effort_seating;s.seating_tangent_N=seating_tangent;s.axis_recovery=axis_recovery;s.cap_axis_damping=cap_damping;s.cap_axis_stiffness=cap_stiffness;s.return_friction_ff=return_ff;s.cap_lateral_limit_N=cap_lateral;s.resolved_rate=resolved_rate;s.side_damping_compensation=side_comp;s.tangent_stiffness=tangent_k;s.tangent_damping=tangent_d;s.cap_drive_load_N=cap_load;s.centering_gain=centering;s.drive_normal_N=drive_normal;s.return_normal_N=return_normal
    from screwdriver_lab.pivot_spin_controller import CapturedSlipController
    factory=AlternatingContactController if cycle=="alternating"else CapturedSlipController
    c=factory(s,amplitude=12,speed=8,cap_offset=meta['cap_offset'],velocity_gain=100 if interface=='velocity'else 0,
        gain=.004,pad_weight=.012,drive_load=drive_normal,slide_load=return_normal,force_p=force_p,force_i=force_i,
        normal_d=2,orientation_gain=orientation,posture_gain=.01)
    c.force_targets=np.asarray(s.handleloads).copy()
    c.start=float(s.d.time);pos=s.d.xpos[s.m.body('palm').id].copy();rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).copy();depth=float(s.d.qpos[s.depthq])
    frames=[];poses=[];samples=[];nextframe=float(s.d.time);evidence=SlipMetrics();held=0.;outcome='timeout';arm_tilt=np.zeros(3)
    pivot=s.d.xpos[s.screw].copy()
    def capture(o):
        nonlocal nextframe,held
        evidence.add(o,s.modes)
        samples.append([o['t'],o['screw_angle_deg'],o['depth_mm'],*o['handle_forces_N'],*o['handle_slip_signed_mm_s'],o['index_cap_load_N'],o['tilt_deg'],o['penetration_mm'],int(o['engaged']),int(s.modes[0]=='slide_return'),int(s.modes[1]=='slide_return'),o['seat_force_N'],o['screw_speed_rad_s']])
        if o['t']>=nextframe-1e-9:
            frames.append(dict(o,phase='turn'if c.phase!='seat'else'seat',contact_phase=c.phase,
                pad_points=s.pad_points.tolist(),tool_axis=s.d.xmat[s.tool].reshape(3,3)[:,2].tolist(),palm_rotation=s.d.xmat[s.m.body('palm').id].reshape(3,3).tolist()))
            poses.append(s.d.qpos.copy());nextframe+=1/30
        ok=o['depth_mm']>=1.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and o['seat_force_N']>.1
        held=held+.001 if ok else 0.
    for k in range(round(duration/.02)):
        o=s.observe()
        if o['depth_mm']>=1.98 and o['seat_force_N']>.1 and c.phase!='seat':c.transition('seat',o)
        q,cmd=c.tick(o)
        from scipy.spatial.transform import Rotation
        axis=s.d.xmat[s.tool].reshape(3,3)[:,2]
        arm_tilt+=np.clip(arm_gain*.02*np.cross(axis,[0,0,1]),-.005,.005);arm_tilt[2]=0
        arm_tilt*=min(1.,np.radians(12)/max(1e-9,np.linalg.norm(arm_tilt)))
        correction=Rotation.from_rotvec(arm_tilt).as_matrix()
        target=pivot+correction@(pos-pivot)-[0,0,s.d.qpos[s.depthq]-depth];delta=target-s.p0
        s.arm_orientation_target=correction@rotation;s.stage='seat'if c.phase=='seat'else'turn'
        events=c.step(q,np.r_[-delta[2],delta[:2],0.],capture)
        if k%50==0:print(round(o['t'],2),c.phase,np.round(s.handleloads,3),'cap',round(s.index_cap_load,3),'angle',round(s.observe()['screw_angle_deg'],2),'tilt',round(s.observe()['tilt_deg'],2),flush=True)
        if k%50==0:(OUT/(name+'.progress.json')).write_text(json.dumps(dict(t=o['t'],phase=c.phase,cycles=c.cycles,metrics=evidence.report(),observation=s.observe())))
        if events:outcome=events[0];break
        if held>=.4:outcome='seated';break
        if c.phase in ['acquire','unload','axis_recover'] and s.d.time-c.start>10:outcome='load_regulation_failed';break
    result=dict(id=name,scope='continuation from physically reached state',prefix=checkpoint,
        interface=interface,outcome=outcome,success=outcome=='seated',frames=frames,final=s.observe(),
        policy_graph=c.graph.schema,controller_debug=getattr(c,'debug_forces',[]),cycles=c.cycles,metrics=evidence.report(),motor=s.hydro.motor.report(),cycle_trace=c.trace,parameters=dict(orientation=orientation,cap_compensation=cap_compensation,cap_orientation=cap_orientation,drive_normal=drive_normal,return_normal=return_normal,arm_gain=arm_gain,cycle=cycle,centering=centering,cap_load=cap_load,tangent_k=tangent_k,tangent_d=tangent_d,side_comp=side_comp,force_p=force_p,force_i=force_i,resolved_rate=resolved_rate,cap_lateral=cap_lateral,return_ff=return_ff,cap_stiffness=cap_stiffness,cap_damping=cap_damping,axis_recovery=axis_recovery,effort_seating=effort_seating,seating_tangent=seating_tangent),
        source_sha256=sources,checkpoint_sha256=hashlib.sha256((OUT/'checkpoints'/(checkpoint+'.npz')).read_bytes()).hexdigest())
    (OUT/(name+'.json')).write_text(json.dumps(result));np.savez_compressed(OUT/(name+'.npz'),qpos=poses,samples_1khz=np.asarray(samples),sample_columns=np.array(['t','screw_angle_deg','depth_mm','thumb_N','middle_N','thumb_slip_mm_s','middle_slip_mm_s','cap_N','tilt_deg','penetration_mm','engaged','thumb_return','middle_return','seat_N','screw_speed_rad_s']));(OUT/(name+'.xml')).write_text(s.xml)
    print('RESULT',outcome,result['final'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--name',required=True)
    p.add_argument('--interface',default='wrench',choices=['pose','wrench','velocity']);p.add_argument('--duration',type=float,default=15)
    p.add_argument('--orientation',type=float,default=.015);p.add_argument('--cap-orientation',type=float,default=0);p.add_argument('--no-cap-compensation',action='store_true');p.add_argument('--drive-normal',type=float,default=.8);p.add_argument('--return-normal',type=float,default=.5);p.add_argument('--arm-gain',type=float,default=0);p.add_argument('--cycle',choices=['paired','alternating'],default='alternating');p.add_argument('--centering',type=float,default=0);p.add_argument('--cap-load',type=float,default=.6);p.add_argument('--tangent-k',type=float,default=150);p.add_argument('--tangent-d',type=float,default=2);p.add_argument('--side-comp',type=float,default=1);p.add_argument('--force-p',type=float,default=.5);p.add_argument('--force-i',type=float,default=.5);p.add_argument('--resolved-rate',action='store_true');p.add_argument('--cap-lateral',type=float,default=.3);p.add_argument('--return-ff',type=float,default=0);p.add_argument('--cap-stiffness',type=float,default=100);p.add_argument('--cap-damping',type=float,default=3);p.add_argument('--axis-recovery',action='store_true');p.add_argument('--effort-seating',action='store_true');p.add_argument('--seating-tangent',type=float,default=.08);a=p.parse_args()
    run(a.checkpoint,a.name,a.interface,a.duration,a.orientation,not a.no_cap_compensation,a.cap_orientation,a.drive_normal,a.return_normal,a.arm_gain,a.cycle,a.centering,a.cap_load,a.tangent_k,a.tangent_d,a.side_comp,a.force_p,a.force_i,a.resolved_rate,a.cap_lateral,a.return_ff,a.cap_stiffness,a.cap_damping,a.axis_recovery,a.effort_seating,a.seating_tangent)
