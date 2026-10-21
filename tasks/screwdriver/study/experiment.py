"""Matched manipulation experiments with recorded independent dynamics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from dexagent.sim.model import prepare,OUT
from dexagent.sim.initialization import fit_grasp
from dexagent.sim.drake_backend import HydroPlant
from dexagent.sim.controllers import ManipulationPolicy
from dexagent.sim.seating import geometry as seating_geometry,contact_reached,flush

def run(name='development',interface='hom',duration=10,slim=True,dt=.001,resolution=.002,modulus=1.7e6,speed=45,drive_load=.4,return_load=.025,cycle='paired',hybrid=False,gap=.00015,tilt=None,thread_segments=24,explicit_steel=False,beta=1.,load_ramp_s=.15,damping_compensation=0.,release_reset_spin=False,grip_friction=.6,hold_cap_load=1.2):
    if any((OUT/(name+suffix)).exists()for suffix in ['.json','.npz']):raise FileExistsError(f'Refusing to overwrite completed trial {name}; choose a new name')
    initial_tilt=tilt;sources={};source_dir=OUT/'source_snapshots';source_dir.mkdir(exist_ok=True,parents=True)
    for key,module in list(sys.modules.items()):
        if not key.startswith(('dexagent.','tasks.'))or not hasattr(module,'__source_bytes__'):continue
        content=module.__source_bytes__;digest=hashlib.sha256(content).hexdigest();filename=key.rsplit('.',1)[-1]+'.py'
        (source_dir/(digest+'_'+filename)).write_bytes(content);sources[filename]=digest
    content=Path(__file__).read_bytes();digest=hashlib.sha256(content).hexdigest();sources['experiment.py']=digest
    (source_dir/(digest+'_experiment.py')).write_bytes(content)
    xml=prepare(slim,thread_segments);q,initial=fit_grasp(xml,gap=gap,tilt_deg=tilt)
    frozen_xml=OUT/(name+'.xml');frozen_xml.write_bytes(xml.read_bytes());xml=frozen_xml
    h=HydroPlant(xml,dt=dt,resolution=resolution,modulus=modulus,explicit_steel=explicit_steel,beta=beta,grip_friction=grip_friction);h.initialize_from_mujoco(q);audit=h.audit();h.sync()
    initial_seating=seating_geometry(h.m,h.d)
    c=ManipulationPolicy(h,initial,interface,speed=speed,drive_load=drive_load,return_load=return_load,cycle=cycle,hybrid=hybrid,load_ramp_s=load_ramp_s,release_reset_spin=release_reset_spin,hold_cap_load=hold_cap_load)
    p=h.plant;poses=[];velocities=[];samples=[];commands=[];times=[];references=[];velocity_refs=[];previous=None;angle=np.zeros(2);startz=h.d.xpos[c.screw,2];outcome='timeout';wall=time.monotonic();seated_time=0.;manipulation_success=False
    for k in range(round(duration/.002)):
        c.measure()
        if samples and c.phase!='hold' and contact_reached(seating_geometry(h.m,h.d),c.seat_load):c.hold_at_contact()
        qref,vref=c.tick();h.motor_step(qref,vref,normal_tasks=c.normal_tasks if hybrid else None,damping_compensation=damping_compensation)
        c.measure(raw_only=True)
        yaws=np.array([np.arctan2(h.d.xmat[b].reshape(3,3)[1,0],h.d.xmat[b].reshape(3,3)[0,0])for b in [c.tool,c.screw]])
        if previous is not None:angle-=np.arctan2(np.sin(yaws-previous),np.cos(yaws-previous))
        previous=yaws;axis=h.d.xmat[c.tool].reshape(3,3)[:,2];tilt=np.degrees(np.arccos(np.clip(axis[2],-1,1)))
        sample=[h.d.time,*np.degrees(angle),(startz-h.d.xpos[c.screw,2])*1000,*c.loads,*c.slips,*c.areas,tilt,*h.last_effort,c.seat_load,c.unexpected_load]
        samples.append(sample)
        references.append(qref.copy());velocity_refs.append(vref.copy())
        if k%10==0:poses.append(h.d.qpos.copy());velocities.append(h.d.qvel.copy());times.append(h.d.time);commands.append(c.last_command)
        if k%50==0:
            progress=dict(t=h.d.time,phase=c.phase,stroke_deg=float(np.degrees(c.stroke)),strokes_deg=np.degrees(c.strokes).tolist(),loads_N=c.loads.tolist(),angle_deg=np.degrees(angle).tolist(),depth_mm=sample[3],tilt_deg=float(tilt),motor_torques_Nm=h.last_effort[:12].tolist(),seat_load_N=c.seat_load,unexpected_load_N=c.unexpected_load,wall_s=time.monotonic()-wall)
            print('PROGRESS',json.dumps(progress),flush=True);(OUT/(name+'.progress.json')).write_text(json.dumps(progress))
        if c.unexpected_load>5:outcome='unexpected_collision';break
        spin_speed=abs(samples[-1][2]-samples[-2][2])/.002 if len(samples)>1 else 0.
        seated=(c.phase=='hold' and flush(seating_geometry(h.m,h.d)) and c.seat_load>.2 and tilt<3 and spin_speed<5 and min(c.loads[[0,2]])>.05 and c.loads[1]>.1)
        seated_time=seated_time+.002 if seated else 0.
        if seated_time>=.5:outcome='seated_prepared_pose';manipulation_success=True;break
        if tilt>12:outcome='axis_lost';break
        if max(c.loads)>8:outcome='contact_overload';break
    report=dict(id=name,backend='Drake 1.58 / SAP hydroelastic',scope='prepared-pose manipulation test; no pickup credit',interface=interface,outcome=outcome,success=False,manipulation_success=manipulation_success,seating_hold_s=seated_time,initial_seating_geometry=initial_seating,final_seating_geometry=seating_geometry(h.m,h.d),parameters=dict(dt=dt,resolution=resolution,modulus=modulus,speed=speed,drive_load=drive_load,return_load=return_load,cycle=cycle,hybrid=hybrid,gap_m=gap,initial_tilt_deg=initial_tilt,slim=slim,thread_segments=thread_segments,explicit_steel=explicit_steel,beta=beta,load_ramp_s=load_ramp_s,damping_compensation=damping_compensation,release_reset_spin=release_reset_spin,grip_friction=grip_friction,hold_load_N=c.hold_load,hold_cap_load_N=c.hold_cap_load),source_sha256=sources,conversion_audit=audit,initialization=initial,graph=c.graph,trace=c.trace,final=dict(t=h.d.time,angle_deg=np.degrees(angle).tolist(),depth_mm=samples[-1][3],tilt_deg=samples[-1][13]),elapsed_wall_s=time.monotonic()-wall,commands=commands,xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),hardware_calibrated=False)
    np.savez_compressed(OUT/(name+'.npz'),initial_qpos=q,qpos=poses,qvel=velocities,time=times,samples=samples,qref_500hz=references,vref_500hz=velocity_refs,columns=np.array(['t','handle_deg','screw_deg','depth_mm','thumb_N','cap_N','middle_N','thumb_slip_mm_s','cap_slip_mm_s','middle_slip_mm_s','thumb_area_m2','cap_area_m2','middle_area_m2','tilt_deg']+[f'motor_{i}_Nm'for i in range(26)]+['seat_load_N','unexpected_load_N']))
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2));(OUT/(name+'.xml')).write_bytes(xml.read_bytes());print('RESULT',json.dumps({k:v for k,v in report.items()if k not in ['commands','graph','initialization','conversion_audit']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='development');p.add_argument('--interface',default='hom');p.add_argument('--duration',type=float,default=10);p.add_argument('--large',action='store_true');p.add_argument('--dt',type=float,default=.001);p.add_argument('--resolution',type=float,default=.002);p.add_argument('--modulus',type=float,default=1.7e6);p.add_argument('--speed',type=float,default=45);p.add_argument('--drive-load',type=float,default=.4);p.add_argument('--return-load',type=float,default=.025);p.add_argument('--cycle',choices=['paired','alternating'],default='paired');p.add_argument('--hybrid',action='store_true');p.add_argument('--gap',type=float,default=.00015);p.add_argument('--tilt',type=float);p.add_argument('--thread-segments',type=int,default=24);p.add_argument('--explicit-steel',action='store_true');p.add_argument('--beta',type=float,default=1.);p.add_argument('--load-ramp',type=float,default=.15);p.add_argument('--damping-compensation',type=float,default=0.);p.add_argument('--release-reset-spin',action='store_true');p.add_argument('--grip-friction',type=float,default=.6);p.add_argument('--hold-cap-load',type=float,default=1.2);a=p.parse_args();run(a.name,a.interface,a.duration,not a.large,a.dt,a.resolution,a.modulus,a.speed,a.drive_load,a.return_load,a.cycle,a.hybrid,a.gap,a.tilt,a.thread_segments,a.explicit_steel,a.beta,a.load_ramp,a.damping_compensation,a.release_reset_spin,a.grip_friction,a.hold_cap_load)

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
