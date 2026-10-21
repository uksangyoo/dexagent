"""Branch from an exact recorded physical state to test the corrected stop.

The original failed long run is immutable. This is explicitly a checkpoint
continuation, not a claimed fresh uninterrupted rollout. No object pose or
velocity is changed at the branch, and all subsequent control is motor torque.
"""
import json,hashlib,sys,time
from pathlib import Path
import mujoco,numpy as np
from dexagent.sim.model import OUT
from dexagent.sim.drake_backend import HydroPlant
from dexagent.sim.controllers import ManipulationPolicy
from dexagent.sim.seating import geometry,contact_reached,flush


def run(source='hom-seating23',name='hom-seat-stop42',duration=3.,hold_cap_load=1.2):
    if (OUT/(name+'.json')).exists():raise FileExistsError(name)
    raw=OUT/(source+'.npz');a=dict(np.load(raw));r=json.loads((OUT/(source+'.json')).read_text());xml=OUT/(source+'.xml')
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
    chosen=None
    for k,q in enumerate(a['qpos']):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);j=int(np.argmin(abs(a['samples'][:,0]-a['time'][k])))
        if contact_reached(geometry(m,d),a['samples'][j,-2]):chosen=(k,j);break
    if chosen is None:raise RuntimeError('No recorded state satisfies the new contact/clearance guard')
    frame,index=chosen;t0=float(a['time'][frame]);q0=a['qpos'][frame].copy();v0=a['qvel'][frame].copy()
    frozen=OUT/(name+'.xml');frozen.write_bytes(xml.read_bytes())
    h=HydroPlant(frozen,dt=.00025,resolution=.002,explicit_steel=True,beta=0)
    h.initialize_from_mujoco(q0,v0);h.sync()
    continuity=dict(position_max_error=float(max(abs(h.d.qpos-q0))),velocity_max_error=float(max(abs(h.d.qvel-v0))))
    assert max(continuity.values())<1e-10,continuity
    h.context.SetTime(t0);h.sync()
    initial=r['initialization'];c=ManipulationPolicy(h,initial,'hom',speed=60,drive_load=.4,return_load=.06,cycle='alternating',hybrid=True,hold_cap_load=hold_cap_load)
    filtered=np.zeros(3)
    for sample in a['samples'][:index+1]:filtered+=.15*(sample[4:7]-filtered)
    c.filtered=filtered;c.loads=a['samples'][index,4:7].copy();c.phase=r['commands'][frame]['phase']
    c.points=[h.d.xpos[b]+h.d.xmat[b].reshape(3,3)@local for b,local in zip(c.bids,c.local)]
    c.hold_at_contact()
    snapshot=OUT/'source_snapshots';sources={}
    for key,module in list(sys.modules.items()):
        if key.startswith(('dexagent.','tasks.'))and hasattr(module,'__source_bytes__'):
            content=module.__source_bytes__;digest=hashlib.sha256(content).hexdigest();filename=key.rsplit('.',1)[-1]+'.py';sources[filename]=digest;(snapshot/(digest+'_'+filename)).write_bytes(content)
    content=Path(__file__).read_bytes();digest=hashlib.sha256(content).hexdigest();sources['seat_continuation.py']=digest;(snapshot/(digest+'_seat_continuation.py')).write_bytes(content)
    poses=list(a['qpos'][:frame+1]);velocities=list(a['qvel'][:frame+1]);times=list(a['time'][:frame+1]);samples=list(a['samples'][:index+1]);commands=list(r['commands'][:frame+1]);references=list(a['qref_500hz'][:index+1]);velocity_refs=list(a['vref_500hz'][:index+1])
    angle=np.radians(a['samples'][index,1:3]);previous=np.array([np.arctan2(h.d.xmat[b].reshape(3,3)[1,0],h.d.xmat[b].reshape(3,3)[0,0])for b in [c.tool,c.screw]])
    startz=h.d.xpos[c.screw,2]+samples[-1][3]/1000
    hold=0.;outcome='continuation_timeout';wall=time.monotonic();trace=[]
    for k in range(round(duration/.002)):
        if k:c.measure()
        qref,vref=c.tick();h.motor_step(qref,vref,normal_tasks=c.normal_tasks);c.measure(raw_only=True)
        yaws=np.array([np.arctan2(h.d.xmat[b].reshape(3,3)[1,0],h.d.xmat[b].reshape(3,3)[0,0])for b in [c.tool,c.screw]])
        angle-=np.arctan2(np.sin(yaws-previous),np.cos(yaws-previous));previous=yaws
        tilt=float(np.degrees(np.arccos(np.clip(h.d.xmat[c.tool].reshape(3,3)[2,2],-1,1))))
        sample=[h.d.time,*np.degrees(angle),(startz-h.d.xpos[c.screw,2])*1000,*c.loads,*c.slips,*c.areas,tilt,*h.last_effort,c.seat_load,c.unexpected_load]
        speed=abs(sample[2]-samples[-1][2])/.002;samples.append(sample);references.append(qref.copy());velocity_refs.append(vref.copy())
        if (k+1)%10==0:poses.append(h.d.qpos.copy());velocities.append(h.d.qvel.copy());times.append(h.d.time);commands.append(c.last_command)
        g=geometry(h.m,h.d);stable=flush(g)and c.seat_load>.2 and tilt<3 and speed<5 and min(c.loads[[0,2]])>.05 and c.loads[1]>.1
        hold=hold+.002 if stable else 0.
        if k%50==0:
            progress=dict(t=h.d.time,hold_s=hold,loads_N=c.loads.tolist(),seat_load_N=c.seat_load,depth_mm=sample[3],screw_deg=sample[2],tilt_deg=tilt,geometry=g,wall_s=time.monotonic()-wall);trace.append(progress);print('PROGRESS',json.dumps(progress),flush=True)
        if hold>=.5:outcome='seated_checkpoint_continuation';break
        if tilt>12 or c.unexpected_load>5 or max(c.loads)>8:outcome='continuation_contact_failure';break
    report=dict(id=name,interface='hom',backend='Drake 1.58 / SAP hydroelastic',scope='prepared-grasp drive prefix plus explicitly checkpoint-resumed seat hold; no pickup credit',outcome=outcome,success=False,manipulation_success=hold>=.5,seating_hold_s=hold,checkpoint=dict(source=source,source_npz_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),source_frame=frame,source_sample=index,time_s=t0,continuity=continuity,selection='First saved 50 Hz state satisfying measured seat force and head-clearance guard; no state adjustment'),source_sha256=sources,prefix_source_sha256=r['source_sha256'],xml_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),parameters={**r['parameters'],'hold_load_N':c.hold_load,'hold_cap_load_N':c.hold_cap_load},initialization=initial,graph=c.graph,commands=commands,trace=trace,final=dict(t=h.d.time,angle_deg=np.degrees(angle).tolist(),depth_mm=samples[-1][3],tilt_deg=tilt),final_seating_geometry=geometry(h.m,h.d),elapsed_wall_s=time.monotonic()-wall,hardware_calibrated=False)
    np.savez_compressed(OUT/(name+'.npz'),qpos=poses,qvel=velocities,time=times,samples=samples,qref_500hz=references,vref_500hz=velocity_refs,columns=a['columns'])
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2));print('RESULT',json.dumps({k:report[k]for k in ['outcome','checkpoint','final','seating_hold_s']}),flush=True)


if __name__=='__main__':run()
