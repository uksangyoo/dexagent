"""Execute generated code/graph against real simulated dynamics, GPU 1 policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse, importlib.util, json, time, hashlib
from pathlib import Path
import numpy as np
from screwdriver_lab.hom.model import prepare, OUT
from screwdriver_lab.hom.initialization import fit_grasp
from screwdriver_lab.hom.drake_backend import HydroPlant
from screwdriver_lab.hom.seating import geometry, contact_reached
from .contracts import validate
from .workflow import digest
from .runtime import GraphPolicy
from .hydro_adapter import HydroHandAdapter


def accepted(observation,goal):
    return (abs(observation['screw_deg']-goal['target'])<=goal['tolerance']
            and observation['settled_s']+1e-9>=goal['hold_s'])


def run(bundle,name,duration=4.):
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    for filename,key in [('graph.json','graph_sha256'),('policy.py','code_sha256')]:
        if digest(bundle/filename)!=manifest[key]:raise ValueError('Policy artifact hash mismatch')
    graph=validate(json.loads((bundle/'graph.json').read_text()));goal=graph['goal']
    if (OUT/(name+'.json')).exists()or(OUT/(name+'.npz')).exists():raise FileExistsError(name)
    sources={}
    for directory in [Path(__file__).parent,Path(__file__).parent.parent/'hom']:
        for file in directory.glob('*.py'):
            sha=digest(file);(OUT/'source_snapshots'/(sha+'_'+file.name)).write_bytes(file.read_bytes());sources[str(file.relative_to(Path(__file__).parents[2]))]=sha
    spec=importlib.util.spec_from_file_location('generated_hand_policy',bundle/'policy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    policy=GraphPolicy(graph,module.NODES)
    xml=prepare(True,48);q,initial=fit_grasp(xml,gap=.00015,tilt_deg=0)
    frozen=OUT/(name+'.xml');frozen.write_bytes(xml.read_bytes())
    h=HydroPlant(frozen,dt=.00025,resolution=.002,explicit_steel=True,beta=0);h.initialize_from_mujoco(q);h.sync()
    hand=HydroHandAdapter(h,initial,graph['abstraction']);c=hand.controller
    start_z=h.d.xpos[c.screw,2];angle=np.zeros(2);previous=None;settled=0.;lost=0.;ever_ready=False
    samples=[];poses=[];velocities=[];times=[];commands=[];refs=[];vrefs=[];agent_ticks=[];outcome='time_budget';wall=time.monotonic()
    last_action=None
    for k in range(round(duration/.002)):
        c.measure();tilt=float(np.degrees(np.arccos(np.clip(h.d.xmat[c.tool].reshape(3,3)[2,2],-1,1))))
        ready=bool(min(c.filtered[[0,2]])>.28 and c.filtered[1]>.3);ever_ready |= ready
        lost=lost+.002 if ever_ready and min(c.loads[[0,2]])<.01 else 0.
        obs=dict(elapsed_s=float(h.d.time),contact_ready=ready,screw_deg=float(np.degrees(angle[1])),depth_mm=float((start_z-h.d.xpos[c.screw,2])*1000),tool_tilt_deg=tilt,contact_lost=lost>.1,unexpected_force_N=c.unexpected_load,seat_contact=contact_reached(geometry(h.m,h.d),c.seat_load),settled_s=settled)
        if tilt>12 or c.unexpected_load>5 or max(c.loads)>8:outcome='physical_guard_failure';break
        if k%25==0:
            decision=policy.step(obs);agent_ticks.append(dict(observation=obs,decision=decision))
            if 'terminal' in decision:
                outcome='subgoal_succeeded'if decision['terminal']=='succeeded'and accepted(obs,goal)else'premature_success_rejected'if decision['terminal']=='succeeded'else'policy_failed';break
            last_action=decision['action'];hand.dispatch(last_action,decision.get('arguments',{}))
        qref,vref,effort=hand.step();c.measure(raw_only=True)
        tilt=float(np.degrees(np.arccos(np.clip(h.d.xmat[c.tool].reshape(3,3)[2,2],-1,1))))
        yaws=np.array([np.arctan2(h.d.xmat[b].reshape(3,3)[1,0],h.d.xmat[b].reshape(3,3)[0,0])for b in [c.tool,c.screw]])
        if previous is not None:angle-=np.arctan2(np.sin(yaws-previous),np.cos(yaws-previous))
        previous=yaws;angle_deg=np.degrees(angle);speed=abs(angle_deg[1]-samples[-1][2])/.002 if samples else 0.
        good=(last_action=='hold'and abs(angle_deg[1]-goal['target'])<=goal['tolerance']and speed<5 and min(c.loads[[0,2]])>.05 and c.loads[1]>.1 and tilt<3)
        settled=settled+.002 if good else 0.
        sample=[h.d.time,*angle_deg,(start_z-h.d.xpos[c.screw,2])*1000,*c.loads,*c.slips,*c.areas,tilt,*effort,c.seat_load,c.unexpected_load];samples.append(sample);refs.append(qref);vrefs.append(vref)
        if k%10==0:poses.append(h.d.qpos.copy());velocities.append(h.d.qvel.copy());times.append(h.d.time);commands.append(c.last_command)
        if k%250==0:print('PROGRESS',json.dumps(dict(t=h.d.time,node=policy.node,operation=last_action,screw_deg=sample[2],loads=c.loads.tolist(),settled_s=settled)),flush=True)
    parameters=dict(dt=.00025,resolution=.002,modulus=1.7e6,grip_friction=.6,thread_segments=48,speed=60,drive_load=.4,return_load=.06,cycle='alternating',hybrid=True)
    report=dict(id=name,interface=graph['abstraction'],scope='agent-authored prepared-grasp rotation subgoal; no pickup or seating credit',backend='Drake SAP hydroelastic',success=outcome=='subgoal_succeeded',full_task_success=False,outcome=outcome,goal=goal,policy_sha256=manifest['graph_sha256'],policy_provenance=manifest['provenance'],agent_graph=graph,agent_transitions=policy.trace,agent_ticks=agent_ticks,abstraction_calls=hand.trace,actuator_layout=hand.layout,initialization=initial,commands=commands,parameters=parameters,source_sha256=sources,xml_sha256=digest(frozen),final=dict(t=h.d.time,screw_deg=float(np.degrees(angle[1])),depth_mm=float((start_z-h.d.xpos[c.screw,2])*1000),settled_s=settled),elapsed_wall_s=time.monotonic()-wall,device_policy=dict(physical_gpu=1,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],physics_device='CPU',gpu_work_in_this_rollout=False),rates_hz=dict(policy_graph=20,motor_servo=500,physics=4000),runtime_llm=False)
    columns=['t','handle_deg','screw_deg','depth_mm','thumb_N','cap_N','middle_N','thumb_slip_mm_s','cap_slip_mm_s','middle_slip_mm_s','thumb_area_m2','cap_area_m2','middle_area_m2','tilt_deg']+[f'motor_{i}_Nm'for i in range(26)]+['seat_load_N','unexpected_load_N']
    np.savez_compressed(OUT/(name+'.npz'),initial_qpos=q,qpos=poses,qvel=velocities,time=times,samples=samples,qref_500hz=refs,vref_500hz=vrefs,columns=columns)
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2));print('RESULT',outcome,report['final'],flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--name',required=True);p.add_argument('--duration',type=float,default=4);a=p.parse_args();run(a.bundle,a.name,a.duration)
