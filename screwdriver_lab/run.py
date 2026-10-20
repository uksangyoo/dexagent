"""Run paired real-physics diagnostics, preserving failed and invalid episodes."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from screwdriver_lab.scene import ROOT
from screwdriver_lab.simulator import Simulator
from screwdriver_lab.controllers import Controller, METHODS, training_bank
from screwdriver_lab.policy import Policy,Decision

CASES=[dict(id='nominal',name='Nominal',friction=.8,radius_mm=18,offset_mm=0,yaw_deg=0),
 dict(id='slippery',name='Low friction',friction=.35,radius_mm=18,offset_mm=0,yaw_deg=0),
 dict(id='offset',name='Lateral offset',friction=.8,radius_mm=18,offset_mm=3,yaw_deg=0),
 dict(id='phase',name='Rotated start',friction=.8,radius_mm=18,offset_mm=0,yaw_deg=30),
 dict(id='wide',name='Wider handle',friction=.8,radius_mm=20,offset_mm=0,yaw_deg=0)]
LIMITS=dict(goal_deg=90.,angle_tolerance_deg=3.,speed_rad_s=.2,hold_s=.4,drift_mm=20.,tilt_deg=20.,force_N=8.,penetration_mm=.5,timeout_s=12.,acquire_s=3.,acquisition_drift_mm=5.,minimum_shaft_clearance_mm=3.)

def acquisition_ok(o):
    return bool(sum(f>.05 for f in o['support_forces'])>=2 and o['speed_rad_s']<.2 and o['tilt_deg']<10 and o['drift_mm']<5 and max(o['forces'])<8 and o['depth_mm']<.5 and o['external_N']<.05 and o['forbidden_contact_N']<.05 and o['shaft_clearance_mm']>=3)

def success_sample(o,goal=90.):
    return bool(abs(goal-o['angle_deg'])<=3 and o['speed_rad_s']<.2 and o['drift_mm']<=20 and o['tilt_deg']<=20 and sum(f>.05 for f in o['support_forces'])>=2 and max(o['forces'])<=8 and o['depth_mm']<=.5 and o['external_N']<.05 and o['forbidden_contact_N']<.05 and o['shaft_clearance_mm']>=3)

def run_trial(case,method,mode,bank,policy_cls=Policy,goal=90.):
    s=Simulator(case,fixture=mode=='fixture');qs=[];frames=[];trace=[];wall=time.monotonic()
    phase='acquire';next_frame=0.;hold=0.;task_started=False;acquire_time=0.
    # Record reset and acquisition. No hidden settling prefix.
    def capture(sim,force=False):
        nonlocal next_frame
        if force or sim.data.time+1e-9>=next_frame:
            o=sim.observe();o['phase']=phase;o['task_t']=max(0.,o['t']-acquire_time) if task_started else 0.
            if not frames or o['t']>frames[-1]['t']+1e-9:frames.append(o);qs.append(sim.data.qpos.copy())
            next_frame=sim.data.time+1/30 if force else next_frame+1/30
    capture(s,True);s.listeners.append(capture)
    trace.append(dict(t=0.,phase='acquire',target_deg=0.,finger=-1,reason='Preload fingertip contacts; continuously enforce tool clearance'))
    acquisition_events=set();acquisition_max_drift=0.;acquisition_min_shaft=s.shaft_clearance_m
    if s.depth>.0002 or s.shaft_clearance_m<.003:acquisition_events.add('initial_geometry')
    s.target=s.ik(s.material_targets(0,compression=.002),gain=1.)
    for k in range(150):
        if acquisition_events:break
        acquisition_events.update(s.step(s.ik(s.material_targets(0,compression=.002),gain=.35)))
        o=s.observe();acquisition_max_drift=max(acquisition_max_drift,o['drift_mm']);acquisition_min_shaft=min(acquisition_min_shaft,s.shaft_clearance_m)
        if o['drift_mm']>5:acquisition_events.add('acquisition_slip')
        if o['tilt_deg']>10:acquisition_events.add('acquisition_tilt')
        if o['external_N']>.05:acquisition_events.add('external_support')
    acquire=s.observe();valid=not acquisition_events and acquisition_ok(acquire)
    initial_hash=hashlib.sha256(s.data.qpos.tobytes()+s.data.qvel.tobytes()+s.target.tobytes()).hexdigest()
    acquisition_metrics=dict(max_drift_mm=acquisition_max_drift,min_shaft_clearance_mm=acquisition_min_shaft*1000,max_force_N=s.max_force,max_penetration_mm=s.max_depth*1000)
    # Reference rebasing changes the measurement only, never the physical pose.
    if valid:
        s.start_pos=s.data.xpos[s.oid].copy();s.start_rot=s.data.xmat[s.oid].reshape(3,3).copy();s.patches=s.capture_patches()
    acquire_time=float(s.data.time);task_started=True;s.max_force=0.;s.max_depth=0.;s.max_selfdepth=0.
    policy=policy_cls(method,goal);controller=Controller(s,method,bank)
    outcome='timeout' if valid else 'invalid_initialization';decision=Decision('acquire',0,'initialization rejected')
    first_failure=dict(t=acquire_time,events=sorted(acquisition_events or {'acquisition_not_settled'})) if not valid else None
    task_samples=[]
    def check_hold(sim):
        nonlocal hold
        if task_started:
            o=sim.observe();hold=hold+sim.model.opt.timestep if success_sample(o,goal) else 0.
    s.listeners.append(check_hold)
    for k in range(600 if valid else 0):
        elapsed=float(s.data.time)-acquire_time
        if k%10==0:
            decision=policy.decide(s.observe(),elapsed)
            if not isinstance(decision,Decision) or decision.phase not in ('rotate','hold','regrasp','repatch','abort') or not np.isfinite(decision.target_deg) or abs(decision.target_deg)>180 or decision.finger not in (-1,0,1,2,3,4):raise ValueError('Invalid policy decision')
            trace.append(dict(t=round(float(s.data.time),4),**asdict(decision)))
        phase=decision.phase
        if phase=='abort':outcome='policy_abort';break
        events=s.step(controller.command(decision,elapsed));o=s.observe()
        if o['drift_mm']>20:events.add('translation_limit')
        if o['tilt_deg']>20:events.add('tilt_limit')
        if o['external_N']>.05:events.add('external_support')
        if events:outcome=sorted(events)[0];first_failure=dict(t=o['t'],events=sorted(events));break
        if hold>=.4:outcome='success';break
    capture(s,True)
    final=s.observe();final['phase']=phase;final['task_t']=float(s.data.time)-acquire_time
    record=dict(id=f'{mode}-{case["id"]}-{method}',method=method,mode=mode,case=case['id'],outcome=outcome,success=outcome=='success',valid_initialization=valid,goal_deg=goal,acquisition=acquire,acquisition_duration_s=acquire_time,acquisition_metrics=acquisition_metrics,acquisition_events=sorted(acquisition_events),initial_state_sha256=initial_hash,final=final,metrics=dict(max_force_N=s.max_force,max_penetration_mm=s.max_depth*1000,max_self_penetration_mm=s.max_selfdepth*1000,min_shaft_clearance_mm=s.min_shaft_clearance_m*1000,elapsed_s=final['task_t'],video_duration_s=final['t'],wall_s=time.monotonic()-wall,policy_calls=len(trace)-1),trace=trace,frames=frames,first_failure=first_failure)
    return record,np.array(qs),s.xml

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'screwdriver_lab/artifacts/evaluation');p.add_argument('--methods',nargs='+',default=list(METHODS));p.add_argument('--modes',nargs='+',default=['free','fixture']);p.add_argument('--cases',nargs='+');p.add_argument('--goal',type=float,default=90);p.add_argument('--policy');p.add_argument('--timestep',type=float,default=.002);a=p.parse_args()
    if not 0<a.goal<=120:raise ValueError('goal must be in (0,120] degrees')
    if a.timestep not in (.001,.002):raise ValueError('Supported timesteps: .001, .002')
    a.output.mkdir(parents=True,exist_ok=True);bank=training_bank();np.savez_compressed(a.output/'training_bank.npz',**bank)
    policy_cls=Policy
    if a.policy:
        module,name=a.policy.split(':');policy_cls=getattr(importlib.import_module(module),name)
    selected=[dict(c,timestep_s=a.timestep) for c in CASES if not a.cases or c['id'] in a.cases]
    manifest=dict(schema_version=2,status='corrected_initialization_rtx_revision',engine=f'MuJoCo {mujoco.__version__}',agent='scripted code-as-policy; no LLM calls',goal_deg=a.goal,methods=METHODS,cases=selected,limits=dict(LIMITS,goal_deg=a.goal),controller_hz=50,policy_hz=5,physics_hz=1/a.timestep,record_fps=30,acquisition_in_video=True,initialization='Collision-refined paired grasp, strict shaft clearance and distal-handle support; acquisition failures latch',synergy_training='3-component PCA on nominal kinematic reference; no held-out physics data',synergy_explained_variance=bank['variance'],grasp_sha256=hashlib.sha256((ROOT/'screwdriver_lab/artifacts/grasp.json').read_bytes()).hexdigest(),source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},results=[])
    for mode in a.modes:
        if mode not in ('free','fixture'):raise ValueError(mode)
        for case in selected:
            for method in a.methods:
                if method not in METHODS:raise ValueError(method)
                r,q,xml=run_trial(case,method,mode,bank,policy_cls,a.goal)
                (a.output/f'{r["id"]}.json').write_text(json.dumps(r));np.savez_compressed(a.output/f'{r["id"]}.npz',qpos=q);(a.output/f'{r["id"]}.xml').write_text(xml)
                manifest['results'].append(r);(a.output/'manifest.json').write_text(json.dumps(manifest))
                print(r['id'],r['outcome'],round(r['final']['angle_deg'],1),round(r['metrics']['max_force_N'],1),flush=True)
    print('Saved',len(manifest['results']),'paired diagnostic episodes',flush=True)

if __name__=='__main__':main()
