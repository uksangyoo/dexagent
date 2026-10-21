"""Outcome measurements from immutable dynamics and command recordings.

Commanded cycles are deliberately distinct from observed loaded sliding.
This module cannot turn a prepared-pose trial into a pickup/full-task success.
"""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from dexagent.sim.model import OUT,ROOT
from dexagent.sim.seating import geometry


def summarize(name):
    r=json.loads((OUT/(name+'.json')).read_text());a=np.load(OUT/(name+'.npz'));s=a['samples']
    t=s[:,0];dt=np.diff(t,prepend=0);commands=r['commands'];ct=a['time']
    index=np.maximum(0,np.searchsorted(ct,t,side='right')-1)
    phases=np.array([c['phase']for c in commands])[index]
    epochs=np.r_[0,np.flatnonzero(phases[1:]!=phases[:-1])+1,len(s)]
    episodes=[]
    for start,stop in zip(epochs[:-1],epochs[1:]):
        phase=phases[start]
        if phase not in ['thumb_drive','middle_drive','return','reset']:continue
        finger=2 if phase=='thumb_drive' else 0 if phase=='middle_drive' else None
        cols=[6 if finger==2 else 4]if finger is not None else[4,6]
        slips=[9 if finger==2 else 7]if finger is not None else[7,9]
        f=s[start:stop,cols];v=s[start:stop,slips];step=dt[start:stop,None]
        loaded=f>.01;sliding=v<-.2
        distance=np.sum(np.maximum(-v,0)*step*loaded,axis=0)
        observed_sliding_time=np.sum(step*sliding,axis=0)
        loaded_fraction=np.sum(step*loaded*sliding,axis=0)/np.maximum(1e-12,observed_sliding_time)
        screw_delta=float(s[stop-1,2]-s[start,2])
        # 24 degrees on a 5 mm handle corresponds to 2.09 mm of surface arc.
        # Require >=1.5 mm observed loaded relative sliding, plus forward
        # screw rotation, to count a useful loaded-sliding episode in this study.
        verified=bool(min(distance)>=1.5 and min(loaded_fraction)>=.95 and screw_delta>=3 and max(s[start:stop,13])<3)
        episodes.append(dict(phase=str(phase),start_s=float(t[start]),stop_s=float(t[stop-1]),return_finger='middle'if finger==2 else'thumb'if finger==0 else'both',loaded_return_distance_mm=distance.tolist(),loaded_fraction_while_sliding=loaded_fraction.tolist(),screw_delta_deg=screw_delta,useful_loaded_sliding_episode=verified))
    sliding_count=sum(e['useful_loaded_sliding_episode']for e in episodes)
    # Distinguish in-hand manipulation from spinning the arm or the cap.
    model=mujoco.MjModel.from_xml_string((OUT/(name+'.xml')).read_text().replace('/home/uyoo/Projects/dexagent/',str(ROOT)+'/'))
    data=mujoco.MjData(model);palm=model.body('palm').id;cap=int(model.geom_bodyid[model.geom('cap').id])
    first=None;palm_excursion=[];cap_yaw=[]
    for q in a['qpos']:
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        R=data.xmat[palm].reshape(3,3)
        if first is None:
            first=R.copy();seat_target=geometry(model,data)['flat_seat_remaining_mm']+float(s[0,3])
        palm_excursion.append(float(np.degrees(Rotation.from_matrix(R@first.T).magnitude())))
        cap_yaw.append(np.arctan2(data.xmat[cap].reshape(3,3)[1,0],data.xmat[cap].reshape(3,3)[0,0]))
    cap_spin=np.degrees(np.unwrap(cap_yaw)-cap_yaw[0])
    return dict(id=name,interface=r['interface'],outcome=r['outcome'],scope=r['scope'],full_task_success=False,
                duration_s=float(t[-1]),screw_deg=float(s[-1,2]),handle_deg=float(s[-1,1]),depth_mm=float(s[-1,3]),
                seating_target_mm=seat_target,
                max_tilt_deg=float(max(s[:,13])),peak_normal_N=float(np.max(s[:,4:7])),
                useful_loaded_sliding_episodes=sliding_count,episodes=episodes,
                palm_max_rotation_deg=max(palm_excursion),cap_max_spin_deg=float(max(abs(cap_spin))),cap_final_spin_deg=float(cap_spin[-1]),
                pitch_residual_final_mm=float(s[-1,3]-s[-1,2]/360),
                trace=s[::10,:14].tolist(),parameters=r['parameters'],source_sha256=r.get('source_sha256',{}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');args=p.parse_args()
    for name in args.names:
        result=summarize(name);(OUT/(name+'.metrics.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items()if k not in ['episodes','trace','parameters','source_sha256']}))
