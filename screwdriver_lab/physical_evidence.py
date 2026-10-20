"""Join exact reached-state continuations and export recorded RTX trajectories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from screwdriver_lab.physical_task import OUT
from screwdriver_lab.export_rtx import export

def join(name):
    r=json.loads((OUT/(name+'.json')).read_text());prefix=r['prefix'];p=json.loads((OUT/(prefix+'.json')).read_text())
    pq=np.load(OUT/(prefix+'.npz'))['qpos'];cq=np.load(OUT/(name+'.npz'))['qpos']
    xml=(OUT/(name+'.xml')).read_text();meta=json.loads((OUT/'checkpoints'/(prefix+'.json')).read_text())
    assert hashlib.sha256(xml.encode()).hexdigest()==meta['xml_sha256']
    assert r['checkpoint_sha256']==hashlib.sha256((OUT/'checkpoints'/(prefix+'.npz')).read_bytes()).hexdigest()
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);a=np.load(OUT/'checkpoints'/(prefix+'.npz'))
    mujoco.mj_setState(m,d,a['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    assert np.array_equal(pq[-1],d.qpos)
    assert abs(p['frames'][-1]['t']-meta['time'])<1e-9
    assert 0<r['frames'][0]['t']-meta['time']<.00101
    frames=[*p['frames'],*r['frames']];poses=np.concatenate([pq,cq]);assert len(frames)==len(poses)
    first=next(f for f in frames if f.get('phase')=='lift');P0=np.array(first['palm_rotation']);axis0=P0.T@np.array(first['tool_axis'])
    start=r['frames'][0];spin0=start['tool_angle_deg'];palm0=np.array(start['palm_rotation']);spins=[]
    for f in frames:
        P=np.array(f['palm_rotation']);axis=P.T@np.array(f['tool_axis'])
        f['inhand_axis_turn_deg']=float(np.degrees(np.arccos(np.clip(np.dot(axis0,axis),-1,1))))
        f['palm_turn_since_lift_deg']=float(np.degrees(np.linalg.norm(Rotation.from_matrix(P@P0.T).as_rotvec())))
        f['handle_spin_since_engagement_deg']=f['tool_angle_deg']-spin0 if f['t']>=start['t'] else 0.
        if f['t']>=start['t']:spins.append(float(np.degrees(Rotation.from_matrix(P@palm0.T).as_rotvec()[2])))
    result=dict(r,id=name+'-full',frames=frames,scope='Open pickup plus exact-state continuation',backend='MuJoCo CPU / explicit thread contact',
        source_chain=[prefix,name],source_hashes={n:hashlib.sha256((OUT/(n+'.json')).read_bytes()).hexdigest()for n in [prefix,name]},
        continuity_audit=dict(exact_checkpoint_state=True,boundary_gap_s=r['frames'][0]['t']-meta['time'],xml_sha256=meta['xml_sha256'],max_drive_palm_axial_rotation_deg=max(abs(x)for x in spins)),
        prefix_physics=p['physics'],initial_clearance_mm=p.get('initial_hand_clearance_mm',p.get('initial_clearance_mm',p.get('initial_clearance'))))
    (OUT/(name+'-full.json')).write_text(json.dumps(result));np.savez_compressed(OUT/(name+'-full.npz'),qpos=poses);(OUT/(name+'-full.xml')).write_text(xml)
    return result,poses,xml

def bundle(name,out,fps=12,drive_scale=30,cycle=False):
    r,q,xml=join(name);ts=np.array([f['t']for f in r['frames']]);handoff=json.loads((OUT/(name+'.json')).read_text())['frames'][0]['t'];end=ts[-1]
    segments=[(0,handoff,2.),(handoff,end,drive_scale)]if not cycle else [(handoff,min(end,handoff+13.),1.)]
    ix=[];scales=[];mapping=[]
    for lo,hi,scale in segments:
        mapping.append(dict(video_start_s=len(ix)/fps,simulation_start_s=lo,simulation_end_s=hi,time_scale=scale))
        for t in np.arange(lo,hi,scale/fps):ix.append(int(np.argmin(abs(ts-t))));scales.append(scale)
    ix.append(int(np.argmin(abs(ts-segments[-1][1]))));scales.append(scales[-1])
    r['frames']=[dict(r['frames'][i],display_scale=s,source_frame_index=i)for i,s in zip(ix,scales)]
    r.update(method='Physical thread contact',case='CPU dynamics / bounded motors',presentation_segments=mapping,video_time_scale=1)
    if cycle:r['outcome']='excerpt end'
    export(xml,q[ix],OUT/'rtx'/out,r);print(out,len(ix),r['continuity_audit'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--output',required=True);p.add_argument('--fps',type=int,default=12);p.add_argument('--drive-scale',type=float,default=30);p.add_argument('--cycle',action='store_true');a=p.parse_args();bundle(a.name,a.output,a.fps,a.drive_scale,a.cycle)
