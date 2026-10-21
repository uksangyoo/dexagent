"""Read-only metrics for continuous pickup rollouts; no inferred full-task credit."""
import argparse, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
from dexagent.sim.model import OUT
from dexagent.planning.geometry import ART

def audit(name):
    report=json.loads((OUT/(name+'.json')).read_text()); data=np.load(OUT/(name+'.npz'))
    m=mujoco.MjModel.from_xml_path(str(OUT/(name+'.xml')));d=mujoco.MjData(m)
    tool=m.body('screwdriver').id;palm=m.body('palm').id
    axes=[];relative=[];positions=[];palms=[];rotations=[]
    for q in data['qpos']:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        R=d.xmat[palm].reshape(3,3);axis=d.xmat[tool].reshape(3,3)[:,2]
        axes.append(axis.copy());relative.append(R.T@axis);positions.append(d.xpos[tool].copy());palms.append(d.xpos[palm].copy());rotations.append(R.copy())
    axes=np.array(axes);relative=np.array(relative);positions=np.array(positions);rotations=np.array(rotations)
    samples=data['samples'];times=data['time'];pivot=next((e['t']for e in report['trace']if e['to']=='pivot'),None)
    metrics=dict(id=name,outcome=report['outcome'],continuous_pickup_pivot_drive_success=bool(report.get('assembly_trace')and report.get('success')),
        pickup_cap_supported=report.get('pickup_success',False),physics='Drake SAP; hydroelastic pads, explicit thread contact',
        geometry='precision-slotted',maximum_tool_lift_mm=float((positions[:,2].max()-positions[0,2])*1000),
        final_tilt_deg=float(np.degrees(np.arccos(np.clip(axes[-1,2],-1,1)))),duration_s=report['final']['t'],
        top_cap_load_recorded=samples.shape[1]>41,
        maximum_unexpected_contact_N=float(samples[:,40].max())if samples.shape[1]>40 else None)
    if pivot is not None:
        at=int(np.searchsorted(times,pivot));at=min(at,len(times)-1)
        selected=samples[:,0]>=pivot;ss=samples[selected]
        rotation=np.degrees(np.arccos(np.clip(relative[at:]@relative[at],-1,1)))
        palmturn=np.degrees(Rotation.from_matrix(rotations[at:]@rotations[at].T).magnitude())
        metrics.update(pivot_start_s=pivot,tilt_at_pivot_start_deg=float(np.degrees(np.arccos(np.clip(axes[at,2],-1,1)))),
            maximum_inhand_axis_turn_after_pivot_deg=float(rotation.max()),maximum_palm_turn_after_pivot_deg=float(palmturn.max()),
            pivot_both_side_contacts_fraction=float(np.mean(np.minimum(ss[:,4],ss[:,6])>.05)),
            minimum_tilt_after_pivot_deg=float(ss[:,13].min()),
            maximum_top_cap_load_after_pivot_N=float(ss[:,41].max())if ss.shape[1]>41 else None,
            pivot_speed_p95_deg_s=float(np.degrees(np.percentile(ss[:,42],95)))if ss.shape[1]>42 else None,
            pivot_normal_load_range_N=[float(np.minimum(ss[:,4],ss[:,6]).min()),float(np.maximum(ss[:,4],ss[:,6]).max())])
        relative_rates=[];loaded=[]
        for frame in range(at,len(times)):
            d.qpos[:]=data['qpos'][frame];d.qvel[:]=data['qvel'][frame];mujoco.mj_forward(m,d)
            Jtool=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,Jtool,tool)
            pinch_axis=rotations[frame]@rotations[0].T@np.array([0.,1.,0.]);rates=[]
            for finger in [1,3]:
                Jfinger=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,Jfinger,m.body(f'link_{finger}_4').id)
                rates.append(float(pinch_axis@((Jtool-Jfinger)@d.qvel)))
            relative_rates.append(rates);nearest=np.argmin(abs(samples[:,0]-times[frame]));loaded.append(min(samples[nearest,4],samples[nearest,6])>.05)
        relative_rates=np.array(relative_rates);loaded=np.array(loaded);deltas=np.diff(times[at:]);valid=loaded[1:]&loaded[:-1]
        spin=np.sum(.5*(relative_rates[1:]+relative_rates[:-1])*deltas[:,None]*valid[:,None],axis=0)
        metrics['loaded_relative_rotation_about_initial_pinch_axis_deg']=np.degrees(spin).tolist()
        metrics['relative_rotation_metric']='Signed integral of tool-minus-finger angular velocity along the palm-carried initial pinch axis, only over adjacent frames with both side loads above 0.05 N; not a tangential-slip-distance measurement.'
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args();records=[audit(n)for n in a.names]
    destination=ART/'pickup-iteration-metrics.json';destination.write_text(json.dumps(dict(status='development attempts, not an abstraction ranking',trials=records),indent=2))
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
