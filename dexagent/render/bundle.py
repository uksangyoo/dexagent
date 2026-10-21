"""Export recorded Drake states at native 25 fps for RTX playback."""
import argparse,json,numpy as np,mujoco
from dexagent.sim.model import OUT
from dexagent.render.export import export
from dexagent.sim.seating import geometry

def bundle(name):
    r=json.loads((OUT/(name+'.json')).read_text());a=np.load(OUT/(name+'.npz'));samples=a['samples'];frames=[]
    pickup='pickup development' in r['scope']
    assembly=r.get('assembly_final') is not None
    model=mujoco.MjModel.from_xml_path(str(OUT/(name+'.xml')));data=mujoco.MjData(model);data.qpos[:]=a['qpos'][0];mujoco.mj_kinematics(model,data)
    target=geometry(model,data)['flat_seat_remaining_mm']+float(samples[0,3])
    if assembly:
        engaged=next((k for k,c in enumerate(r['commands'])if c['phase']=='drive'),None)
        if engaged is not None:
            data.qpos[:]=a['qpos'][engaged];mujoco.mj_kinematics(model,data)
            j=int(np.argmin(abs(samples[:,0]-a['time'][engaged])))
            target=geometry(model,data)['flat_seat_remaining_mm']+float(samples[j,3])
    for k in range(0,len(a['qpos']),2):
        t=float(a['time'][k]);j=np.argmin(abs(samples[:,0]-t));s=samples[j];cmd=r['commands'][k]
        frames.append(dict(t=t,phase=cmd['phase']if pickup else'turn',contact_phase=cmd['phase'],screw_angle_deg=float(s[2]),depth_mm=float(s[3]),tilt_deg=float(s[13]),cap_force_N=float(s[5]),handle_spin_since_engagement_deg=float(s[1]),handle_forces_N=s[[4,6]].tolist(),handle_slip_signed_mm_s=s[[7,9]].tolist(),display_scale=1))
        frames[-1].update(cmd.get('assembly',{}))
        frames[-1]['slip_recorded']=bool(cmd.get('drive_command'))
    export((OUT/(name+'.xml')).read_text(),a['qpos'][::2],OUT/'rtx'/name,dict(method=r['interface']+' / hydroelastic',interface=r['interface'],case=('prepared assembly diagnostic'if r.get('checkpoint')else'continuous assembly')if assembly else'pickup development'if pickup else'checkpoint-resumed hold'if'checkpoint'in r else'prepared pose',seating_target_mm=target,video_time_scale=1,outcome=r['outcome'],frames=frames))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:bundle(name)
