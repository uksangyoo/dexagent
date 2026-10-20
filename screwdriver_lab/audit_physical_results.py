"""Recompute task acceptance from saved raw observations and model structure."""
import argparse,hashlib,json
import numpy as np,mujoco
from screwdriver_lab.physical_task import OUT

def audit(name):
    r=json.loads((OUT/(name+'.json')).read_text());raw=np.load(OUT/(name+'.npz'))
    x=raw['samples_1khz'];columns=list(raw['sample_columns']);col=lambda key:x[:,columns.index(key)]
    m=mujoco.MjModel.from_xml_string((OUT/(name+'.xml')).read_text())
    assert m.neq==0 and not np.any(m.body_gravcomp)
    screw_joints=[m.joint(key).id for key in ['screw_depth','screw_x','screw_y','screw_tilt_x','screw_tilt_y','screw_angle']]
    assert not np.any(m.jnt_limited[screw_joints])
    assert not any(int(j)in screw_joints for j in m.actuator_trnid[:,0])
    assert np.allclose(np.diff(col('t')),.001,atol=1e-8)
    peak=np.asarray(r['motor']['peak_motor_torque_Nm']);assert np.all(peak[:20]<=.11+1e-12)
    mask=(col('depth_mm')>=1.98)&(col('engaged')>0)&(col('screw_speed_rad_s')<.15)&(col('seat_N')>.1)
    streak=0;longest=0
    for valid in mask:streak=streak+1 if valid else 0;longest=max(longest,streak)
    if r['success']:assert streak>=400,(name,streak)
    seat_times=[e['t']for e in r['cycle_trace']if e.get('to')=='seat']
    drive=col('t')<(seat_times[0]if seat_times else float('inf'))
    loads=[];slides=[]
    for finger in ['thumb','middle']:
        returning=(col(finger+'_return')>0)&drive
        loaded=col(finger+'_N')>.01;sliding=col(finger+'_slip_mm_s')<-.1
        loads.append(float(loaded[returning].mean())if returning.any()else None)
        slides.append(float((loaded&sliding)[returning].mean())if returning.any()else None)
    result=dict(id=name,outcome=r['outcome'],raw_samples=len(x),raw_sha256=hashlib.sha256((OUT/(name+'.npz')).read_bytes()).hexdigest(),
        source_xml_sha256=hashlib.sha256((OUT/(name+'.xml')).read_bytes()).hexdigest(),
        no_equalities=True,no_body_gravity_compensation=True,unactuated_six_dof_screw=True,
        peak_finger_motor_Nm=float(peak[:20].max()),longest_stable_seat_s=longest*.001,final_stable_seat_s=streak*.001,
        final_depth_mm=float(col('depth_mm')[-1]),final_screw_angle_deg=float(col('screw_angle_deg')[-1]),
        max_penetration_mm=float(col('penetration_mm').max()),max_tilt_deg=float(col('tilt_deg').max()),
        drive_return_loaded_fraction=loads,drive_return_loaded_sliding_fraction=slides,
        score_window_end_s=seat_times[0]if seat_times else float(col('t')[-1]),
        notes=['Return fractions exclude seating hold; historical seat modes could retain a stale return label.',
               'Motor bounds cover compensation; per-command assertions prohibit applied tool/screw forces.',
               '1 kHz observations do not resolve sub-millisecond contact loss.'])
    (OUT/(name+'-audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');audit(p.parse_args().name)
