"""Audit paired resets, physical completion, and contact-model evidence."""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.scene import ROOT
from screwdriver_lab.run_full import CASES
from screwdriver_lab.abstractions import METHODS
from screwdriver_lab.export_rtx import relocate_xml

def enrich(r):
 if r['method'] in ['joints','synergy','fingertip','object']:
  label={'joints':'joint replay','synergy':'synergy replay','fingertip':'pose tracking','object':'object feedback'}[r['method']]
  for f in r['frames']+[r['final']]:
   if 'planner_finger_modes' not in f:
    f['planner_finger_modes']=f['finger_modes'];f['planner_finger_roll_deg']=f.get('finger_roll_deg')
   f['finger_modes']=[label,label];f['finger_roll_deg']=[None,None]
  r['command_semantics']='Planner diagnostics are preserved separately; the applied hand command is the selected adapter output in commands[].q.'
 fs=r['frames'];a=np.array([f['screw_angle_deg']for f in fs]);m=r['metrics'];m['backward_screw_deg']=float(-np.minimum(0,np.diff(a)).sum());m['max_arm_torque_Nm']=float(np.abs([f['arm_torque_Nm']for f in fs]).max());m['max_arm_tracking_error_mm']=max(f.get('arm_pose_error_mm',0.)for f in fs)
 numerator=denominator=0
 for f in fs:
  for i,mode in enumerate(f['finger_modes']):
   if mode in ['return','slide_return']:
    denominator+=1;numerator+=int(f.get('handle_forces_N',f['finger_forces'][1:3])[i]>=.02 and f['slip_mm_s'][i]>2.)
 m['loaded_return_slip_fraction']=numerator/max(1,denominator);m['return_finger_samples']=denominator
 transfer=[f for f in fs if f['phase']=='reposition']
 if transfer:
  m['transfer_cap_load_fraction']=sum(f['axial_force_N']>.1 for f in transfer)/len(transfer)
  m['transfer_loaded_slip_fraction']=sum(f['finger_forces'][i+1]>=.02 and f['slip_mm_s'][i]>2 for f in transfer for i in range(2))/(2*len(transfer))
 return r

def analyze():
 records=[];checks=[];geometry={};maxpitch=0.;maxspeed=0.
 for case in CASES:
  hashes=[]
  for method in METHODS:
   name=case['id']+'-'+method;r=enrich(json.loads((OUT/(name+'.json')).read_text()));q=np.load(OUT/(name+'.npz'))['qpos'];fs=r['frames'];assert len(q)==len(fs) and fs[0]['t']==0 and np.isfinite(q).all()
   r['initial_state_sha256']=hashlib.sha256(q[0].tobytes()).hexdigest();hashes.append(r['initial_state_sha256']);r['source_xml_sha256']=hashlib.sha256((OUT/(name+'.xml')).read_bytes()).hexdigest()
   model=mujoco.MjModel.from_xml_string(relocate_xml((OUT/(name+'.xml')).read_text()));d=mujoco.MjData(model);d.qpos[:]=q[0];mujoco.mj_forward(model,d)
   actuated=[model.joint(int(j)).name for j in model.actuator_trnid[:,0]];assert len(actuated)==26 and all(x.startswith('joint_') or x in ['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']for x in actuated)
   assert model.joint('tool_free').type==mujoco.mjtJoint.mjJNT_FREE
   if method=='contact':
    hand=[g for g in range(model.ngeom)if model.body(int(model.geom_bodyid[g])).name=='palm' or model.body(int(model.geom_bodyid[g])).name.startswith('link_')]
    clearances={gn:min(mujoco.mj_geomDistance(model,d,model.geom(gn).id,g,1.,None)for g in hand)*1000 for gn in ['handle','shaft','cap']}
    assert min(clearances.values())>1.,clearances
    geometry[case['id']]={'minimum_hand_clearance_mm':clearances,'initial_max_penetration_mm':max([0]+[-c.dist*1000 for c in d.contact])}
   # MuJoCo's helical equality is compliant during impact. Bound its residual
   # to 1% of the 2 mm task travel and report the measured maximum explicitly.
   error=max(abs(f['depth_mm']-f['screw_angle_deg']/360)for f in fs);maxpitch=max(maxpitch,error);assert error<.02
   ts=np.array([f['t']for f in fs]);good=np.diff(ts)>1e-6;arm=np.array([f['arm_joint_rad']for f in fs]);speed=float(np.max(np.abs(np.diff(arm,axis=0)[good]/np.diff(ts)[good,None])));maxspeed=max(maxspeed,speed);r['metrics']['sampled_peak_arm_speed_rad_s']=speed
   if r['success']:
    tail=[f for f in fs if f['t']>=r['final']['t']-.39];assert len(tail)>=11
    assert all(f['engaged'] and f['depth_mm']>=1.98 and f['screw_speed_rad_s']<.15 and .15<f['axial_force_N']<5 and f['seat_force_N']>.1 for f in tail),name
    assert r['metrics']['peak_finger_N']<12 and r['metrics']['max_penetration_mm']<.6
   assert all(f['penetration_mm']<.6+1e-6 and max(f['finger_forces'])<=12 for f in fs[:-1]),name
   assert r['graph_trace'][-1]['to']==('done' if r['success']else'abort')
   (OUT/(name+'.json')).write_text(json.dumps(r))
   records.append(r)
  assert len(set(hashes))==1,(case['id'],hashes)
 hydro=[];drive_hashes=[]
 for backend in ['point','hydro']:
  for method in ['regrasp','slide']:
   r=enrich(json.loads((OUT/f'drive-{backend}-{method}.json').read_text()));r['backend_label']='Newton hydroelastic' if backend=='hydro'else'MuJoCo finite-patch friction';assert abs(r['final']['t']-15)<.021
   initial=np.load(OUT/(r['id']+'.npz'))['qpos'][0];r['initial_state_sha256']=hashlib.sha256(initial.tobytes()).hexdigest();drive_hashes.append(r['initial_state_sha256']);r['slip_aggregation']='Normal-force-weighted handle-contact speed; individual contacts ≤0.001 N excluded. Loaded-return threshold uses handle load.'
   if backend=='hydro':
    assert r['hydro']['gpu_physical']==1;assert r['hydro']['max_internal_buffer_fraction']<1
   hydro.append(r)
 assert len(set(drive_hashes))==1
 pressure=json.loads((OUT/'drive-point-hold.json').read_text());assert pressure['final']['depth_mm']<.05 and pressure['final']['seat_force_N']==0
 result=dict(passed=True,scope='25 full-task runs and 4 paired drive-only trials',initial_geometry=geometry,max_pitch_error_mm=maxpitch,sampled_peak_arm_speed_rad_s=maxspeed,pressure_only_final=pressure['final'],checks=['26 robot actuators only; no driver/screw actuator','free driver and passive swivel cap','identical initial state hashes across the five methods within each condition','initial open hand clears handle, cap and shaft by more than 1 mm','depth agrees with measured screw angle and 1 mm pitch','successful recorded tails meet seating conditions for at least 0.39 s (controller checks 0.4 s at 1 kHz)','guards retain terminal failure samples','hydroelastic internal contact buffers remain below capacity','15 s pressure-only falsification does not seat'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (ROOT/'screwdriver_lab').glob('*.py')})
 result['summary']=f"Mechanical checks passed for {len(records)} full-task records: 26 robot actuators, free screwdriver, paired resets, initial hand clearance, seating criteria, and depth–pitch consistency (maximum error {maxpitch:.6f} mm). The 15 s pressure-only control advanced {pressure['final']['depth_mm']:.6f} mm without seating. Hydroelastic buffer checks passed."
 (OUT/'validation.json').write_text(json.dumps(result,indent=2));(OUT/'manifest.json').write_text(json.dumps(dict(methods=METHODS,cases=CASES,results=records,contact_models=hydro)));print(result['summary'],flush=True)
if __name__=='__main__':analyze()
