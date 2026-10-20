"""Mechanical and evidence checks for screw insertion; no rendering involved."""
import json,hashlib
import numpy as np,mujoco
from screwdriver_lab.screw_sim import ScrewSim
from screwdriver_lab.screw_scene import OUT,ROOT

def main():
 data=json.loads((OUT/'manifest.json').read_text());s=ScrewSim()
 driven=[s.m.joint(int(j)).name for j in s.m.actuator_trnid[:,0]]
 assert len(driven)==21 and 'feed' in driven
 assert all(n=='feed' or n.startswith('joint_') for n in driven)
 assert s.m.joint('tool_free').type==mujoco.mjtJoint.mjJNT_FREE
 initial_geometry={}
 for case in data['cases']:
  init=ScrewSim(case);shaft=init.m.geom('shaft').id;dist=[]
  for g in range(init.m.ngeom):
   name=init.m.body(init.m.geom_bodyid[g]).name
   if name=='palm' or name.startswith('link_'):dist.append(mujoco.mj_geomDistance(init.m,init.d,shaft,g,1.,None))
  clearance=min(dist)*1000;assert clearance>3 and init.pen<.0001
  initial_geometry[case['id']]=dict(shaft_clearance_mm=clearance,max_initial_penetration_mm=init.pen*1000)
 maxerror=0
 for c in data['cases']:
  runs=[r for r in data['results'] if r['case']==c['id']];assert len(runs)==5
  assert len(set(r['initial_state_sha256'] for r in runs))==1
  for r in runs:
   q=np.load(OUT/(r['id']+'.npz'))['qpos'];assert len(q)==len(r['frames']);assert r['frames'][0]['t']==0
   error=max(abs(f['depth_mm']-f['screw_angle_deg']/360) for f in r['frames']);maxerror=max(maxerror,error);assert error<.002
   assert np.isfinite(q).all()
   if r['success']:assert r['metrics']['max_penetration_mm']<.5
   else:assert all(f['penetration_mm']<.5 and max(f['finger_forces'])<=8 for f in r['frames'][:-1])
   if r['success']:
    final=r['final'];assert final['engaged'] and final['depth_mm']>=.98 and final['seat_force_N']>.1
    tail=[f for f in r['frames'] if f['t']>=final['t']-.39]
    assert len(tail)>=11
    assert all(f['engaged'] and f['depth_mm']>=.98 and f['screw_speed_rad_s']<.15 and .3<f['axial_force_N']<5 and f['seat_force_N']>.1 for f in tail)
 # Falsification: same active cap pressure and no commanded tangential turn.
 pressure=[]
 for k in range(1750):
  events=s.step(s.ik(0,0));assert not events
  if k%50==0:pressure.append(s.observe())
 final=s.observe();assert final['depth_mm']<.05 and final['seat_force_N']==0
 files=['screw_scene.py','screw_sim.py','screw_control.py','run_screw.py','render_screw.py']
 result=dict(passed=True,initial_geometry=initial_geometry,checks=['21 hand/feed actuators only; no screw or driver actuator','free driver; passive screw and swivel-cap joints','paired exact reset hashes for all five methods in each case','depth matches pitch times measured screw turn','successful runs meet continuous seating conditions in recorded tail','successful runs within penetration guard; failed runs terminate at first detected violation','35 s pressure-only control does not seat'],max_pitch_error_mm=maxerror,pressure_only_final=final,pressure_only_trace=pressure,source_sha256={n:hashlib.sha256((ROOT/'screwdriver_lab'/n).read_bytes()).hexdigest() for n in files},summary=f"Mechanical checks passed: no driver/screw actuator, paired resets, and depth–pitch consistency (maximum error {maxerror:.6f} mm). Successful recorded tails satisfy the seating rules. A 35 s pressure-only control reaches {final['depth_mm']:.3f} mm with no seating contact, versus the 1 mm objective.")
 (OUT/'validation.json').write_text(json.dumps(result,indent=2));(ROOT/'screwdriver_site/dist/data/screw-validation.json').write_text(json.dumps(result,indent=2));print(result['summary'],flush=True)
if __name__=='__main__':main()
