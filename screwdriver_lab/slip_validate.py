"""Mechanical and provenance checks independent of rendered appearance."""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco
from screwdriver_lab.publish_slip import OUT,SITE,validate,read
records=[read(p.stem)for p in sorted(OUT.glob('slip-study-*.json'))];result=validate(records);seat=read('slip-verified-seating');results=[]
for r in records+[seat]:
 name=r['id'];m=mujoco.MjModel.from_xml_path(str(OUT/(name+'.xml')));d=mujoco.MjData(m);q=np.load(OUT/(name+'.npz'))['qpos'];depth=m.joint('screw_depth').qposadr[0];angle=m.joint('screw_angle').qposadr[0]
 # Soft equality constraint: require error below 1% of the 1 mm thread pitch.
 # Ideal thread linkage: depth = clockwise hinge angle times pitch.
 reported=np.array([f['depth_mm'] for f in r['frames']]);assert np.max(abs(q[:,depth]*1000-reported))<1e-8
 consistency=np.max(abs(q[:,depth]-q[:,angle]*.001/(2*np.pi)))*1000;assert consistency<.01,(name,consistency)
 d.qpos[:]=q[0];mujoco.mj_forward(m,d);initial_pen=max([max(0,-c.dist)for c in d.contact]+[0])*1000;assert initial_pen<.3,(name,initial_pen)
 assert r['metrics']['max_penetration_mm']<.6,(name,r['metrics']['max_penetration_mm'])
 assert r['metrics']['controlled_slip']['sample_rate_Hz']==1000
 for filename,digest in r['source_sha256'].items():
  f=OUT/'source_snapshots'/(digest+'_'+filename);assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
 results.append(dict(id=name,outcome=r['outcome'],initial_penetration_mm=initial_pen,max_thread_relation_error_mm=float(consistency),frames=len(q),initial_state_sha256=r['initial_state_sha256']))
assert seat['success'] and seat['outcome']=='seated';f=seat['final'];assert f['depth_mm']>=1.98 and f['engaged'] and f['screw_speed_rad_s']<.15 and .15<f['axial_force_N']<5 and f['seat_force_N']>.1
# Independent 30 Hz samples over the last .4 seconds must meet the same seat test.
last=[o for o in seat['frames'] if o['t']>=f['t']-.39];assert len(last)>=11
assert all(o['engaged'] and o['depth_mm']>=1.98 and o['screw_speed_rad_s']<.15 and .15<o['axial_force_N']<5 and o['seat_force_N']>.1 for o in last)
result.update(records=results,seat_final_sample_checks=True,seat_verification_samples=len(last),physics_note='All tests concern this idealized simulator, not physical hardware transfer.');(SITE/'data/slip-validation.json').write_text(json.dumps(result,indent=2));(OUT/'slip-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
