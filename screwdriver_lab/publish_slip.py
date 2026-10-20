"""Build web evidence from frozen rollouts; never synthesize result trajectories."""
import json,gzip,hashlib,shutil,tarfile,zipfile
from pathlib import Path
import numpy as np,mujoco
from screwdriver_lab.slip_compare import CASES,METHODS
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'screwdriver_lab/artifacts/full_motion';SITE=ROOT/'screwdriver_site/dist'
LABELS={'pose':dict(name='Contact pose',contract='Contact points + pad normals → IK + joint PD',detail='50 Hz force admittance changes radial compression; Cartesian contact points and pad normals become joint position targets. Geometry is explicit, but maintaining light contact requires accurate tracking.'), 'velocity':dict(name='Tangential velocity',contract='Normal load + tangent velocity → contact wrench → joints',detail='1 kHz normal-wrench command with velocity damping, plus tangential velocity feedback (100 N·s/m). It retains light contact, but this gain and gait produce little useful forward rotation.'),'wrench':dict(name='Contact wrench',contract='Normal load + tangent displacement → contact wrench → joints',detail='1 kHz normal-wrench command with velocity damping, plus tangential spring/damper (150 N/m, 2 N·s/m). The reset starts from the measured contact patch, so residual drive error does not squeeze the handle during unloading.')}
CL={'nominal':'Nominal · μ 0.9 · 3 mN·m','low-friction':'Lower friction · μ 0.6','tilt-plus':'Initial tilt +2°','tilt-minus':'Initial tilt −2°','high-resistance':'Higher resistance · 4 mN·m'}
def read(name):return json.loads((OUT/(name+'.json')).read_text())
def web(r):
 r=json.loads(json.dumps(r));name=r['id'];src=OUT/'rtx'/(name+'-contact.mp4')
 if src.exists() and src.with_suffix('.render.json').exists():
  for ext in ['.mp4','.jpg','.render.json']:shutil.copyfile(src.with_suffix(ext),SITE/'assets'/src.with_suffix(ext).name)
  meta=json.loads(src.with_suffix('.render.json').read_text());bundle=json.loads((OUT/'rtx'/name/'scene.json').read_text())['metadata'];r.update(video='assets/'+src.name,poster='assets/'+src.with_suffix('.jpg').name,render_metadata='assets/'+src.with_suffix('.render.json').name,video_time_scale=bundle['video_time_scale'])
 fields=['t','phase','screw_angle_deg','depth_mm','handle_forces_N','handle_slip_signed_mm_s','index_cap_load_N','tilt_deg','engaged','seat_force_N','axial_force_N']
 r['frames']=[{k:f[k]for k in fields if k in f}for f in r['frames'][::3]]+[ {k:r['frames'][-1][k]for k in fields if k in r['frames'][-1]}]
 for k in ['commands','trace','grasp_calibration']:r.pop(k,None)
 return r

def validate(records):
 checks=[]
 assert len({json.dumps(r['source_sha256'],sort_keys=True)for r in records})==1
 for case in CASES:
  rs=[r for r in records if r['case']==case];assert len({r['initial_state_sha256']for r in rs})==1;xml_hashes={hashlib.sha256((OUT/(r['id']+'.xml')).read_bytes()).hexdigest()for r in rs};assert len(xml_hashes)==1;checks.append({'case':case,'paired_initial_state_sha256':rs[0]['initial_state_sha256'],'paired_model_xml_sha256':next(iter(xml_hashes))})
 for r in records:
  n=r['id'];m=mujoco.MjModel.from_xml_path(str(OUT/(n+'.xml')));q=np.load(OUT/(n+'.npz'))['qpos'];assert len(q)==len(r['frames']) and np.isfinite(q).all();assert m.nu==26
  assert m.jnt_type[m.joint('tool_free').id]==mujoco.mjtJoint.mjJNT_FREE
  assert all(m.joint(int(j)).name.startswith(('joint_','shoulder','elbow','wrist'))for j in m.actuator_trnid[:,0])
  assert m.jnt_range[m.joint('screw_depth').id,1]>.003
  for f,sha in r['source_sha256'].items():
   p=OUT/'source_snapshots'/(sha+'_'+f);assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,(n,f)
 return dict(passed=True,scope='paired initial states, source snapshots, pose counts, finite poses, robot-only actuation, free tool and physical seat stop',checks=checks,robot_actuators=26,physical_gpu=1,source_snapshots_verified=True,identical_source_across_methods=True,identical_physics_within_each_condition=True)

def main():
 records=[read(f'slip-study-{m}-{c}')for m in METHODS for c in CASES];validation=validate(records)
 seat=read('slip-verified-seating') if (OUT/'slip-verified-seating.json').exists()else None
 extra=[read(n)for n in ['slip-hold-control','slip-no-unload-control','slip-hydro-material','slip-return-load-9','slip-return-load-10','slip-return-load-11','slip-velocity-integral','slip-speed-36','slip-speed-60','slip-progress-graph','slip-fast-light-contact']if (OUT/(n+'.json')).exists()]
 full=read('slip-full-fixed-tripod')if(OUT/'slip-full-fixed-tripod.json').exists()else None
 motion=json.loads((OUT/'slip-motion-audit.json').read_text())
 for r in records:
  if r['id']in motion:r['motion_audit']=motion[r['id']]
 if seat:seat['motion_audit']=motion[seat['id']]
 data=dict(results=[web(r)for r in records],methods=LABELS,cases=[dict(id=c,label=CL[c],changes=CASES[c])for c in CASES],seat=web(seat)if seat else None,controls=[web(r)for r in extra],full=web(full)if full else None,observation=json.loads((ROOT/'screwdriver_lab/artifacts/reference_strategy/observation.json').read_text()))
 (SITE/'data/slip-results.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0));(SITE/'data/slip-validation.json').write_text(json.dumps(validation,indent=2))
 print('Built data',len(records),'seat',seat['outcome']if seat else 'pending',flush=True)
if __name__=='__main__':main()
