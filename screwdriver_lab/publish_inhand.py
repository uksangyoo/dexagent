"""Publish full-task evidence separately from the frozen drive-only comparison."""
from pathlib import Path
import json,gzip,hashlib,shutil,numpy as np,mujoco,csv
from screwdriver_lab.audit_inhand import audit
from screwdriver_lab.full_scene import OUT,ROOT
SITE=ROOT/'screwdriver_site/dist'
DEVELOPMENT={'pivot-gravity-06':'Initial gravity-assisted pivot','pivot-captured-07':'Measured side-contact handoff','pivot-force-feedback-08':'Side normal-force feedback','pivot-soft-09':'Softer pads, 15 ms','pivot-balanced-10':'Intermediate pads, 8 ms','pivot-cap-load-11':'Force-limited cap transport','pivot-index-patch-12':'Measured index contact patch','pivot-surface-13':'Compliant axial side motion','pivot-rate-14':'Pivot-rate feedback after lift','pivot-rate-lift-15':'Pivot-rate feedback during lift','pivot-damped-16':'Stronger normal damping','pivot-loaded-17':'Higher 0.10 N reset load','pivot-contact-frame-18':'Contact-frame grasp normalization'}
def read(name):return json.loads((OUT/(name+'.json')).read_text())
def snapshot(r):
 for name,digest in r['source_sha256'].items():
  dest=OUT/'source_snapshots'/(digest+'_'+name);src=ROOT/'screwdriver_lab'/name
  if not dest.exists()and src.exists()and hashlib.sha256(src.read_bytes()).hexdigest()==digest:dest.write_bytes(src.read_bytes())
  assert dest.exists()and hashlib.sha256(dest.read_bytes()).hexdigest()==digest,(r['id'],name,digest)
def compact(r,include_frames=False):
 r=dict(r)
 for key in ['commands','trace','grasp_calibration']:r.pop(key,None)
 if include_frames:
  fields=['t','phase','contact_phase','tilt_deg','inhand_axis_turn_deg','palm_turn_since_lift_deg','handle_spin_since_engagement_deg','screw_angle_deg','depth_mm','handle_forces_N','handle_slip_signed_mm_s','engaged','seat_force_N','axial_force_N'];r['frames']=[{k:f[k]for k in fields if k in f}for f in r['frames'][::3]]+[{k:r['frames'][-1][k]for k in fields if k in r['frames'][-1]}]
 else:r.pop('frames',None)
 return r

def build():
 full,summary=audit('pivot-captured-long');snapshot(full);full['audit']=summary
 trials=[]
 for name,label in DEVELOPMENT.items():
  if not(OUT/(name+'.json')).exists():continue
  r=read(name);snapshot(r);r['label']=label;trials.append(compact(r))
 records=[read('pivot-study-'+m+'-'+c)for m in ['pose','velocity','wrench']for c in ['nominal','offset-minus','offset-plus']];checks=[]
 assert len({json.dumps(r['source_sha256'],sort_keys=True)for r in records})==1
 for y in [-2,0,2]:
  group=[r for r in records if r['initial_y_mm']==y];assert len(group)==3;assert len({r['initial_state_sha256']for r in group})==1;assert len({hashlib.sha256((OUT/(r['id']+'.xml')).read_bytes()).hexdigest()for r in group})==1;checks.append(dict(initial_y_mm=y,matched_initial_state_sha256=group[0]['initial_state_sha256']))
 for r in records+[full]:
  snapshot(r);name=r['id'];m=mujoco.MjModel.from_xml_path(str(OUT/(name+'.xml')));q=np.load(OUT/(name+'.npz'))['qpos'];assert np.isfinite(q).all()and len(q)==len(r['frames']);assert m.nu==26 and m.jnt_type[m.joint('tool_free').id]==mujoco.mjtJoint.mjJNT_FREE;assert all(m.joint(int(i)).name.startswith(('joint_','shoulder','elbow','wrist'))for i in m.actuator_trnid[:,0]);r['skill_success']=bool(r['final']['screw_angle_deg']>=90 and r['final']['engaged']and r['outcome']in ['timeout','seated'])
 f=full['frames'][-13:];assert all(x['depth_mm']>=1.98 and x['engaged']and x['screw_speed_rad_s']<.15 and .15<x['axial_force_N']<5 and x['seat_force_N']>.1 for x in f)
 old=np.load(OUT/'pivot-captured-07.npz')['qpos'];canonical=np.load(OUT/'pivot-study-wrench-nominal.npz')['qpos'];long=np.load(OUT/'pivot-captured-long.npz')['qpos'];assert np.array_equal(old,canonical);assert np.array_equal(old[:-1],long[:len(old)-1])
 validation=dict(passed=True,matched_trials=9,initial_conditions=checks,identical_sources=True,robot_only_actuation=True,free_tool=True,canonical_nominal_matches_legacy_short_exactly=True,long_prefix_matches_short_exactly=True,seat_final_samples=len(f),initial_hand_clearance_mm=full['initial_hand_clearance_mm'],maximum_recorded_penetration_mm=full['metrics']['max_penetration_mm'],scope='Simulation, deterministic trials; 1 kHz aggregate contact metrics and 30 Hz rigid-pose audit. Seating does not imply continuous loaded sliding.')
 (SITE/'data/pivot-validation.json').write_text(json.dumps(validation,indent=2));(OUT/'pivot-validation.json').write_text(json.dumps(validation,indent=2));(SITE/'data/pivot-motion-audit.json').write_text(json.dumps(summary,indent=2))
 media={}
 for key,bundle_name in [('intro','pivot-captured-07-intro'),('overview','pivot-full-overview')]:
  video=OUT/'rtx'/(bundle_name+'-motion.mp4');meta=video.with_suffix('.render.json');bundle=json.loads((OUT/'rtx'/bundle_name/'scene.json').read_text())['metadata']
  if not video.exists()or not meta.exists():continue
  render=json.loads(meta.read_text());assert all('61:00'in d['id']and d['type']=='OPTIX'for d in render['devices']);assert render['source_bundle_sha256']==hashlib.sha256((OUT/'rtx'/bundle_name/'scene.npz').read_bytes()).hexdigest()
  for ext in ['.mp4','.jpg','.render.json']:shutil.copyfile(video.with_suffix(ext),SITE/'assets'/video.with_suffix(ext).name)
  media[key]=dict(video='assets/'+video.name,poster='assets/'+video.with_suffix('.jpg').name,metadata='assets/'+meta.name,fps=render['fps'],segments=bundle['presentation_segments'],source_id=bundle['source_id'],frames=[{k:f[k]for k in ['t','phase','display_scale','tilt_deg','screw_angle_deg','depth_mm']}for f in bundle['frames']])
 downloads=[dict(path='assets/inhand-controller-code.zip',label='Full policy + controller source',detail='Executable graph, source snapshots and reproduction commands'),dict(path='assets/inhand-full-record.tar.xz',label='Complete seated rollout',detail='All recorded poses, observations and physical scene'),dict(path='assets/inhand-comparison.tar.xz',label='Nine matched full-task trials',detail='Three interfaces × three pickup offsets'),dict(path='assets/inhand-development-1.tar.xz',label='Development trials · part 1',detail='Contact-model and handoff refinements'),dict(path='assets/inhand-development-2.tar.xz',label='Development trials · part 2',detail='Motion, damping and reset-load refinements'),dict(path='data/pivot-validation.json',label='Validation',detail='Matched resets, source hashes, free tool and seating checks'),dict(path='data/pivot-motion-audit.json',label='Tool / palm motion audit',detail='In-hand pivot and axial spin measured separately')]
 data=dict(full=compact(full,True),comparison=[compact(r)for r in records],development=trials,media=media,downloads=downloads);(SITE/'data/pivot-results.json.gz').write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
 with (SITE/'data/pivot-summary.csv').open('w')as f:
  writer=csv.writer(f);writer.writerow(['group','id','interface','offset_y_mm','outcome','time_s','screw_deg','depth_mm','thumb_return_loaded','middle_return_loaded','thumb_dropout_s','middle_dropout_s'])
  for group,rs in [('full_demonstration',[full]),('matched_comparison',records),('development',trials)]:
   for r in rs:
    c=r['metrics']['controlled_slip'];writer.writerow([group,r['id'],r.get('interface',''),r.get('initial_y_mm',''),r['outcome'],r['final']['t'],r['final']['screw_angle_deg'],r['final']['depth_mm'],*c['return_loaded_fraction'],*c['longest_contact_dropout_s']])
 print('Built',len(records),'matched trials;',len(trials),'development trials;',len(media),'RTX videos')
if __name__=='__main__':build()
