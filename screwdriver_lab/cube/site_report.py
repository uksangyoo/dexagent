"""Completed cube trials, with failed development attempts retained."""
import json,shutil,zipfile,gzip,csv
from pathlib import Path
import numpy as np
from .model import OUT
ROOT=Path(__file__).resolve().parents[2];SITE=ROOT/'screwdriver_site/dist'
REPLAY_ALIASES={
 'cube40-pitch30-hom-full-v16':'cube40-pitch30-hom-full-v12',
 'cube40-pitch30-hom-full-v9':'cube40-pitch30-hom-full-v12',
 'cube40-pitch30-cartesian-full-v6':'cube40-pitch30-cartesian-full-v12',
 'cube40-pitch30-object-full-v6':'cube40-pitch30-object-full-v12'}

def record(path):
 r=json.loads(path.read_text());npz=path.with_suffix('.npz')
 if not npz.exists()or 'samples'not in np.load(npz).files:return None
 a=np.load(npz);s=a['samples'];name=path.stem
 frames=[]
 for t,c in zip(a['time'],r['commands']):
  j=int(np.argmin(abs(s[:,0]-t)));v=s[j]
  frames.append(dict(t=float(t),phase=c['phase'],error=float(v[1]),inhand=float(v[2]),palm=float(v[3]),lift=float(v[4]),loads=v[5:8].tolist(),speed=float(v[8]),unexpected=float(v[9]),hold=float(v[10]),motor=v[11:37].tolist()))
 assets={};media_name=name
 if name in REPLAY_ALIASES and (OUT/(REPLAY_ALIASES[name]+'.render.json')).exists():
  target=REPLAY_ALIASES[name];other=np.load(OUT/(target+'.npz'))
  assert all(np.array_equal(a[k],other[k])for k in ['time','qpos','qvel','samples'])
  assert path.with_suffix('.xml').read_bytes()==(OUT/(target+'.xml')).read_bytes()
  media_name=target;assets['video_source_recording']=target;assets['video_equivalence']='Exact same recorded time, qpos, qvel and samples; verified before sharing this RTX replay.'
  for extension in ['.mp4','.jpg']:(SITE/'assets'/(name+extension)).unlink(missing_ok=True)
 for extension in ['.mp4','.jpg']:
  p=OUT/(media_name+extension)
  if extension=='.mp4'and not(OUT/(media_name+'.render.json')).exists():
   (SITE/'assets'/p.name).unlink(missing_ok=True);continue
  if p.exists():shutil.copy2(p,SITE/'assets'/p.name);assets['video'if extension=='.mp4'else'poster']='assets/'+p.name
 initial_path=OUT/(name+'-initial.json');initial=json.loads(initial_path.read_text())if initial_path.exists()else{}
 grasped=s[(s[:,4]>60)&(np.minimum(s[:,5],s[:,7])>.05)]
 # Secondary diagnostic only: keep the strict angular-speed success oracle.
 pose_ok=(s[:,1]<=5)&(s[:,3]<2)&(s[:,4]>=60)&(np.minimum(s[:,5],s[:,7])>.05)&(s[:,9]<.1)
 longest=current=0.
 for ok in pose_ok:
  current=current+.002 if ok else 0.;longest=max(longest,current)
 retained_speed=s[pose_ok,8]
 return dict(pose_retention_s=longest,pose_retention_excludes_speed_test=True,retained_speed_p95_deg_s=float(np.percentile(retained_speed,95))if len(retained_speed)else None,id=name,interface=r['interface'],outcome=('prepared_rotation_succeeded'if r.get('prepared_rotation_success')else r['outcome']),success=r['success'],prepared_rotation_success=r.get('prepared_rotation_success',False),checkpoint=r.get('checkpoint'),goal=r['goal_deg'],axis=r.get('goal_axis','z'),edge_mm=1000*initial.get('edge_m',.02),mass_g=1000*initial.get('mass_kg',.008),balanced=initial.get('balanced_pregrasp',False),pulp=initial.get('pulp_contact',False),parameters=r['parameters'],final=r['final'],best_goal_error_while_lifted_deg=float(min(grasped[:,1]))if len(grasped)else None,maximum_lift_mm=float(max(s[:,4])),maximum_palm_turn_deg=float(max(s[:,3])),maximum_unexpected_load_N=float(max(s[:,9])),frames=frames,graph=r['graph'],transitions=r['transitions'],provenance=r['provenance'],source_capture_limit='First v1 source hashes were collected at run end; concurrent development means those hashes are not guaranteed to identify executed source.'if name in ['cube-cartesian-v1','cube-object-v1']else None,**assets)

def main():
 records=[]
 for p in sorted(OUT.glob('*.json')):
  if any(x in p.stem for x in ['-initial','.progress']):continue
  r=record(p)
  if r:records.append(r)
 result=dict(task='cube_rotation',goal='90 degree in-hand rotation about the initial vertical axis after physical pickup',contract=dict(orientation_error_deg=5,lift_mm=60,palm_turn_deg=2,speed_deg_s=5,hold_s=.3),comparison_ids=[f'cube-{k}-v1'for k in ['cartesian','object','hom']],records=records,authoring='Codex-authored graphs execute observation-conditioned Python nodes. No independent runtime model API trial is claimed.',physics='Free rigid cube, hydroelastic fingertip contact, gravity, and bounded 20 + 6 robot motors. Pad friction 0.6; cube friction 0.5. Declared, uncalibrated material parameters.',success=any(r['success']and r['goal']==90 and r['axis']=='z' for r in records),precision_success=any(r['success']and r['goal']==30 for r in records))
 result['comparison_sets']=[dict(label='90° vertical rotation · 20 mm cube · first study',ids=result['comparison_ids'])]
 pitch=[f'cube40-pitch30-{k}-full-v2'for k in ['cartesian','object','hom']]
 if all(any(r['id']==name for r in records)for name in pitch):result['comparison_sets'].insert(0,dict(label='30° pinch-axis rotation · 40 mm cube · matched starting scene',ids=pitch))
 repaired=[f'cube40-pitch30-{k}-full-v6'for k in ['cartesian','object','hom']]
 if all(any(r['id']==name for r in records)for name in repaired):result['comparison_sets'].insert(0,dict(label='30° pinch-axis rotation · revised adapter handoff',ids=repaired))
 fixed=[f'cube40-pitch30-{k}-full-v12'for k in ['cartesian','object','hom']]
 if all(any(r['id']==name for r in records)for name in fixed):result['comparison_sets'].insert(0,dict(label='Earlier 30° comparison · interface-specific grip braking',ids=fixed))
 common=[f'cube40-pitch30-{k}-full-v16'for k in ['cartesian','object','hom']]
 if all(any(r['id']==name for r in records)for name in common):result['comparison_sets'].insert(0,dict(label='Latest: 30° rotation · common grip brake · matched adapters',ids=common))
 yaw_ids=['cube40-pitch30-hom-ym2-v9','cube40-pitch30-hom-full-v9','cube40-pitch30-hom-yp2-v9']
 if all(any(r['id']==name and r.get('video')for r in records)for name in yaw_ids):result['comparison_sets'].append(dict(label='Initial yaw −2° / 0° / +2° · same relative-contact policy',kind='initial_yaw',ids=yaw_ids))
 step_ids=['cube40-pitch30-hom-full-v12','cube40-pitch30-hom-dt125-v12']
 if all(any(r['id']==name and r.get('video')for r in records)for name in step_ids):result['comparison_sets'].append(dict(label='Physics timestep 250 / 125 μs · same relative-contact policy',kind='timestep',ids=step_ids))
 clear=[f'cube40-cleararm-{k}-y0-v1'for k in ['cartesian','object','hom']]
 if all(any(r['id']==name and r.get('video')for r in records)for name in clear):result['comparison_sets'].append(dict(label='30° pinch-axis rotation · alternate arm branch',ids=clear))
 campaign=OUT/'cleararm-campaign.json'
 result['sensitivity']=None
 if campaign.exists():
  study=json.loads(campaign.read_text());study['completed_ids']=[r['id']for r in records if r['id']in study['trial_ids']];study['complete']=len(study['completed_ids'])==len(study['trial_ids']);result['sensitivity']=study
 audit_path=OUT/'matched-timestep-v12-audit.json'
 result['timestep_audit']=json.loads(audit_path.read_text())if audit_path.exists()else None
 result['prepared_example']=next((r['id']for r in records if r.get('prepared_rotation_success')and r.get('video')),None)
 result['pickup_example']=next((r['id']for r in reversed(records)if r['success']and r.get('video')),None)
 with (SITE/'data/cube-trials.csv').open('w')as f:
  keys=['trial','interface','goal_deg','axis','initial_yaw_deg','prepared_start','success','outcome','duration_s','final_error_deg','best_lifted_error_deg','lift_mm','palm_turn_deg','hold_s','peak_unexpected_N','pose_retention_s','retained_speed_p95_deg_s']
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for r in records:
   writer.writerow(dict(trial=r['id'],interface=r['interface'],goal_deg=r['goal'],axis=r['axis'],initial_yaw_deg=r['parameters'].get('initial_yaw_deg',0),prepared_start=bool(r['checkpoint']),success=r['success'],outcome=r['outcome'],duration_s=r['final']['t'],final_error_deg=r['final']['orientation_error_deg'],best_lifted_error_deg=r['best_goal_error_while_lifted_deg'],lift_mm=r['final']['lift_mm'],palm_turn_deg=r['maximum_palm_turn_deg'],hold_s=r['final']['hold_s'],peak_unexpected_N=r['maximum_unexpected_load_N'],pose_retention_s=r['pose_retention_s'],retained_speed_p95_deg_s=r['retained_speed_p95_deg_s']))
 (OUT/'site-results.json').write_text(json.dumps(result,indent=2))
 with gzip.open(SITE/'data/cube-results.json.gz','wt')as f:json.dump(result,f,separators=(',',':'))
 (SITE/'data/cube-results.json').unlink(missing_ok=True)
 with zipfile.ZipFile(SITE/'assets/cube-evidence.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
  for p in (ROOT/'screwdriver_lab/cube').glob('*.py'):z.write(p,'source/'+p.name)
  for r in records:
   for suffix in ['.json','.xml','-initial.json','-initial.npz']:
    p=OUT/(r['id']+suffix)
    if p.exists():z.write(p,'records/'+p.name)
  raw_ids=set(result['comparison_sets'][0]['ids'])
  if result['pickup_example']:raw_ids.add(result['pickup_example'])
  for name in raw_ids:
   p=OUT/(name+'.npz');z.write(p,'records/'+p.name)
  for p in (OUT/'source_snapshots').glob('*'):z.write(p,'source_snapshots/'+p.name)
  for pattern in ['matched-*-audit.json','matched-*-repeatability.json','*-independent-hold-check.json','cleararm-campaign.json','*-arm-plan.json']:
   for p in OUT.glob(pattern):z.write(p,'audits/'+p.name)
  for p in (OUT/'policies').rglob('*'):
   if p.is_file()and '__pycache__'not in str(p):z.write(p,str(p.relative_to(OUT)))
  z.writestr('README.txt','Development evidence. No success is implied by a large rotation magnitude: full orientation error, retained lift, loaded contact and a stable hold must all pass. Initial penetrating configurations are rejected before simulation. Raw NPZ recordings for the latest matched comparison and featured completed pickup are included, together with initial states for every recorded trial. Other raw recordings remain in the local workspace. Early v1 source-hash capture limitations are recorded in the website data.\n')
 print(json.dumps(dict(completed=len(records),success=result['success'])))
if __name__=='__main__':main()
