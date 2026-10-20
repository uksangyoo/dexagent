"""Package executable policy and exact source snapshots without robot duplication."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;LAB=ROOT/'screwdriver_lab';OUT=LAB/'artifacts/full_motion';SITE=ROOT/'screwdriver_site'
modules=['controlled_slip','controlled_slip_full','reference_control','reference_full','reference_cycle','slip_graph','slip_metrics','slip_hydro','hydro_backend','slip_compare','slip_load_sweep','slip_speed_sweep','slip_progress_graph','slip_media_check','slip_velocity_integral','slip_full_fixed_tripod','slip_transport','slip_validate','slip_csv','slip_contact_audit','publish_slip','slip_pack','slip_code_pack','slip_tip_grasp','export_slip_video','export_rtx','render_controlled_slip','render_slip_study','serve_screw','arm_scene','arm_sim','full_scene','full_motion','screw_scene','screw_sim','scene']
files=[LAB/(n+'.py')for n in modules]
files += [LAB/'artifacts'/n/'grasp.json'for n in ['controlled_slip_tip','controlled_slip_clear','reference_strategy']]
files += [LAB/'artifacts/reference_strategy'/n for n in ['CONTROLLED_SLIP_REPRODUCE.txt','observation.json','slip-environment.json']]
audit=json.loads((OUT/'slip-contact-audit-validation.json').read_text());files += [OUT/'source_snapshots'/(audit['source_sha256']+'_slip_contact_audit.py'),OUT/'slip-contact-audit-validation.json']
files += [SITE/'qa/slip_check.py',SITE/'dist/slip.js',SITE/'dist/slip.css',SITE/'dist/reference.css',SITE/'dist/compressed-data.js',SITE/'dist/index.html',OUT/'slip-validation.json',OUT/'slip-motion-audit.json']
for record in list(OUT.glob('slip-study-*.json'))+[OUT/'slip-verified-seating.json',OUT/'slip-full-fixed-tripod.json',OUT/'slip-velocity-integral.json']:
 r=json.loads(record.read_text())
 for n,digest in r['source_sha256'].items():
  p=OUT/'source_snapshots'/(digest+'_'+n);assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==digest;files.append(p)
with zipfile.ZipFile(SITE/'dist/assets/controlled-slip-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
 for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
print('Code bundle:',len(set(files)),'files')
