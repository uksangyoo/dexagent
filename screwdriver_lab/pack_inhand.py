"""Preserve full evidence and reproducible source within per-file hosting limits."""
from pathlib import Path
import json,tarfile,zipfile,hashlib
from screwdriver_lab.full_scene import OUT,ROOT
from screwdriver_lab.publish_inhand import DEVELOPMENT,snapshot
ASSETS=ROOT/'screwdriver_site/dist/assets'
def pack(label,names):
 path=ASSETS/('inhand-'+label+'.tar.xz')
 if path.exists()and path.stat().st_size<16*2**20:
  with tarfile.open(path,'r:xz')as prior:
   expected=[OUT/(n+ext)for n in names for ext in ['.json','.npz','.xml']]
   if set(prior.getnames())=={str(p.relative_to(ROOT))for p in expected}and all(hashlib.sha256(prior.extractfile(str(p.relative_to(ROOT))).read()).digest()==hashlib.sha256(p.read_bytes()).digest()for p in expected):
    print('Verified existing',path.name,flush=True);return
 with tarfile.open(path,'w:xz',preset=6)as tar:
  for name in names:
   for suffix in ['.json','.npz','.xml']:
    p=OUT/(name+suffix);tar.add(p,arcname=str(p.relative_to(ROOT)))
 assert path.stat().st_size<16*2**20,(path,path.stat().st_size)
 print(path.name,round(path.stat().st_size/2**20,2),flush=True)
def main():
 comparison=[f'pivot-study-{m}-{c}'for m in ['pose','velocity','wrench']for c in ['nominal','offset-minus','offset-plus']];development=[n for n in DEVELOPMENT if(OUT/(n+'.json')).exists()];names=['pivot-captured-long']+comparison+development
 pack('full-record',['pivot-captured-long']);pack('comparison',comparison);pack('development-1',development[:7]);pack('development-2',development[7:])
 files=[]
 for name in names:
  r=json.loads((OUT/(name+'.json')).read_text());snapshot(r)
  for filename,sha in r['source_sha256'].items():files.append(OUT/'source_snapshots'/(sha+'_'+filename))
 files += [ROOT/'screwdriver_lab'/n for n in ['inhand_policy.py','pivot_spin_controller.py','pivot_pose_controller.py','run_inhand_comparison.py','audit_inhand.py','publish_inhand.py','pack_inhand.py','export_inhand_video.py','render_inhand_pivot.py','render_inhand_study.py']]
 files += [ROOT/'screwdriver_lab/artifacts/reference_strategy/INHAND_REPRODUCE.txt',OUT/'pivot-validation.json',OUT/'pivot-captured-long-audit.json',ROOT/'screwdriver_site/dist/pivot.js']
 # Include all development entrypoints so every failure remains executable.
 files += [ROOT/'screwdriver_lab'/n for n in ['slip_inhand_pivot.py','slip_inhand_tuned.py','slip_inhand_soft.py','slip_inhand_balanced.py','slip_inhand_capload.py','slip_inhand_indexpatch.py','slip_inhand_surface.py','slip_inhand_rate.py','slip_inhand_rate_lift.py','slip_inhand_damped.py','slip_inhand_loaded.py','slip_inhand_contactframe.py','pivot_cap_servo.py','pivot_surface_servo.py']]
 with zipfile.ZipFile(ASSETS/'inhand-controller-code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
  for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
 print('Code files',len(set(files)))
if __name__=='__main__':main()
