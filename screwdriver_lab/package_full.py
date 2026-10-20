"""Portable code/assets plus bounded trace and RTX media archives."""
import json,zipfile,zlib
from pathlib import Path
from screwdriver_lab.scene import ROOT
from screwdriver_lab.full_scene import OUT
SITE=ROOT/'screwdriver_site/dist';ASSETS=SITE/'assets'
def write_zip(output,paths):
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for path in sorted(set(paths)):
   assert path.is_file(),path;z.write(path,path.relative_to(ROOT))
 with zipfile.ZipFile(output)as z:assert z.testzip()is None
 assert output.stat().st_size<16*1024**2,(output,output.stat().st_size)
 return 'assets/'+output.name

def packs(prefix,paths):
 groups=[];group=[];size=0
 for path in paths:
  estimate=len(zlib.compress(path.read_bytes(),6))+1024
  if group and size+estimate>9*1024**2:groups.append(group);group=[];size=0
  group.append(path);size+=estimate
 if group:groups.append(group)
 return [write_zip(ASSETS/f'{prefix}-{i}.zip',g)for i,g in enumerate(groups,1)]

def main(core_only=False,media_archives=False):
 data=json.loads((SITE/'data/full-results.json').read_text());records=data['results']+data['contact_models'];ids=[r['id']for r in records]+['drive-point-hold']
 if (OUT/'hydro-full-probe.json').exists():ids+=['hydro-full-probe']
 ids+=[r['id']for r in data.get('rank_sensitivity',[])if r['id']not in ids]
 traces=[OUT/(name+ext)for name in ids for ext in ['.json','.npz','.xml']]
 if not core_only:data['trace_packs']=packs('full-traces',traces)
 media=[ASSETS/'human-reference.mp4',ASSETS/'human-reference.jpg']
 for r in records:
  if 'video'in r:media.extend([SITE/r['video'],SITE/r['poster'],SITE/r['render_metadata']])
 if not core_only:data['media_packs']=packs('full-media',media)if media_archives else []
 (SITE/'data/full-results.json').write_text(json.dumps(data,separators=(',',':')))
 core=list((ROOT/'screwdriver_lab').glob('*.py'))+[ROOT/'screwdriver_lab/FULL_MOTION.md',ROOT/'screwdriver_lab/requirements.txt',ROOT/'screwdriver_lab/graph/workflow.json',ROOT/'screwdriver_lab/artifacts/grasp.json',ROOT/'screwdriver_lab/artifacts/screw_in/precision_grasp.json',OUT/'abstraction-bank.npz',OUT/'abstraction-bank.json',OUT/'validation.json']
 core+=[p for p in (ROOT/'third-party/universal_robots_ur5e').rglob('*')if p.is_file()]
 core+=[ROOT/'third-party/tesollo_model/LICENSE',ROOT/'third-party/tesollo_model/dg5fs/dg5fs_right.xml']+list((ROOT/'third-party/tesollo_model/dg5fs/meshes/dg5fs_right').glob('*'))
 core+=[ROOT/'dextrain/__init__.py',ROOT/'dextrain/resource_guard.py']
 core+=[SITE/n for n in ['index.html','full.js','full.css','screw.css','style.css','data/full-results.json','data/full-validation.json','data/workflow.json']]
 write_zip(ASSETS/'full-pipeline.zip',core)
 print('Pipeline',round((ASSETS/'full-pipeline.zip').stat().st_size/1048576,2),'MB;',len(data['trace_packs']),'trace packs;',len(data['media_packs']),'media packs',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--core-only',action='store_true');p.add_argument('--media-archives',action='store_true',help='Create optional duplicate offline media ZIPs; omit for hosted archive size limits');a=p.parse_args();main(a.core_only,a.media_archives)
