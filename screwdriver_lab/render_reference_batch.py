"""Export and RTX-render the corrected-strategy study on physical GPU 1 only."""
import subprocess,os,json,hashlib,concurrent.futures
from pathlib import Path
from screwdriver_lab.full_scene import OUT
ROOT=Path(__file__).resolve().parents[1];PY=os.environ.get('DEXAGENT_PHYSICS_PYTHON','/home/uyoo/miniconda3/envs/newton/bin/python');BPY=os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python');ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OPENBLAS_NUM_THREADS':'1'}
IDS=['reference-full-audited']+[f'reference-study-{c}-{m}'for c in ['nominal','friction','resistance']for m in ['point','pose','wrench']]
def render(name):
 scale=1 if 'study-nominal'in name else 3
 subprocess.run([PY,'-m','screwdriver_lab.export_full',name,'--scale',str(scale)],cwd=ROOT,env=ENV,check=True,stdout=subprocess.DEVNULL)
 bundle=OUT/'rtx'/name;digest=hashlib.sha256((bundle/'scene.json').read_bytes()+(bundle/'scene.npz').read_bytes()).hexdigest()
 with bundle.with_suffix('.log').open('w')as f:subprocess.run([BPY,'-m','screwdriver_lab.render_reference','--bundle',str(bundle),'--output',str(bundle.with_suffix('.mp4'))],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 meta=json.loads(bundle.with_suffix('.render.json').read_text());assert all('61:00'in d['id']for d in meta['devices']);meta['source_bundle_sha256']=digest;bundle.with_suffix('.render.json').write_text(json.dumps(meta,indent=2));return name
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
  for name in pool.map(render,IDS):print('REFERENCE_RTX_DONE',name,flush=True)
 print('ALL_REFERENCE_RTX_DONE',flush=True)
