"""RTX render queue restricted to physical GPU 1."""
import argparse,os,subprocess,concurrent.futures,json,hashlib
from pathlib import Path
from screwdriver_lab.full_scene import OUT
PY=os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python')
p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
def run(name):
 dest=OUT/'rtx'/name
 def input_hash():return hashlib.sha256((dest/'scene.json').read_bytes()+(dest/'scene.npz').read_bytes()).hexdigest()
 source_hash=input_hash()
 if dest.with_suffix('.render.json').exists():
  previous=json.loads(dest.with_suffix('.render.json').read_text())
  if previous.get('renderer_sha256')==hashlib.sha256(Path(__file__).with_name('render_full.py').read_bytes()).hexdigest() and previous.get('source_bundle_sha256')==source_hash:return name+' rendered'
 with dest.with_suffix('.log').open('w')as f:
  code=subprocess.call([PY,'-m','screwdriver_lab.render_full','--bundle',str(dest),'--output',str(dest.with_suffix('.mp4'))],env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OPENBLAS_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
 if code:raise RuntimeError(name+' renderer failed')
 r=json.loads(dest.with_suffix('.render.json').read_text());assert all('61:00' in d['id']for d in r['devices']);assert input_hash()==source_hash,'Export bundle changed during rendering';r['source_bundle_sha256']=source_hash;dest.with_suffix('.render.json').write_text(json.dumps(r,indent=2));return name+' rendered'
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:
 for result in pool.map(run,a.ids):print(result,flush=True)
