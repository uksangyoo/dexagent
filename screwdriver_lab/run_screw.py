"""Paired screw-in experiments and GPU-1-only RTX rendering."""
import argparse,concurrent.futures,json,os,subprocess
import numpy as np
from screwdriver_lab.screw_scene import ROOT,OUT
from screwdriver_lab.screw_sim import trial
from screwdriver_lab.screw_control import METHODS
from screwdriver_lab.export_rtx import export
CASES=[dict(id='nominal',name='Aligned · nominal grip',friction=.9,slot_yaw_deg=0),dict(id='slot20',name='Slot rotated 20°',friction=.9,slot_yaw_deg=20),dict(id='lowgrip',name='Low contact friction',friction=.25,slot_yaw_deg=0)]
SITE=ROOT/'screwdriver_site/dist'
def simulate(args):
 method,case=args;r,q,xml=trial(method,case,verbose=False)
 (OUT/(r['id']+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(r['id']+'.npz'),qpos=q);(OUT/(r['id']+'.xml')).write_text(xml)
 print(r['id'],r['outcome'],round(r['final']['depth_mm'],3),round(r['final']['t'],2),flush=True);return r

def render(r):
 stem='screw-in-'+r['id'];bundle=OUT/'bundles'/stem
 export((OUT/(r['id']+'.xml')).read_text(),np.load(OUT/(r['id']+'.npz'))['qpos'],bundle,r)
 output=SITE/'assets'/f'{stem}.mp4'
 with (bundle/'render.log').open('w') as log:
  subprocess.run([os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python'),str(ROOT/'screwdriver_lab/render_screw.py'),'--bundle',str(bundle),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4'))
 r.update(video=f'assets/{stem}.mp4',poster=f'assets/{stem}.jpg',rendering=json.loads(output.with_suffix('.render.json').read_text()))
 print('RTX',r['id'],flush=True);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args()
 if a.render:
  data=json.loads((OUT/'manifest.json').read_text())
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:data['results']=list(pool.map(render,data['results']))
  data['rendering']=dict(renderer='Cycles / NVIDIA OptiX',GPU=1,resolution=[1280,960],fps=30,samples=24)
  (SITE/'data/screw-in.json').write_text(json.dumps(data))
 else:
  with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(simulate,[(m,c) for c in CASES for m in METHODS]))
  data=dict(task='Drive a pre-aligned screw 1 mm into a workpiece and verify seating',methods=METHODS,cases=CASES,results=results)
  (OUT/'manifest.json').write_text(json.dumps(data))
if __name__=='__main__':main()
