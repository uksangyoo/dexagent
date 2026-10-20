"""Render the corrected evidence with at most two RTX jobs on the selected GPU."""
import concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from screwdriver_lab.scene import ROOT
from screwdriver_lab.export_rtx import export
import numpy as np

BASE=ROOT/'screwdriver_lab/artifacts/revision_rtx'
SITE=ROOT/'screwdriver_site/dist'
BLENDER=os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python')

def job(source,r):
    stem=f'r2-g{r["goal_deg"]:g}-{r["id"]}';bundle=BASE/'bundles'/stem
    export((source/f'{r["id"]}.xml').read_text(),np.load(source/f'{r["id"]}.npz')['qpos'],bundle,r)
    output=SITE/'assets'/f'{stem}.mp4'
    with (bundle/'render.log').open('w') as log:
        subprocess.run([BLENDER,str(ROOT/'screwdriver_lab/render_rtx.py'),'--bundle',str(bundle),'--output',str(output),'--samples','24'],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4'))
    metadata=json.loads(output.with_suffix('.render.json').read_text())
    r.update(video=f'assets/{stem}.mp4',poster=f'assets/{stem}.jpg',render_fps=30,end_card_s=1,rendering=metadata)
    print('RENDERED',stem,'frames',len(r['frames']),'seconds',round(metadata['elapsed_s'],1),flush=True)
    return r

def main():
    sets=[(BASE/'main','manifest.json',50),(BASE/'local_motor','local-motor.json',10)]
    datasets=[];jobs=[]
    for source,name,count in sets:
        d=json.loads((source/'manifest.json').read_text());assert len(d['results'])==count
        datasets.append((d,name))
        jobs.extend((source,r) for r in d['results'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,*args) for args in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            f.result();(BASE/'render_progress.json').write_text(json.dumps(dict(done=i+1,total=len(jobs))))
    for d,name in datasets:
        d['rendering']=dict(renderer='Blender Cycles / NVIDIA OptiX',RTX=True,resolution=[1280,960],fps=30,samples=24,acquisition_included=True)
        (SITE/'data'/name).write_text(json.dumps(d))
    print('ALL_RTX_COMPLETE',flush=True)
if __name__=='__main__':main()
