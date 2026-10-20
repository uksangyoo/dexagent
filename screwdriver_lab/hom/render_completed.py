"""Render only completed, recorded trials, using physical GPU 1."""
import os,time,subprocess,argparse
os.environ['CUDA_VISIBLE_DEVICES']='1'
from screwdriver_lab.hom.model import OUT,ROOT
from screwdriver_lab.hom.export import bundle

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();deadline=time.monotonic()+1800
    while not (OUT/(a.name+'.json')).exists():
        if time.monotonic()>deadline:raise TimeoutError('Trial has not completed; no video was fabricated')
        time.sleep(2)
    bundle(a.name)
    subprocess.run([os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python'),'-m','screwdriver_lab.hom.render','--bundle',str(OUT/'rtx'/a.name),'--output',str(OUT/'rtx'/(a.name+'.mp4')),'--fps','25','--no-end-card'],cwd=ROOT,env=os.environ.copy(),check=True)
