"""Bound OptiX memory by releasing the renderer between 60-frame segments."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
from screwdriver_lab.full_scene import OUT,ROOT
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--chunk',type=int,default=60);p.add_argument('--samples',type=int,default=24);p.add_argument('--fps',type=int,default=15);a=p.parse_args()
bundle=OUT/'rtx'/a.name;output=OUT/'rtx'/(a.name+'-contact.mp4');parts=OUT/'rtx'/(a.name+'-chunks');parts.mkdir(exist_ok=True);n=len(np.load(bundle/'scene.npz')['transforms']);metas=[];files=[]
for start in range(0,n,a.chunk):
 stop=min(n,start+a.chunk);dest=parts/f'{start:05d}.mp4';log=parts/f'{start:05d}.log';cmd=[os.environ.get('DEXAGENT_RENDER_PYTHON','/home/uyoo/miniconda3/envs/blender-render/bin/python'),'-m','screwdriver_lab.render_controlled_slip','--bundle',str(bundle),'--output',str(dest),'--start',str(start),'--stop',str(stop),'--samples',str(a.samples),'--fps',str(a.fps)]
 if stop<n:cmd+=['--no-end-card']
 with log.open('w')as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 meta=json.loads(dest.with_suffix('.render.json').read_text());assert all('61:00'in x['id']for x in meta['devices']);metas.append(meta);files.append(dest);print('CHUNK',start,stop,n,flush=True)
listing=parts/'concat.txt';listing.write_text(''.join("file '"+str(f)+"'\n"for f in files));subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(output)],check=True)
meta=dict(metas[0]);meta.update(frames=sum(x['frames']for x in metas),end_card_s=1,frame_start=0,frame_stop=n,chunks=metas,source_bundle_sha256=hashlib.sha256((bundle/'scene.npz').read_bytes()).hexdigest(),elapsed_s=sum(x['elapsed_s']for x in metas));output.with_suffix('.render.json').write_text(json.dumps(meta,indent=2));output.with_suffix('.jpg').write_bytes(files[0].with_suffix('.jpg').read_bytes());print('COMPLETE',output,flush=True)
