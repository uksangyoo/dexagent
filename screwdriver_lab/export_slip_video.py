"""Sample exact recorded poses at explicit simulation-time spacing."""
import argparse,json,numpy as np
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.export_rtx import export
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--scale',type=float,default=6);p.add_argument('--fps',type=int,default=15);a=p.parse_args()
r=json.loads((OUT/(a.name+'.json')).read_text());q=np.load(OUT/(a.name+'.npz'))['qpos'];ts=np.array([f['t']for f in r['frames']]);ix=np.array([int(np.argmin(abs(ts-t)))for t in np.arange(0,ts[-1]+1e-6,a.scale/a.fps)])
if ix[-1]!=len(ts)-1:ix=np.r_[ix,len(ts)-1]
r['frames']=[r['frames'][i]for i in ix];r.pop('commands',None);r.pop('trace',None);r['video_time_scale']=a.scale;r['video_fps']=a.fps
export((OUT/(a.name+'.xml')).read_text(),q[ix],OUT/'rtx'/a.name,r)
