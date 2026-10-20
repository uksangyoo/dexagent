"""Export exact sampled trajectories, with explicit playback time compression."""
import argparse,json,numpy as np
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.export_rtx import export
p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');p.add_argument('--scale',type=float,default=6);p.add_argument('--still',type=float);a=p.parse_args()
for name in a.ids:
 r=json.loads((OUT/(name+'.json')).read_text());poses=np.load(OUT/(name+'.npz'))['qpos'];ts=np.array([f['t']for f in r['frames']]);targets=[a.still] if a.still is not None else np.arange(0,ts[-1]+1e-6,a.scale/30);ix=np.array([int(np.argmin(abs(ts-t)))for t in targets])
 if a.still is None and abs(ts[ix[-1]]-ts[-1])>1e-8:ix=np.r_[ix,len(ts)-1]
 r['frames']=[r['frames'][i]for i in ix];r.pop('commands',None);r.pop('trace',None);r['video_time_scale']=a.scale
 dest=OUT/'rtx'/((name+f'-still{a.still:g}') if a.still is not None else name);export((OUT/(name+'.xml')).read_text(),poses[ix],dest,r);print(dest,flush=True)
