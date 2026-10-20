"""RTX input built exclusively from measured cube rollout poses."""
import argparse,json,numpy as np
from .model import OUT
from screwdriver_lab.export_rtx import export

def bundle(name):
 r=json.loads((OUT/(name+'.json')).read_text());a=np.load(OUT/(name+'.npz'));frames=[]
 for c in r['commands'][::2]:
  frames.append(dict(**c,handle_forces_N=[c['loads_N'][0],c['loads_N'][2]],display_scale=1))
 return export((OUT/(name+'.xml')).read_text(),a['qpos'][::2],OUT/'rtx'/name,dict(method=r['interface']+' / hydroelastic',interface=r['interface'],case='cube rotation',goal_deg=r['goal_deg'],goal_axis=r.get('goal_axis','z'),prepared_start=bool(r.get('checkpoint')),video_time_scale=1,outcome=r['outcome'],frames=frames))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();print(bundle(a.name))
