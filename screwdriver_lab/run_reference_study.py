"""Matched corrected-role drive comparison; no pickup or full-task success claim."""
import json,concurrent.futures,numpy as np
from screwdriver_lab.reference_control import run,OUT
CASES=[{'id':'nominal'},{'id':'friction','friction':.55},{'id':'resistance','thread_torque_Nm':.0016}]
METHODS={'point':('rolling',0.),'pose':('rolling',.012),'wrench':('velocity',0.),'hold':('hold',0.)}
def one(item):
 case,method=item;mode,weight=METHODS[method];name='reference-study-'+case['id']+'-'+method;r,q,xml=run(20,mode,pad_weight=weight,case=case);r['id']=name;r['interface']=method
 (OUT/(name+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(name+'.npz'),qpos=q);(OUT/(name+'.xml')).write_text(xml);return dict(id=name,outcome=r['outcome'],depth_mm=r['final']['depth_mm'],t=r['final']['t'])
if __name__=='__main__':
 items=[(c,m)for c in CASES for m in ['point','pose','wrench']]+[(CASES[0],'hold')]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3)as pool:
  for r in pool.map(one,items):print('STUDY_RESULT',json.dumps(r),flush=True)
