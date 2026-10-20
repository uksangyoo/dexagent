"""Record 1 kHz contact signals and prove instrumentation leaves poses unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
import screwdriver_lab.slip_metrics as metrics
from screwdriver_lab.slip_compare import BASE,METHODS
from screwdriver_lab.controlled_slip import run,OUT,ROOT
Original=metrics.SlipMetrics
class AuditMetrics(Original):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.rows=[];self.mode_codes=[]
 def add(self,o,modes):
  self.rows.append([o['t'],*o['handle_forces_N'],*o['handle_slip_signed_mm_s'],o['screw_angle_deg']]);self.mode_codes.append(2 if modes[0]=='slide_return'else 1 if modes[0]=='drive'else 0);super().add(o,modes)
 def report(self):
  a=np.array(self.rows);np.savez_compressed(OUT/'slip-contact-1khz.npz',time_s=a[:,0],pad_load_N=a[:,1:3].astype('float32'),signed_slip_mm_s=a[:,3:5].astype('float32'),screw_angle_deg=a[:,5],mode=np.array(self.mode_codes,dtype='uint8'));return super().report()
if __name__=='__main__':
 metrics.SlipMetrics=AuditMetrics;r=run('slip-contact-audit',**{**BASE,**METHODS['wrench']});base=json.loads((OUT/'slip-study-wrench-nominal.json').read_text());a=np.load(OUT/'slip-contact-audit.npz')['qpos'];b=np.load(OUT/'slip-study-wrench-nominal.npz')['qpos'];assert np.array_equal(a,b);assert r['metrics']==base['metrics']
 with np.load(OUT/'slip-contact-1khz.npz')as z:
  active=(z['time_s']>=1.5)&(z['mode']==2);loads=z['pad_load_N']>.01;fractions=loads[active].mean(axis=0);distance=np.maximum(0,-z['signed_slip_mm_s'][active]).astype(float)*loads[active]*.001;assert np.allclose(fractions,base['metrics']['controlled_slip']['return_loaded_fraction'],atol=1e-8);assert np.allclose(distance.sum(axis=0),base['metrics']['controlled_slip']['loaded_return_sliding_distance_mm'],atol=1e-5)
 result=dict(passed=True,reference_id=base['id'],samples=0,identical_qpos=True,identical_1khz_aggregate_metrics=True,contact_threshold_N=.01,signal_file='slip-contact-1khz.npz',mode_codes={'other':0,'drive':1,'unload_or_slide_return':2},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with np.load(OUT/'slip-contact-1khz.npz')as z:result['samples']=len(z['time_s'])
 (OUT/'slip-contact-audit-validation.json').write_text(json.dumps(result,indent=2));print(result)
