"""Matched low-level interfaces, common graph, geometry and initial conditions."""
import argparse,json,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from screwdriver_lab.controlled_slip import OUT,ROOT,run
BASE=dict(timeout=60,amplitude=24,speed=12,drive_load=.4,slide_load=.08,force_p=0,force_i=0,normal_d=5,orientation_gain=0,posture_gain=0,pad_time=.015,torsion=.0005,thread_torque=.003,native_ccd=0,grasp='screwdriver_lab/artifacts/controlled_slip_tip')
CASES={'nominal':{},'low-friction':{'friction':.6},'tilt-plus':{'initial_tilt':2},'tilt-minus':{'initial_tilt':-2},'high-resistance':{'thread_torque':.004}}
METHODS={'pose':dict(servo='admittance',gain=.004,pad_weight=.012,surface_tracking=0),'velocity':dict(servo='paired',velocity_gain=100),'wrench':dict(servo='paired')}
def trial(method,case):
 name=f'slip-study-{method}-{case}';r=run(name,**{**BASE,**METHODS[method],**CASES[case]});r['method']=method;r['case']=case;r['scope']='60-second matched controlled-slip skill benchmark';m=r['metrics']['controlled_slip'];r['skill_success']=r['final']['screw_angle_deg']>=90 and r['outcome']=='timeout' and r['final']['engaged'];r['controlled_slip_verified']=r['skill_success']and min(m['return_loaded_fraction'])>=.95 and min(m['loaded_return_sliding_distance_mm'])>1 and m['backward_rotation_deg']<5;r['parameters']['evaluation_target_deg']=90;(OUT/(name+'.json')).write_text(json.dumps(r));print('EVALUATED',name,r['skill_success'],r['controlled_slip_verified'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);p.add_argument('--case',choices=CASES);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.method:trial(a.method,a.case)
 else:
  def launch(pair):
   method,case=pair;log=ROOT/'screwdriver_lab/artifacts/reference_strategy'/f'slip-study-{method}-{case}.log'
   with log.open('w')as f:subprocess.run([os.environ.get('DEXAGENT_PHYSICS_PYTHON','/home/uyoo/miniconda3/envs/newton/bin/python'),'-m','screwdriver_lab.slip_compare','--method',method,'--case',case],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
   return method,case
  with ThreadPoolExecutor(max_workers=a.workers)as pool:
   for pair in pool.map(launch,[(m,c)for m in METHODS for c in CASES]):print('COMPLETE',*pair,flush=True)
