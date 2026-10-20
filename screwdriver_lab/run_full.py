"""Reproducible paired full-task benchmark; CPU dynamics, no GPU allocation."""
import concurrent.futures,json,os,subprocess,sys
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.abstractions import METHODS
CASES=[dict(id='nominal',name='Nominal'),dict(id='offset',name='Pickup +1.5 mm',pickup_dy=.0015),dict(id='roll',name='Tool roll +20°',tool_roll_deg=20),dict(id='friction',name='Contact μ = 0.55',friction=.55),dict(id='resistance',name='2× thread resistance',thread_torque_Nm=.0016)]
def execute(item):
 c,method=item;name=c['id']+'-'+method
 if (OUT/(name+'.json')).exists():return name+' existing'
 cmd=[sys.executable,'-m','screwdriver_lab.full_motion','--arm','--mode','regrasp','--timeout','190','--name',name,'--abstraction',method,'--case-json',json.dumps(c)]
 with (OUT/(name+'.log')).open('w') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1'})
 if result.returncode:raise RuntimeError(name+' failed; inspect log')
 r=json.loads((OUT/(name+'.json')).read_text());return f"{name}: {r['outcome']}, {r['final']['depth_mm']:.3f} mm, {r['final']['t']:.1f} s"
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in pool.map(execute,[(c,m)for c in CASES for m in METHODS]):print(result,flush=True)
