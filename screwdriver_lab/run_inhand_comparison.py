"""Run independent CPU simulations; any GPU work remains restricted to GPU 1."""
import subprocess,os,sys
from concurrent.futures import ThreadPoolExecutor
from screwdriver_lab.reference_control import REF
jobs=[(m,c,y)for m in ['wrench','pose','velocity']for c,y in [('nominal',0),('offset-minus',-2),('offset-plus',2)]]
def run(job):
 m,c,y=job;name=f'pivot-study-{m}-{c}'
 with (REF/(name+'.log')).open('w')as log:
  r=subprocess.run([sys.executable,'-m','screwdriver_lab.inhand_policy','--name',name,'--interface',m,'--initial-y-mm',str(y),'--timeout','100'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True);return r.returncode
with ThreadPoolExecutor(max_workers=3)as pool:
 codes=list(pool.map(run,jobs))
assert not any(codes),codes
