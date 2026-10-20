"""Development sensitivity: positive reset load versus continuity and progress."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from screwdriver_lab.slip_compare import BASE,METHODS,ROOT
from screwdriver_lab.controlled_slip import run
if len(sys.argv)>1:
 load=float(sys.argv[1]);run('slip-return-load-'+str(round(load*100)),**{**BASE,**METHODS['wrench'],'slide_load':load})
else:
 def trial(load):
  with (ROOT/'screwdriver_lab/artifacts/reference_strategy'/f'slip-return-load-{round(load*100)}.log').open('w')as f:subprocess.run([sys.executable,'-m','screwdriver_lab.slip_load_sweep',str(load)],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
 with ThreadPoolExecutor(3)as pool:list(pool.map(trial,[.09,.10,.11]))
