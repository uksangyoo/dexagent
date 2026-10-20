"""Nominal-only speed/stroke sensitivity; not part of the matched benchmark."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from screwdriver_lab.slip_compare import BASE,METHODS,ROOT
from screwdriver_lab.controlled_slip import run
if len(sys.argv)>1:
 speed=float(sys.argv[1]);amplitude=30 if speed==36 else 36;run('slip-speed-'+str(round(speed)),**{**BASE,**METHODS['wrench'],'speed':speed,'amplitude':amplitude})
else:
 def trial(speed):
  with (ROOT/'screwdriver_lab/artifacts/reference_strategy'/f'slip-speed-{speed}.log').open('w')as f:subprocess.run([sys.executable,'-m','screwdriver_lab.slip_speed_sweep',str(speed)],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
 with ThreadPoolExecutor(2)as pool:list(pool.map(trial,[36,60]))
