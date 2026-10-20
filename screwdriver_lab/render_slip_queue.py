"""Sequential RTX work, restricted to physical GPU 1."""
import os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
# Let the already-running inspection finish before allocating GPU memory.
while True:
 try:os.kill(2677515,0);time.sleep(10)
 except ProcessLookupError:break
jobs=[('slip-study-wrench-nominal',6),('slip-study-pose-nominal',6),('slip-study-velocity-nominal',6),('slip-study-wrench-tilt-minus',12),('slip-study-wrench-low-friction',12),('slip-verified-seating',30)]
for name,scale in jobs:
 while not (ROOT/'screwdriver_lab/artifacts/full_motion'/f'{name}.json').exists():time.sleep(10)
 for cmd in [['-m','screwdriver_lab.export_slip_video',name,'--scale',str(scale)],['-m','screwdriver_lab.render_slip_study',name,'--samples','12']]:
  print('RUN',name,cmd,flush=True);subprocess.run([os.environ.get('DEXAGENT_PHYSICS_PYTHON','/home/uyoo/miniconda3/envs/newton/bin/python'),*cmd],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1'},check=True)
 print('DONE',name,flush=True)
