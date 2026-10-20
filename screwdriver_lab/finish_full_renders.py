"""Export the validated final dataset, then render without exceeding three workers."""
import json,os,subprocess,sys,time,psutil
from screwdriver_lab.full_scene import OUT
while not (OUT/'manifest.json').exists():time.sleep(5)
data=json.loads((OUT/'manifest.json').read_text());full=[r['id']for r in data['results']];drive=[r['id']for r in data['contact_models']]
for ids,scale in [(full[1:],12),(drive,3)]:
 subprocess.run([sys.executable,'-m','screwdriver_lab.export_full',*ids,'--scale',str(scale)],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
# The nominal contact export remains at 6×. All other clips use their declared scale.
while any('-m'in(p.info['cmdline']or[])and'screwdriver_lab.render_full_batch'in(p.info['cmdline']or[])for p in psutil.process_iter(['cmdline'])):time.sleep(5)
subprocess.run([sys.executable,'-m','screwdriver_lab.render_full_batch',*full,*drive,'--workers','3'],check=True)
print('ALL_FULL_RTX_DONE',flush=True)
