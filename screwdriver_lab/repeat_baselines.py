"""Re-evaluate records that could be affected by extending trajectory replay."""
import concurrent.futures,json,shutil
from screwdriver_lab.run_full import CASES,METHODS,execute,OUT
archive=OUT/'iterations'/'before_cycle_repeat';archive.mkdir(parents=True,exist_ok=True)
items=[]
for c in CASES:
 if c['id'] not in ['roll','friction']:continue
 for method in METHODS:
  if method=='contact':continue
  name=c['id']+'-'+method;path=OUT/(name+'.json')
  if not path.exists():continue
  r=json.loads(path.read_text())
  if not any(f['phase']=='turn' and f['t']>146.4 for f in r['frames']):continue
  for ext in ['.json','.npz','.xml','.log']:
   source=OUT/(name+ext)
   if source.exists():shutil.move(source,archive/source.name)
  items.append((c,method))
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
 for result in pool.map(execute,items):print(result,flush=True)
