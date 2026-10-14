import json
from pathlib import Path
root=Path(__file__).resolve().parents[2];run=root/'runs/physics_revision_20261012'
for p in sorted(run.glob('*_result.json')):
 r=json.loads(p.read_text());f=r['final']
 print(p.stem, 'PASS' if r.get('success') else 'FAIL/DIAGNOSTIC','angle',round((f.get('angle_error_deg') or 0),2),'pos_mm',round(1000*(f.get('position_error_m') or 0),2),'stop',r.get('violation'))
