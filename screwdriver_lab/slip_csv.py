"""Flat metrics export; benchmark targets stay distinct from seating success."""
import csv,json
from pathlib import Path
from screwdriver_lab.full_scene import OUT,ROOT
names=[p.stem for p in sorted(OUT.glob('slip-study-*.json'))]+['slip-verified-seating','slip-hold-control','slip-no-unload-control','slip-hydro-material','slip-return-load-9','slip-return-load-10','slip-return-load-11','slip-velocity-integral','slip-speed-36','slip-speed-60','slip-progress-graph','slip-fast-light-contact','slip-full-fixed-tripod'];rows=[]
for n in names:
 r=json.loads((OUT/(n+'.json')).read_text());s=r['metrics']['controlled_slip'];row=dict(id=n,protocol_group='matched_60s' if n.startswith('slip-study-')else 'baseline_seating' if n=='slip-verified-seating'else 'legacy_hydro_pilot'if n=='slip-hydro-material'else 'full_integration_attempt'if n=='slip-full-fixed-tripod'else 'nominal_development',backend=r['parameters'].get('backend','mujoco'),method=r['method'],condition=r['case'],scope=r['scope'],outcome=r['outcome'],seat_success=r['success'],skill_90deg_success=r.get('skill_success',''),strict_slip_check=r.get('controlled_slip_verified',''),simulation_s=r['final']['t'],rotation_deg=r['final']['screw_angle_deg'],depth_mm=r['final']['depth_mm'],cycles=r['metrics']['cycles'],reverse_rotation_deg=s['backward_rotation_deg'],peak_finger_N=r['metrics']['peak_finger_N'],peak_penetration_mm=r['metrics']['max_penetration_mm'])
 for i,finger in enumerate(['thumb','middle']):
  for k,label,scale in [('return_loaded_fraction','return_contact_pct',100),('return_loaded_sliding_fraction','return_loaded_sliding_pct',100),('loaded_return_sliding_distance_mm','loaded_reverse_slip_mm',1),('longest_contact_dropout_s','longest_dropout_ms',1000)]:row[finger+'_'+label]=s[k][i]*scale
 rows.append(row)
path=ROOT/'screwdriver_site/dist/data/slip-summary.csv'
with path.open('w')as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('CSV records',len(rows))
