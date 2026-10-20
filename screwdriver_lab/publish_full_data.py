"""Build the website manifest from measured runs and verified render outputs."""
import json,shutil,hashlib
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.scene import ROOT
from screwdriver_lab.analyze_full import enrich
SITE=ROOT/'screwdriver_site/dist'
KEYS=['t','phase','depth_mm','screw_angle_deg','axial_force_N','tilt_deg','bit_lateral_mm','seat_force_N','engaged','finger_forces','slip_mm_s','finger_modes']
def small(r):
 r=dict(r);r.pop('commands',None);r.pop('trace',None);fs=r['frames'];keep=[];next_t=0
 for f in fs:
  if f['t']>=next_t-1e-6:keep.append({k:f[k]for k in KEYS});next_t=f['t']+.19
 if keep[-1]['t']!=fs[-1]['t']:keep.append({k:fs[-1][k]for k in KEYS})
 r['frames']=keep;return r

def add_media(r):
 name=r['id'];meta=OUT/'rtx'/(name+'.render.json');video=meta.with_suffix('').with_suffix('.mp4')
 if not meta.exists():return r
 render=json.loads(meta.read_text());assert render['RTX'] and all('61:00' in d['id']for d in render['devices'])
 render['display_changes']='PBR materials, shading normals, studio lighting, cameras, floor and a display-only pedestal under the fixed arm base. Robot/tool/workpiece vertices and recorded poses are unchanged.'
 assert render['renderer_sha256']==hashlib.sha256((ROOT/'screwdriver_lab/render_full.py').read_bytes()).hexdigest()
 scene=json.loads((OUT/'rtx'/name/'scene.json').read_text());r['video_time_scale']=scene['metadata']['video_time_scale'];r['rendering']=render
 for ext,key in [('.mp4','video'),('.jpg','poster'),('.render.json','render_metadata')]:
  source=OUT/'rtx'/(name+ext);target=SITE/'assets'/('full-'+name+ext);shutil.copy2(source,target);r[key]='assets/'+target.name
  if ext=='.render.json':target.write_text(json.dumps(render,indent=2))
 return r

def build():
 data=json.loads((OUT/'manifest.json').read_text());data['calibration']=json.loads((OUT/'abstraction-bank.json').read_text());data['results']=[add_media(small(r))for r in data['results']];data['contact_models']=[add_media(small(r))for r in data['contact_models']]
 successes=[r for r in data['results']if r['success']]
 counts={k:sum(r['success'] for r in data['results']if r['method']==k)for k in data['methods']}
 data['summary']=f"{len(successes)} of {len(data['results'])} runs meet the full seating criterion. By interface: "+'; '.join(data['methods'][k]['name']+f' {v}/5'for k,v in counts.items())+'. The nominal calibration favors replay in-distribution. Perturbations expose pickup, grasp-transfer, and driving sensitivity; each cell is one run, not a success-rate estimate.'
 data['contact_model_summary']='All four drive-only trials use a 15 s budget and the same aligned initial grasp. Loaded return slip is the fraction of return-mode finger samples with ≥0.02 N handle load and >2 mm/s normal-force-weighted handle slip. In the hydroelastic sliding trial, loaded return slip coexists with almost no net insertion: contact patches do not supply a successful sliding policy. Finite-patch and hydroelastic material parameters are not fitted to hardware. Hydroelastic deformation and native contact penetration are model-dependent quantities.'
 data['rendering']={'GPU':1,'renderer':'Blender Cycles / NVIDIA OptiX','physics_poses_modified':False}
 data['rank_sensitivity']=[]
 for rank,name in [(3,'nominal-synergy'),(6,'rank6-synergy'),(10,'rank10-synergy')]:
  path=OUT/(name+'.json')
  if path.exists():
   r=enrich(json.loads(path.read_text()));r['rank']=rank;r['variance_retained']=data['calibration']['variance_by_rank'][str(rank)];data['rank_sensitivity'].append(small(r))
 probe=OUT/'hydro-full-probe.json'
 if probe.exists():
  r=enrich(json.loads(probe.read_text()));data['hydro_full_probe']=small(r);data['hydro_full_probe_summary']=f"Additional full-task hydroelastic probe: {r['outcome'].replace('_',' ')} in {r['final']['stage']} at {r['final']['t']:.2f} s, depth {r['final']['depth_mm']:.4f} mm, peak finger load {r['metrics']['peak_finger_N']:.1f} N. The controller loses the grasp during lift and stops after a force overshoot. This is separate from the 25-run finite-patch sweep and does not establish full-task hydroelastic success."
 (SITE/'data/full-results.json').write_text(json.dumps(data,separators=(',',':')));shutil.copy2(OUT/'validation.json',SITE/'data/full-validation.json');shutil.copy2(ROOT/'screwdriver_lab/graph/workflow.json',SITE/'data/workflow.json')
 print(data['summary']);print('RTX clips',sum('video'in r for r in data['results'])+sum('video'in r for r in data['contact_models']))
if __name__=='__main__':build()
