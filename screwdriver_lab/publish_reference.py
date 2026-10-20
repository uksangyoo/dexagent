"""Publish a candid reference-strategy audit, corrected-role traces and validation."""
import argparse,json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.reference_control import REF,ROOT
SITE=ROOT/'screwdriver_site/dist';ASSETS=SITE/'assets';DATA=SITE/'data'
METHODS={'point':{'name':'Contact point targets','contract':'3D contact points → joint IK','detail':'Track two orbiting contact points. Index maintains cap pressure. The solver does not constrain pad orientation.'},'pose':{'name':'Point + pad normal','contract':'Contact points + pad normals → joint IK','detail':'Add pad-normal tracking to the same point targets. Four finger joints approximately satisfy the combined position/orientation objective; normal weight is 0.012 m.'},'wrench':{'name':'Force + tangent velocity','contract':'Normal load + tangent speed → joint torque targets','detail':'Use the live closest contact point, normal-force feedback and tangential velocity feedback. Joint torques are expressed through the same bounded position actuators. This implementation can back-drive and lose engagement.'}}
CASES=[{'id':'nominal','label':'Nominal · μ 0.9 · 0.8 mN·m'},{'id':'friction','label':'Lower contact friction · μ 0.55'},{'id':'resistance','label':'Double thread resistance · 1.6 mN·m'}]
def enrich(r):
 f=[x for x in r['frames']if x['phase']in ['turn','rolling','drive','compact_return','return','unload','reacquire']]
 if not f:f=r['frames']
 a=np.array([x['screw_angle_deg']for x in f]);returns=[]
 for x in f:
  for i,mode in enumerate(x.get('finger_modes',[])):
   if mode in ['light_return','slide_return','return','compact_return']:returns.append((x['handle_forces_N'][i],x['slip_mm_s'][i]))
 r['metrics'].update(index_cap_contact_fraction=float(np.mean([x['index_cap_load_N']>.1 for x in f])),forward_turn_deg=float(np.maximum(np.diff(a),0).sum()),backward_turn_deg=float(-np.minimum(np.diff(a),0).sum()),loaded_return_slip_fraction=float(np.mean([load>.02 and slip>2 for load,slip in returns]))if returns else None,mean_return_load_N=float(np.mean([x[0]for x in returns]))if returns else None)
 return r

def main():
 records=[enrich(json.loads((OUT/f'reference-study-{c["id"]}-{m}.json').read_text()))for c in CASES for m in METHODS]
 full=enrich(json.loads((OUT/'reference-full-audited.json').read_text()));hold=enrich(json.loads((OUT/'reference-study-nominal-hold.json').read_text()));recovery=enrich(json.loads((OUT/'reference-full-recovery.json').read_text()))
 hashes={r['initial_state_sha256']for r in records+[hold]};assert len(hashes)==1,hashes
 for r in records+[hold,full]:
  m=mujoco.MjModel.from_xml_string((OUT/(r['id']+'.xml')).read_text());assert m.nu==26;assert m.jnt_type[m.joint('tool_free').id]==mujoco.mjtJoint.mjJNT_FREE
  assert all(m.joint(int(j)).name.startswith(('joint_','shoulder','elbow','wrist'))for j in m.actuator_trnid[:,0])
 for r in records+[hold,full]:
  assert all(hashlib.sha256((ROOT/'screwdriver_lab'/name).read_bytes()).hexdigest()==digest for name,digest in r['source_sha256'].items()),r['id']
 assert min(full['initial_hand_clearance_mm'].values())>1
 transitions=full['graph_trace'];sequence=[x['from_node']for x in transitions if x['exit']=='complete'];assert sequence[:6]==['approach','grasp','lift','reorient','cap_support','align'],sequence
 assert not full['success'];assert full['outcome']in ['side_contact_lost','axis_drift']
 # Actual arm excursion during turning, reconstructed from recorded articulated poses.
 m=mujoco.MjModel.from_xml_string((OUT/(full['id']+'.xml')).read_text());d=mujoco.MjData(m);q=np.load(OUT/(full['id']+'.npz'))['qpos'];palm=m.body('palm').id;poses=[]
 for i,f in enumerate(full['frames']):
  if f['phase']=='turn':d.qpos[:]=q[i];mujoco.mj_forward(m,d);poses.append((d.xpos[palm].copy(),d.xmat[palm].reshape(3,3).copy()))
 full['metrics']['arm_translation_from_turn_start_mm']=float(max(np.linalg.norm(p-poses[0][0])*1000 for p,R in poses));full['metrics']['arm_rotation_from_turn_start_deg']=float(max(np.degrees(np.linalg.norm(Rot.from_matrix(R@poses[0][1].T).as_rotvec()))for p,R in poses))
 d.qpos[:]=q[-1];mujoco.mj_forward(m,d)
 full['metrics']['terminal_side_pad_gaps_mm']=[float(mujoco.mj_geomDistance(m,d,m.geom(f'finger_tip_{i}').id,m.geom('handle').id,.1,None)*1000)for i in [1,3]]
 data=dict(full=full,results=records,hold=hold,recovery=recovery,methods=METHODS,cases=CASES,observation=json.loads((REF/'observation.json').read_text()),workflow=json.loads((REF/'workflow.json').read_text()),summary='Corrected finger roles and task order; full replication is not yet achieved. The full rollout advances the screw, then stops on loss of side contact. No corrected-role trial verifies complete seating.')
 media=[]
 for r in records+[full]:
  name=r['id'];base=OUT/'rtx'/name
  if base.with_suffix('.render.json').exists():
   for suffix in ['.mp4','.jpg','.render.json']:
    src=base.with_suffix(suffix);dest=ASSETS/('ref-'+name+suffix);shutil.copyfile(src,dest);media.append(dest)
   r.update(video='assets/ref-'+name+'.mp4',poster='assets/ref-'+name+'.jpg',render_metadata='assets/ref-'+name+'.render.json',video_time_scale=1 if 'study-nominal'in name else 3)
  # The complete data remains in the trace archive; keep web observations at about 10 Hz.
  stride=3;r['frames']=r['frames'][::stride]+([r['frames'][-1]]if (len(r['frames'])-1)%stride else[]);r.pop('commands',None);r.pop('trace',None);r.pop('grasp_calibration',None)
 for r in [hold,recovery]:r.pop('commands',None);r.pop('trace',None);r.pop('frames',None);r.pop('grasp_calibration',None)
 for name in ['human-10s.png','pickup.jpg','drive-cycle.jpg','later-cycle.jpg']:
  shutil.copyfile(REF/name,ASSETS/('ref-'+name))
 validation=dict(passed=True,scope='mechanical/model and record checks; not task success',robot_actuators=26,free_tool=True,record_controller_source_hashes_match=True,paired_reset_sha256=next(iter(hashes)),full_initial_clearance_mm=full['initial_hand_clearance_mm'],executed_phase_sequence=sequence,full_outcome=full['outcome'],corrected_full_seats=0,pressure_only_20s_depth_mm=hold['final']['depth_mm'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (ROOT/'screwdriver_lab').glob('reference*.py')},media_count=len(media)//3)
 (DATA/'reference-results.json').write_text(json.dumps(data,separators=(',',':')));(DATA/'reference-validation.json').write_text(json.dumps(validation,indent=2));(REF/'validation.json').write_text(json.dumps(validation,indent=2))
 # Small code overlay: extract alongside the earlier pipeline bundle; no duplicate robot meshes.
 files=list((ROOT/'screwdriver_lab').glob('reference*.py'))+[ROOT/'screwdriver_lab/render_reference.py',ROOT/'screwdriver_lab/render_reference_batch.py',ROOT/'screwdriver_lab/run_reference_study.py',Path(__file__),REF/'grasp.json',REF/'workflow.json',REF/'observation.json',REF/'validation.json',REF/'REPRODUCE.txt',ROOT/'screwdriver_lab/check_reference_media.py',ROOT/'screwdriver_site/qa/reference_check.py',REF/'reference_cycle_timed_unload.py',REF/'reference_full_before_recovery.py']
 files.extend(SITE/x for x in ['index.html','reference.js','reference.css'])
 for r in records+[full,hold,recovery]:files.extend(OUT/(r['id']+s)for s in ['.json','.npz','.xml'])
 with zipfile.ZipFile(ASSETS/'reference-revision.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for p in files:z.write(p,p.relative_to(ROOT))
 assert (ASSETS/'reference-revision.zip').stat().st_size<16*1024**2
 print(json.dumps({'full_outcome':full['outcome'],'depth_mm':full['final']['depth_mm'],'metrics':full['metrics'],'study':[{'id':r['id'],'depth_mm':r['final']['depth_mm'],'t':r['final']['t'],'outcome':r['outcome']}for r in records],'media':len(media)//3,'overlay_MB':(ASSETS/'reference-revision.zip').stat().st_size/1048576}),flush=True)
if __name__=='__main__':main()
