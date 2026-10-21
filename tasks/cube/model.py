"""Free rigid cube, real Tesollo geometry, collision-free fixture pickup.

Only the time-zero robot configuration uses IK. The cube has a declared 20 mm
edge and density 1000 kg/m^3, not calibrated hardware material parameters.
"""
from pathlib import Path
import json, xml.etree.ElementTree as E
import numpy as np, mujoco
from scipy.optimize import least_squares
from dexagent.planning.initialization import initial_state
from dexagent.planning.geometry import ART

OUT=ART.parent/'cube_revision'

def build(name='cube',yaw=0,balanced=False,edge=.02,pulp=False,initial_from=None):
 OUT.mkdir(exist_ok=True)
 if initial_from:
  # Reuse a declared collision-free time-zero scene, never a held checkpoint.
  import shutil,hashlib
  from scipy.spatial.transform import Rotation
  src=OUT/(initial_from+'-initial.npz');report=json.loads((OUT/(initial_from+'-initial.json')).read_text())
  if report['edge_m']!=edge or report['pulp_contact']!=pulp or report['balanced_pregrasp']!=balanced:raise ValueError('Initial scene contract mismatch')
  xml=OUT/(name+'.xml');shutil.copy2(OUT/(initial_from+'.xml'),xml)
  m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=np.load(src)['qpos'][0]
  a=m.joint('cube_free').qposadr[0];d.qpos[a+3:a+7]=Rotation.from_euler('z',yaw,degrees=True).as_quat(scalar_first=True);mujoco.mj_forward(m,d)
  bad=[dict(pair=[m.geom(c.geom1).name,m.geom(c.geom2).name],mm=-c.dist*1000)for c in d.contact if c.dist< -1e-6]
  report.update(initial_penetrations=bad,initial_yaw_deg=yaw,initial_state_source=initial_from,initial_state_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
  (OUT/(name+'-initial.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/(name+'-initial.npz'),qpos=d.qpos[None,:])
  if bad:raise RuntimeError('Reject penetrating perturbed initial scene')
  return xml,d.qpos.copy(),report['contacts']
 source,q0,_=initial_state(ART/'precision-slotted.xml',True,output_stem=name+'-seed')
 old=mujoco.MjModel.from_xml_path(str(source));root=E.parse(source).getroot();world=root.find('worldbody')
 for b in list(world.findall('body')):
  if b.get('name')in ['screwdriver','screw','female_thread']:world.remove(b)
 for g in list(world.findall('geom')):
  if g.get('name','').startswith(('pickup_','workpiece_','cradle')):world.remove(g)
 for section in ['sensor','keyframe','equality']:
  item=root.find(section)
  if item is not None:root.remove(item)
 used={g.get('mesh')for g in root.iter('geom')}
 for a in list(root.find('asset')):
  if a.tag=='mesh'and a.get('name')not in used and a.get('name')is not None:root.find('asset').remove(a)
 # The cube rests on a small physical pedestal, with all fingers initially clear.
 E.SubElement(world,'geom',name='cube_pedestal',type='box',pos=f'-.22 .1 {(0.43-edge/2+.11)/2}',size=f'.008 .008 {(0.43-edge/2-.11)/2}',rgba='.12 .28 .31 1')
 body=E.SubElement(world,'body',name='cube',pos='-.22 .1 .43')
 E.SubElement(body,'freejoint',name='cube_free')
 E.SubElement(body,'geom',name='cube',type='box',size=' '.join([str(edge/2)]*3),mass=str(edge**3*(400 if edge>.02 else 1000)),rgba='.8 .2 .08 1',contype='1',conaffinity='1')
 xml=OUT/(name+'.xml');E.indent(root);xml.write_text(E.tostring(root,encoding='unicode'))
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
 for j in range(m.njnt):
  if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:d.qpos[m.jnt_qposadr[j]]=q0[old.joint(m.joint(j).name).qposadr[0]]
 mujoco.mj_forward(m,d);center=d.xpos[m.body('cube').id].copy();records=[]
 for i,n,local in [(1,[0,-1,0],[.0466,-.0029,.0016]),(2,[0,0,1],[.0357,-.0029,.0016]),(3,[0,1,0],[.0357,-.0029,.0016])]:
  local=np.array(([.0289+.00588841464,.00746608745,0]if i==1 else[.018+.00588841464,.00746608745,0])if pulp else local);pad_normal=np.array([.187702240,.982222897,-.00245951835])if pulp else np.array([1,0,0]);n=np.array([1,0,0]if balanced and i==2 and edge==.02 else n,float);b=m.body(f'link_{i}_4').id
  target=center+n*(edge/2+(.003 if i!=2 else (.002 if balanced else .01)))
  ids=np.array([m.joint(f'joint_{i}_{j}').id for j in range(1,5)]);qi=m.jnt_qposadr[ids];seed=d.qpos[qi].copy()
  if pulp:seed[3]-=1.4
  def error(q):
   d.qpos[qi]=q;mujoco.mj_kinematics(m,d);R=d.xmat[b].reshape(3,3)
   return np.r_[1000*(d.xpos[b]+R@local-target),(15 if pulp else 3)*(R@pad_normal+n),.01*(q-seed)]
  opt=least_squares(error,np.clip(seed,m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4),bounds=(m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4),max_nfev=500)
  e=error(opt.x);records.append(dict(finger=i,local=local.tolist(),normal=n.tolist(),target=target.tolist(),fit_mm=float(np.linalg.norm(e[:3]))))
 if balanced:
  # Jointly choose a time-zero arm/pregrasp pose with finger travel remaining.
  # This is pregrasp IK only; no object pose is changed and no contact is disabled.
  from scipy.spatial.transform import Rotation
  joint_ids=[m.joint(f'joint_{i}_{j}').id for i in [1,2,3,4,5]for j in range(1,5)]+[m.joint(n).id for n in ['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']]
  qi=m.jnt_qposadr[joint_ids];seed=d.qpos[qi].copy();palm=m.body('palm').id;Rp=d.xmat[palm].reshape(3,3).copy();pp=d.xpos[palm].copy()
  lo=np.r_[m.jnt_range[joint_ids[:20],0]+np.radians(3),seed[20:]-.8];hi=np.r_[m.jnt_range[joint_ids[:20],1]-np.radians(3),seed[20:]+.8]
  robot_geoms=[g for g in range(m.ngeom)if m.geom(g).name.startswith(('finger_tip_','source_geometry_','arm_collision_'))];original_margin=m.geom_margin.copy();m.geom_margin[robot_geoms]=.0005
  def joint_error(q):
   d.qpos[qi]=q;mujoco.mj_forward(m,d);e=[];pen=np.zeros(m.ngeom)
   for c in d.contact:
    if c.geom1 in robot_geoms or c.geom2 in robot_geoms:
     for g in [c.geom1,c.geom2]:pen[g]=max(pen[g],max(0,.0003-c.dist)*1000)
   for rec in records:
    b=m.body(f'link_{rec["finger"]}_4').id;R=d.xmat[b].reshape(3,3);point=d.xpos[b]+R@np.array(rec['local'])
    e.extend(1000*(point-np.array(rec['target'])));e.extend((15 if pulp else 3)*(R@pad_normal+np.array(rec['normal'])))
   return np.r_[e,10*pen,.1*(q[:20]-(lo[:20]+hi[:20])/2),.1*(q[20:]-seed[20:]),5*(d.xpos[palm]-pp),.5*Rotation.from_matrix(d.xmat[palm].reshape(3,3)@Rp.T).as_rotvec()]
  rng=np.random.default_rng(1419);fit=None
  for attempt in range(3):
   start=seed if attempt==0 else seed+rng.normal(0,.25,len(seed))
   candidate=least_squares(joint_error,np.clip(start,lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=400,diff_step=1e-5)
   if fit is None or candidate.cost<fit.cost:fit=candidate
   print('PREGRASP_IK',attempt,float(candidate.cost),flush=True)
   if fit.cost<1:break
  joint_error(fit.x);m.geom_margin[:]=original_margin
  for rec in records:
   b=m.body(f'link_{rec["finger"]}_4').id;point=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array(rec['local']);rec['fit_mm']=float(1000*np.linalg.norm(point-rec['target']))
 if yaw:
  from scipy.spatial.transform import Rotation
  qa=m.joint('cube_free').qposadr[0];d.qpos[qa+3:qa+7]=Rotation.from_euler('z',yaw,degrees=True).as_quat(scalar_first=True)
 mujoco.mj_forward(m,d)
 bad=[dict(pair=[m.geom(c.geom1).name,m.geom(c.geom2).name],mm=-c.dist*1000)for c in d.contact if c.dist< -1e-6]
 report=dict(initial_penetrations=bad,contacts=records,edge_m=edge,mass_kg=edge**3*(400 if edge>.02 else 1000),initial_yaw_deg=yaw,robot_refit_for_perturbation=False,balanced_pregrasp=balanced,pulp_contact=pulp)
 (OUT/(name+'-initial.json')).write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/(name+'-initial.npz'),qpos=d.qpos[None,:])
 print('INITIAL',json.dumps(report),flush=True)
 if bad:raise RuntimeError('Reject initial penetration')
 if balanced and max(r['fit_mm']for r in records)>2.5:raise RuntimeError('Reject inaccurate pregrasp IK')
 return xml,d.qpos.copy(),records

if __name__=='__main__':build()
