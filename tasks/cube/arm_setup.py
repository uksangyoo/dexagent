"""Offline UR5e branch selection, preserving the complete time-zero hand pose.

Recorded cube motion is used only for a clearance preflight. A selected arm
configuration is an initial condition; dynamic execution still uses motors.
"""
import argparse,json,shutil,hashlib
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tasks.cube.model import OUT
from dexagent.planning.arm_plan import NAMES

def plan(source,name):
 xml=OUT/(source+'.xml');m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
 recording=np.load(OUT/(source+'.npz'));q0=recording['initial_qpos'].copy();d.qpos[:]=q0;mujoco.mj_forward(m,d)
 palm=m.body('palm').id;ids=[m.joint(n).id for n in NAMES];qi=m.jnt_qposadr[ids];lo=m.jnt_range[ids,0]+1e-4;hi=m.jnt_range[ids,1]-1e-4
 seed=q0[qi].copy();p0=d.xpos[palm].copy();R0=d.xmat[palm].reshape(3,3).copy();arm={g for g in range(m.ngeom)if m.geom(g).name.startswith('arm_collision_')}
 for g in arm:m.geom_margin[g]=.003
 def residual(x,p,R):
  d.qpos[qi]=x;mujoco.mj_kinematics(m,d)
  return np.r_[100*(d.xpos[palm]-p),Rotation.from_matrix(R@d.xmat[palm].reshape(3,3).T).as_rotvec()]
 def bad_contacts():
  mujoco.mj_forward(m,d);bad=[]
  for c in d.contact:
   names=[m.geom(g).name for g in [c.geom1,c.geom2]]
   touches_arm=c.geom1 in arm or c.geom2 in arm
   clearance=.003 if touches_arm and any(n=='cube'or n.startswith(('finger_tip_','source_geometry_'))for n in names)else -1e-6
   if c.dist<clearance and touches_arm:bad.append(dict(pair=names,distance_mm=float(1000*c.dist)))
  return bad
 rng=np.random.default_rng(724);branches=[];solutions=[]
 mirrored=seed.copy();mirrored[1]+=seed[2];mirrored[3]+=seed[2];mirrored[2]*=-1
 for k in range(32):
  d.qpos[:]=q0;start=seed if k==0 else mirrored if k==1 else seed+rng.normal(0,2,6)
  fit=least_squares(lambda x:residual(x,p0,R0),np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=180)
  if np.linalg.norm(residual(fit.x,p0,R0))>1e-5:continue
  x=(fit.x+np.pi)%(2*np.pi)-np.pi
  if any(np.linalg.norm(np.arctan2(np.sin(x-y),np.cos(x-y)))<.05 for y in solutions):continue
  solutions.append(x.copy());start=x.copy();failures=[];maxerror=0.
  for index,q in enumerate(recording['qpos'][::5]):
   d.qpos[:]=q;mujoco.mj_forward(m,d);p=d.xpos[palm].copy();R=d.xmat[palm].reshape(3,3).copy()
   fit=least_squares(lambda v:residual(v,p,R),x,bounds=(lo,hi),max_nfev=70);x=fit.x;maxerror=max(maxerror,float(np.linalg.norm(residual(x,p,R))))
   failures.extend(dict(frame=index*5,**c)for c in bad_contacts())
  branch=dict(initial_arm_q=start.tolist(),violations=failures,max_pose_residual=maxerror);branches.append(branch)
  print('BRANCH',len(branches),len(failures),maxerror,flush=True)
  if not failures and maxerror<1e-4:break
 selected=next((b for b in branches if not b['violations']and b['max_pose_residual']<1e-4),None)
 result=dict(scope='Offline alternate-arm-branch clearance preflight along recorded nominal motion; not an executed trial or proof for perturbed poses',source=source,source_npz_sha256=hashlib.sha256((OUT/(source+'.npz')).read_bytes()).hexdigest(),selected=selected,branches=branches)
 (OUT/(name+'-arm-plan.json')).write_text(json.dumps(result,indent=2))
 if selected is None:raise RuntimeError('No arm branch passed the recorded-motion preflight')
 q=q0.copy();q[qi]=selected['initial_arm_q'];d.qpos[:]=q;m.geom_margin[:]=0;mujoco.mj_forward(m,d)
 bad=[(m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist))for c in d.contact if c.dist< -1e-6]
 if bad:raise RuntimeError(('Initial penetration',bad))
 shutil.copy2(xml,OUT/(name+'.xml'));np.savez_compressed(OUT/(name+'-initial.npz'),qpos=q[None,:])
 report=json.loads((OUT/(source+'-initial.json')).read_text());report.update(arm_branch_preflight=str(OUT/(name+'-arm-plan.json')),initial_penetrations=[],initial_state_source=source,initial_arm_changed_only=True)
 (OUT/(name+'-initial.json')).write_text(json.dumps(report,indent=2))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--name',required=True);a=p.parse_args();plan(a.source,a.name)
