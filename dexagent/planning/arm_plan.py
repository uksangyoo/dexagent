"""Choose an initial UR5e IK branch with clearance along the task corridor.

Kinematic planning only. Every dynamic replay still executes bounded arm motors.
The object is never moved by this planner during a dynamic rollout.
"""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from dexagent.planning.geometry import ART
from dexagent.sim.model import OUT

NAMES=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']

def solve(xml,q,output=None,max_starts=60):
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 palm=m.body('palm').id;p0=d.xpos[palm].copy();R0=d.xmat[palm].reshape(3,3).copy();ids=[m.joint(n).id for n in NAMES];qi=m.jnt_qposadr[ids];seed=q[qi].copy();lo=m.jnt_range[ids,0]+1e-4;hi=m.jnt_range[ids,1]-1e-4
 arm_geoms={g for g in range(m.ngeom)if m.geom(g).name.startswith(('arm_collision_','finger_tip_','source_geometry_'))}
 for g in arm_geoms:m.geom_margin[g]=max(m.geom_margin[g],.005)
 carried={'handle','cap','driver_bit','shaft','stripe'};rng=np.random.default_rng(1509);candidates=[]
 def pose(x,target):
  d.qpos[qi]=x;mujoco.mj_kinematics(m,d)
  return np.r_[100*(d.xpos[palm]-target),Rotation.from_matrix(R0@d.xmat[palm].reshape(3,3).T).as_rotvec()]
 def collisions():
  mujoco.mj_forward(m,d)
  return [dict(names=[m.geom(c.geom1).name,m.geom(c.geom2).name],depth_mm=-float(c.dist)*1000)for c in d.contact if c.dist<(.005 if any(m.geom(g).name.startswith(('pickup_','workpiece_'))for g in [c.geom1,c.geom2])else -1e-6) and (c.geom1 in arm_geoms or c.geom2 in arm_geoms)and not({m.geom(c.geom1).name,m.geom(c.geom2).name}&carried)]
 initial_solutions=[]
 mirrored=seed.copy();mirrored[1]+=seed[2];mirrored[3]+=seed[2];mirrored[2]*=-1;mirrored=(mirrored+np.pi)%(2*np.pi)-np.pi
 for attempt in range(max_starts):
  start=seed.copy()if attempt==0 else mirrored.copy()if attempt==1 else seed+rng.normal(0,2.,6)
  fit=least_squares(lambda x:pose(x,p0),np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  err=pose(fit.x,p0);x=(fit.x+np.pi)%(2*np.pi)-np.pi
  if np.linalg.norm(err)>1e-4 or any(np.linalg.norm(np.arctan2(np.sin(x-y),np.cos(x-y)))<.05 for y in initial_solutions):continue
  initial_solutions.append(x.copy());pose(x,p0);bad=collisions();path=[];fail=bad[:];maxerror=0.
  tool=m.body('screwdriver').id;screw=m.body('screw').id;delta=d.xpos[screw]-d.xpos[tool];delta[2]+=.123
  waypoints=[np.array([0,0,0]),np.array([0,0,.14]),np.r_[delta[:2],.14],delta]
  for left,right in zip(waypoints[:-1],waypoints[1:]):
   for alpha in np.linspace(0,1,21)[1:]:
    target=p0+left*(1-alpha)+right*alpha
    fit=least_squares(lambda y:pose(y,target),x,bounds=(lo,hi),max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-9);x=fit.x;err=pose(x,target);maxerror=max(maxerror,float(np.linalg.norm(err)));bad=collisions();path.append(dict(q=x.tolist(),offset=(target-p0).tolist(),collisions=bad))
    if bad:fail+=bad
  candidate=dict(initial_q=initial_solutions[-1].tolist(),collision_count=len(fail),worst_penetration_mm=max([r['depth_mm']for r in fail]+[0]),max_pose_residual=maxerror,path=path)
  candidates.append(candidate);print('ARM_BRANCH',len(candidates),candidate['collision_count'],candidate['worst_penetration_mm'],maxerror,flush=True)
  if candidate['collision_count']==0 and maxerror<1e-3:break
 if not candidates:raise RuntimeError('No IK solution')
 best=min(candidates,key=lambda r:(r['collision_count']>0,r['max_pose_residual']>.001,r['worst_penetration_mm'],r['collision_count']))
 if output:
  Path(output).write_text(json.dumps(dict(scope='Full robot geometry corridor preflight with frozen pregrasp; not executed motion and 5 mm fixture clearance; not a guarantee for moving finger configurations',selected=best,candidates=candidates),indent=2))
 if best['collision_count']or best['max_pose_residual']>.001:raise RuntimeError('No clear task corridor')
 result=q.copy();result[qi]=best['initial_q'];return result,best

if __name__=='__main__':
 q=np.load(OUT/'sequence-pivot-v17.npz')['initial_qpos'];q,plan=solve(OUT/'sequence-pivot-v17.xml',q,ART/'arm-corridor-plan.json');np.savez_compressed(ART/'arm-corridor-initial.npz',qpos=q)
