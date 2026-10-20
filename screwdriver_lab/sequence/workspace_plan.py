"""Kinematic robot posture planning around existing measured object contacts.

Planning uses a separate scratch model. The returned joint target is not a
physical state reset, and feasibility here is not a manipulation success.
"""
import json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def plan(m,q,locals_,margin=.10):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 names=[f'joint_{i}_{j}'for i in [1,2,3]for j in range(1,5)]+['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']
 ids=np.array([m.joint(n).id for n in names]);qi=m.jnt_qposadr[ids];seed=q[qi].copy();bounds=m.jnt_range[ids].copy();bounds[:12,0]+=margin;bounds[:12,1]-=margin
 bounds[12:,0]=np.maximum(bounds[12:,0],seed[12:]-.25);bounds[12:,1]=np.minimum(bounds[12:,1],seed[12:]+.25)
 fingers=[m.body(f'link_{i}_4').id for i in [1,2,3]];palm=m.body('palm').id
 points=[d.xpos[b]+d.xmat[b].reshape(3,3)@l for b,l in zip(fingers,locals_)]
 p0=d.xpos[palm].copy();R0=d.xmat[palm].reshape(3,3).copy()
 def residual(x):
  d.qpos[qi]=x;mujoco.mj_kinematics(m,d)
  positions=np.concatenate([1000*(d.xpos[b]+d.xmat[b].reshape(3,3)@l-point)for b,l,point in zip(fingers,locals_,points)])
  return np.r_[positions,30*Rotation.from_matrix(d.xmat[palm].reshape(3,3)@R0.T).as_rotvec(),20*(d.xpos[palm]-p0),.05*(x-seed)]
 fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-6,bounds[:,1]-1e-6),bounds=(bounds[:,0],bounds[:,1]),max_nfev=250,xtol=1e-9,ftol=1e-9,gtol=1e-9)
 residual(fit.x);candidate=d.qpos.copy();errors=np.linalg.norm(residual(fit.x)[:9].reshape(3,3),axis=1)
 target_p=d.xpos[palm].copy();target_R=d.xmat[palm].reshape(3,3).copy();collisions=[]
 allowed=[{'handle','finger_tip_1'},{'cap','finger_tip_2'},{'handle','finger_tip_3'}]
 for fraction in np.linspace(0,1,21):
  d.qpos[:]=q;d.qpos[qi]=seed+fraction*(fit.x-seed);mujoco.mj_forward(m,d)
  for c in d.contact:
   pair={m.geom(g).name for g in [c.geom1,c.geom2]}
   if c.dist<-.00001 and pair not in allowed and any(n.startswith(('finger_tip_','source_geometry_','arm_collision_'))for n in pair):collisions.append(dict(fraction=float(fraction),pair=sorted(pair),penetration_mm=float(-1000*c.dist)))
 result=dict(scope='Kinematic robot-only posture plan; no physical rollout credit',joint_names=names,joint_target=fit.x.tolist(),joint_margin_rad=margin,contact_position_error_mm=errors.tolist(),palm_translation_m=(target_p-p0).tolist(),palm_rotation_deg=float(np.degrees(Rotation.from_matrix(target_R@R0.T).magnitude())),collision_samples=collisions,accepted=bool(max(errors)<.05 and not collisions))
 return candidate,result

if __name__=='__main__':
 import argparse
 from pydrake.all import HydroelasticContactRepresentation
 from screwdriver_lab.hom.model import OUT
 from screwdriver_lab.hom.drake_backend import HydroPlant
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--frame',type=int,default=0);a=p.parse_args()
 data=np.load(OUT/(a.source+'.npz'));q=data['qpos'][a.frame];h=HydroPlant(OUT/(a.source+'.xml'),dt=.00025,resolution=.002,explicit_steel=True,beta=0);h.initialize_from_mujoco(q);h.sync()
 query=h.scene.get_query_output_port().Eval(h.scene.GetMyContextFromRoot(h.context));ins=query.inspector();surfaces,_=query.ComputeContactSurfacesWithFallback(HydroelasticContactRepresentation.kTriangle)
 locals_=[None]*3
 for surface in surfaces:
  names={ins.GetName(g).split('::')[-1]for g in [surface.id_M(),surface.id_N()]}
  for i,target in [(0,'handle'),(1,'cap'),(2,'handle')]:
   if names=={f'finger_tip_{i+1}',target}:
    b=h.m.body(f'link_{i+1}_4').id;locals_[i]=h.d.xmat[b].reshape(3,3).T@(surface.centroid()-h.d.xpos[b])
 if any(l is None for l in locals_):raise ValueError('Three actual contact patches required')
 target,result=plan(h.m,q,locals_);out=OUT/(a.source+'-workspace-plan.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
