"""Fit centered distal-pad contacts and inward pad normals for a rolling pinch."""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.screw_scene import ROOT,build,CENTER
OUT=ROOT/'screwdriver_lab/artifacts/controlled_slip_tip';OUT.mkdir(exist_ok=True)
def fit(radius=.012):
 old=json.loads((ROOT/'screwdriver_lab/artifacts/controlled_slip_clear/grasp.json').read_text());m=mujoco.MjModel.from_xml_string(build({'radius_mm':radius*1000}));d=mujoco.MjData(m);palm=m.body('palm').id;R0=np.array(old['rotation']);ji=np.array([m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)]);qi=m.jnt_qposadr[ji];x0=np.r_[old['q'],old['palm'],[0,0,0]];lo=np.r_[m.jnt_range[ji,0]+1e-4,np.array(old['palm'])-.08,[-1.2]*3];hi=np.r_[m.jnt_range[ji,1]-1e-4,np.array(old['palm'])+.08,[1.2]*3]
 local=np.array([[.0468,0,0],[.0359,0,0],[.0359,0,0]]);normals=np.array([[0,-1,0],[0,0,1],[0,1,0]]);patches=np.array([[0,-radius,0],[0,0,.043],[0,radius,0]]);bids=[m.body(f'link_{i+1}_4').id for i in range(3)]
 hand=[g for g in range(m.ngeom)if m.body(int(m.geom_bodyid[g])).name.startswith('link_')];targets=[m.geom(n).id for n in ['handle','cap','shaft']];pairs=[(m.geom(f'finger_tip_{i+1}').id,m.geom('cap'if i==1 else'handle').id)for i in range(3)]
 def state(x):m.body_pos[palm]=x[20:23];m.body_quat[palm]=Rot.from_matrix(Rot.from_rotvec(x[23:26]).as_matrix()@R0).as_quat(scalar_first=True);d.qpos[qi]=x[:20];mujoco.mj_forward(m,d)
 def fun(x):
  state(x);terms=[]
  for i,b in enumerate(bids):
   R=d.xmat[b].reshape(3,3);terms.extend((d.xpos[b]+R@local[i]-CENTER-patches[i])*1500);terms.extend((R[:,0]+normals[i])*12)
  for a in hand:
   for b in targets:terms.append(max(0,(-.00002 if(a,b)in pairs else .001)-mujoco.mj_geomDistance(m,d,a,b,.02,None))*3000)
  selfpen=np.zeros(m.ngeom)
  for c in d.contact:
   if c.geom1 in hand and c.geom2 in hand:selfpen[c.geom1]=max(selfpen[c.geom1],-c.dist-.00001)
  return np.r_[terms,selfpen*6000,.03*(x[:20]-x0[:20]),.1*(x[20:]-x0[20:])]
 rng=np.random.default_rng(7);best=None
 for k in range(5):
  start=x0.copy()if best is None else best.x.copy()
  if k:start[:12]+=rng.normal(0,.15,12);start[23:]+=rng.normal(0,.12,3)
  f=least_squares(fun,np.clip(start,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=200,diff_step=1e-5);score=np.linalg.norm(fun(f.x));print('FIT',k,f.nfev,score,flush=True)
  if best is None or score<np.linalg.norm(fun(best.x)):best=f
 x=best.x;state(x);print('res',fun(x)[:18],flush=True)
 g=dict(q=x[:20].tolist(),palm=x[20:23].tolist(),rotation=(Rot.from_rotvec(x[23:26]).as_matrix()@R0).tolist(),patches=patches.tolist(),local=local.tolist(),normals=normals.tolist(),roles=old['roles'],radius_mm=radius*1000,fit_residual=float(np.linalg.norm(fun(x))),fit_goal='Centered distal tips with opposed pad normals')
 (OUT/'grasp.json').write_text(json.dumps(g,indent=2))
 import screwdriver_lab.reference_control as rc
 from screwdriver_lab.reference_full import clear_arm
 saved=rc.REF;rc.REF=OUT
 try:s=rc.ReferenceSim({'radius_mm':radius*1000})
 finally:rc.REF=saved
 aq=clear_arm(s,s.p0,s.R0);g['arm_q']=aq.tolist();g['arm_fixture_lift']=.12;(OUT/'grasp.json').write_text(json.dumps(g,indent=2));print('ARM_OK',flush=True)
if __name__=='__main__':fit()
