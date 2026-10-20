"""Fit the observed index-cap / thumb-middle handle grasp without changing old baselines."""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.screw_scene import build,CENTER,R,ROOT
R=Rot.from_euler('y',90,degrees=True).as_matrix()@R
OUT=ROOT/'screwdriver_lab/artifacts/reference_strategy'

def fit():
 OUT.mkdir(exist_ok=True,parents=True);m=mujoco.MjModel.from_xml_string(build());d=mujoco.MjData(m);palm=m.body('palm').id
 ji=np.array([m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)]);qi=m.jnt_qposadr[ji];lo=m.jnt_range[ji,0]+1e-4;hi=m.jnt_range[ji,1]-1e-4
 q0=np.array(json.loads((ROOT/'screwdriver_lab/artifacts/screw_in/precision_grasp.json').read_text())['q']);pairs=[(m.geom(f'finger_tip_{i+1}').id,m.geom('cap'if i==1 else'handle').id)for i in range(3)]
 handgeoms=[g for g in range(m.ngeom)if m.body(int(m.geom_bodyid[g])).name.startswith('link_')];toolgeoms=[m.geom(n).id for n in ['cap','handle','shaft']]
 best=None
 for sign in [-90,-45,0,-135]:
  direction=np.array([np.cos(np.radians(sign)),np.sin(np.radians(sign)),0.]);desired=np.array([CENTER+direction*.012+[0,0,0.],[0,0,.29412],CENTER-direction*.012+[0,0,0.]])
  # First fit pad origins, then actual convex surfaces and contact clearance.
  targets=desired+np.array([direction*.010,[0,0,.008],-direction*.010])
  def setstate(x):
   m.body_pos[palm]=x[20:23];m.body_quat[palm]=Rot.from_matrix(Rot.from_rotvec(x[23:26]).as_matrix()@R).as_quat(scalar_first=True);d.qpos[qi]=x[:20];mujoco.mj_forward(m,d)
  def coarse(x):
   setstate(x);return np.r_[(d.site_xpos[:3]-targets).ravel()*250,.025*(x[:20]-q0),.1*x[23:26]]
  low=np.r_[lo,[.045,-.10,.16],[-1.2,-1.2,-1.2]];high=np.r_[hi,[.17,.10,.36],[1.2,1.2,1.2]]
  high[6]=1.1;high[7]=.8;q0[6:8]=[.6,.3]
  x=least_squares(coarse,np.r_[q0,[.10,0,.28],[0,0,0]],bounds=(low,high),max_nfev=250).x;ref=x.copy()
  def surface(x):
   setstate(x);terms=[]
   for(a,b),target in zip(pairs,desired):
    line=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,a,b,.3,line);terms.extend([dist*1800]);terms.extend((line[3:]-target)*350)
   pen=[max(0,(-.00002 if (m.geom(g).name=='finger_tip_2'and m.geom(h).name=='cap')or(m.geom(g).name in ['finger_tip_1','finger_tip_3']and m.geom(h).name=='handle')else .001)-mujoco.mj_geomDistance(m,d,g,h,.1,None))for g in handgeoms for h in toolgeoms]
   selfpen=np.zeros(m.ngeom)
   for c in d.contact:
    if c.geom1 in handgeoms and c.geom2 in handgeoms:selfpen[c.geom1]=max(selfpen[c.geom1],-c.dist-.00002)
   return np.r_[terms,selfpen*8000,np.array(pen)*8000,.015*(x[:20]-ref[:20]),.02*(x[20:]-ref[20:])]
  if False:
   previous=json.loads((OUT/'grasp.json').read_text());x=np.r_[previous['q'],previous['palm'],Rot.from_matrix(np.array(previous['rotation'])@R.T).as_rotvec()];x=np.clip(x,low+1e-6,high-1e-6)
  x=least_squares(surface,x,bounds=(low,high),max_nfev=400,diff_step=1e-4).x;res=surface(x);cost=np.linalg.norm(res[:-26]);print('candidate',sign,'cost',cost,'res',res[:13],flush=True)
  if best is None or cost<best[0]:best=(cost,x.copy(),desired.copy())
 cost,x,desired=best;setstate(x);patches=[];local=[];normals=[]
 for i,(a,b)in enumerate(pairs):
  line=np.zeros(6);mujoco.mj_geomDistance(m,d,a,b,.3,line);bid=m.body(f'link_{i+1}_4').id;patches.append((line[3:]-CENTER).tolist());local.append((d.xmat[bid].reshape(3,3).T@(line[:3]-d.xpos[bid])).tolist());n=np.array([0.,0.,1.])if i==1 else np.r_[line[3:5]-CENTER[:2],0];normals.append((n/np.linalg.norm(n)).tolist())
 result=dict(q=x[:20].tolist(),palm=x[20:23].tolist(),rotation=(Rot.from_rotvec(x[23:26]).as_matrix()@R).tolist(),patches=patches,local=local,normals=normals,roles=['thumb: handle drive','index: axial cap support','middle: handle drive'],fit_residual=float(cost))
 (OUT/'grasp.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':fit()
