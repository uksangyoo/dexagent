"""Move contact patches away from the rigid bases of the fingertip pads."""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.screw_scene import ROOT,build,CENTER
OUT=ROOT/'screwdriver_lab/artifacts/controlled_slip_opposed';OUT.mkdir(exist_ok=True)
def fit():
 old=json.loads((ROOT/'screwdriver_lab/artifacts/controlled_slip_clear/grasp.json').read_text());m=mujoco.MjModel.from_xml_string(build());d=mujoco.MjData(m);palm=m.body('palm').id;R0=np.array(old['rotation']);ji=np.array([m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)]);qi=m.jnt_qposadr[ji];x0=np.r_[old['q'],old['palm'],[0,0,0]];lo=np.r_[m.jnt_range[ji,0]+1e-4,np.array(old['palm'])-.045,[-.6]*3];hi=np.r_[m.jnt_range[ji,1]-1e-4,np.array(old['palm'])+.045,[.6]*3]
 hand=[g for g in range(m.ngeom)if m.body(int(m.geom_bodyid[g])).name.startswith('link_')];targets=[m.geom(n).id for n in ['handle','cap','shaft']];pairs=[(m.geom(f'finger_tip_{i+1}').id,m.geom('cap'if i==1 else'handle').id)for i in range(3)];desired=np.array([[0,-.012,0],[0,0,.043],[0,.012,0]])+CENTER;minimum_x=[.038,.027,.027]
 def state(x):m.body_pos[palm]=x[20:23];m.body_quat[palm]=Rot.from_matrix(Rot.from_rotvec(x[23:26]).as_matrix()@R0).as_quat(scalar_first=True);d.qpos[qi]=x[:20];mujoco.mj_forward(m,d)
 def fun(x):
  state(x);terms=[]
  for i,(a,b)in enumerate(pairs):
   line=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,a,b,.1,line);bid=m.body(f'link_{i+1}_4').id;local=d.xmat[bid].reshape(3,3).T@(line[:3]-d.xpos[bid]);terms.extend([dist*2000,max(0,minimum_x[i]-local[0])*3000]);terms.extend((line[3:]-desired[i])*1500)
  for a in hand:
   for b in targets:
    margin=-.00001 if(a,b)in pairs else .002
    terms.append(max(0,margin-mujoco.mj_geomDistance(m,d,a,b,.1,None))*5000)
  selfpen=np.zeros(m.ngeom)
  for c in d.contact:
   if c.geom1 in hand and c.geom2 in hand:selfpen[c.geom1]=max(selfpen[c.geom1],-c.dist-.00002)
  return np.r_[terms,selfpen*6000,.03*(x[:20]-x0[:20]),.1*(x[20:]-x0[20:])]
 print('INITIAL',np.linalg.norm(fun(x0)),flush=True);fit=least_squares(fun,x0,bounds=(lo,hi),max_nfev=300,diff_step=1e-4);x=fit.x;print('FIT',fit.nfev,np.linalg.norm(fun(x)),flush=True);state(x);patches=[];local=[];normals=[]
 for i,(a,b)in enumerate(pairs):
  line=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,a,b,.1,line);bid=m.body(f'link_{i+1}_4').id;patches.append((line[3:]-CENTER).tolist());local.append((d.xmat[bid].reshape(3,3).T@(line[:3]-d.xpos[bid])).tolist());n=np.array([0.,0.,1.])if i==1 else np.r_[line[3:5]-CENTER[:2],0.];normals.append((n/np.linalg.norm(n)).tolist());print('PAD',i,'gap',dist*1000,'local',local[-1],flush=True)
 g=dict(q=x[:20].tolist(),palm=x[20:23].tolist(),rotation=(Rot.from_rotvec(x[23:26]).as_matrix()@R0).tolist(),patches=patches,local=local,normals=normals,roles=old['roles'],fit_residual=float(np.linalg.norm(fun(x))),fit_goal='Distal pad contact with 2 mm rigid-link clearance')
 (OUT/'grasp.json').write_text(json.dumps(g,indent=2))
 import screwdriver_lab.reference_control as rc
 from screwdriver_lab.reference_full import clear_arm
 saved=rc.REF;rc.REF=OUT
 try:s=rc.ReferenceSim()
 finally:rc.REF=saved
 aq=clear_arm(s,s.p0,s.R0);g['arm_q']=aq.tolist();g['arm_fixture_lift']=.12;(OUT/'grasp.json').write_text(json.dumps(g,indent=2));print('ARM_OK',flush=True)
if __name__=='__main__':fit()
