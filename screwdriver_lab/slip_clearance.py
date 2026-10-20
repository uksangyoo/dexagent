"""Retract unused fingers with geometric clearance from the active fingers."""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from screwdriver_lab.controlled_slip import SlipSim
s=SlipSim('screwdriver_lab/artifacts/controlled_slip');m,d=s.m,s.d
active=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('link_1_','link_2_','link_3_')) or m.geom(g).name in ['handle','cap','shaft']]
inactive=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('link_4_','link_5_'))]
pairs=[(a,b)for a in active for b in inactive]
pairs += [(a,b)for a in inactive for b in inactive if a<b and m.geom_bodyid[a]!=m.geom_bodyid[b] and m.body_parentid[m.geom_bodyid[a]]!=m.geom_bodyid[b] and m.body_parentid[m.geom_bodyid[b]]!=m.geom_bodyid[a]]
x0=d.qpos[s.qi[12:]].copy();limits=m.jnt_range[s.jids[12:]]
def fun(x):
 d.qpos[s.qi[12:]]=x;mujoco.mj_forward(m,d)
 return np.r_[[max(0,.004-mujoco.mj_geomDistance(m,d,a,b,.02,None))*1000 for a,b in pairs],.02*(x-x0)]
f=least_squares(fun,x0,bounds=(limits[:,0]+1e-4,limits[:,1]-1e-4),max_nfev=150,diff_step=1e-4)
print('fit',f.nfev,np.linalg.norm(fun(f.x)),f.x,flush=True)
g=s.grasp_calibration;g['q'][12:]=f.x.tolist();g['inactive_clearance_m']=.004
out=Path('screwdriver_lab/artifacts/controlled_slip_clear');out.mkdir(exist_ok=True);(out/'grasp.json').write_text(json.dumps(g,indent=2))
print('minimum',min(mujoco.mj_geomDistance(m,d,a,b,.02,None)for a,b in pairs),flush=True)
