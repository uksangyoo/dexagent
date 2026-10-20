from screwdriver_lab.full_scene import build_full,OUT
from screwdriver_lab.screw_scene import CENTER
import mujoco,numpy as np
from scipy.optimize import differential_evolution,least_squares
xml,g=build_full();m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);j=[m.joint(f'joint_{i}_{k}').id for i in range(1,6) for k in range(1,5)];qi=m.jnt_qposadr[j];d.qpos[qi]=g['q'];a=m.joint('tool_free').qposadr[0];d.qpos[a:a+3]=CENTER+[0,0,.025];site=m.site('tip_1').id
target=CENTER+[.022,0,.025]
def fun(q):
 d.qpos[qi[:4]]=q;mujoco.mj_forward(m,d);return np.linalg.norm(d.site_xpos[site]-target)**2
f=differential_evolution(fun,m.jnt_range[j[:4]],seed=4,maxiter=120,popsize=10,tol=1e-8);print('xyzfit',f.fun,f.x,d.site_xpos[site],flush=True)
handle=m.geom('handle').id;tip=m.geom('finger_tip_1').id;target=CENTER+[.012,0,.025]
def res(q):
 d.qpos[qi[:4]]=q;mujoco.mj_forward(m,d);line=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,tip,handle,.3,line)
 pen=sum(max(0,-c.dist-.00001)**2 for c in d.contact)
 return np.r_[dist*1200,(line[3:]-target)*200,np.sqrt(pen)*5000]
f2=least_squares(res,f.x,bounds=(m.jnt_range[j[:4],0]+1e-5,m.jnt_range[j[:4],1]-1e-5),diff_step=1e-4,max_nfev=250);print('surface',res(f2.x),'q',f2.x,flush=True)
import json
line=np.zeros(6);mujoco.mj_geomDistance(m,d,tip,handle,.3,line);b=m.body('link_1_4').id
out=dict(q=f2.x.tolist(),local=(d.xmat[b].reshape(3,3).T@(line[:3]-d.xpos[b])).tolist(),patch=(line[3:]-CENTER).tolist(),normal=[1.,0.,0.])
(OUT/'pickup-thumb.json').write_text(json.dumps(out))
