"""Initial robot IK only: reserve fingertip joint travel for manipulation."""
import json
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def balanced_grasp(xml,q,records,margin=.12,starts=12,horizontal_cap=False,pulp=False):
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
    palm=m.body('palm').id;tool=m.body('screwdriver').id;center=d.xpos[tool].copy()
    R0=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy()
    names=[f'joint_{i}_{j}'for i in [1,2,3,4,5]for j in range(1,5)]+['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']
    ids=np.array([m.joint(n).id for n in names]);qi=m.jnt_qposadr[ids];seed=d.qpos[qi].copy()
    bounds=m.jnt_range[ids].copy();bounds[:20,0]+=margin;bounds[:20,1]-=margin;bounds[20:,0]=seed[20:]-.8;bounds[20:,1]=seed[20:]+.8
    loc=[np.array(records[0]['local']),np.array([.0357,-.0029,.0016]),np.array(records[1]['local'])]
    targets=[center+[0,-.008,0],center+([.039,0,.009] if horizontal_cap else [0,0,.053]),center+[0,.008,0]]
    if pulp:loc=[np.array([base+.00588841464,.00746608745,0])for base in [.0289,.018,.018]]
    surface_normal=np.array([.187702240,.982222897,-.00245951835])if pulp else np.array([1.,0,0])
    normal=[np.array([0,1,0]),np.array([0,0,-1]),np.array([0,-1,0])]
    bids=[m.body(f'link_{i}_4').id for i in [1,2,3]]
    robot_geoms=[i for i in range(m.ngeom)if m.geom(i).name.startswith(('source_geometry_','finger_tip_','arm_collision_'))]
    geometry_index={g:k for k,g in enumerate(robot_geoms)}
    original_margin=m.geom_margin.copy();m.geom_margin[robot_geoms]=.002
    def residual(x):
        d.qpos[qi]=x;mujoco.mj_forward(m,d);r=[]
        for i,b in enumerate(bids):
            R=d.xmat[b].reshape(3,3);r.extend(1000*(d.xpos[b]+R@loc[i]-targets[i]));r.extend(15*(R@surface_normal-normal[i]))
        r.extend(.15*(x[:20]-(bounds[:20,0]+bounds[:20,1])/2))
        r.extend(.3*(x[20:]-seed[20:]));r.extend(2*Rotation.from_matrix(R0@d.xmat[palm].reshape(3,3).T).as_rotvec());r.extend(10*(d.xpos[palm]-p0))
        penalties=np.zeros(len(robot_geoms))
        for c in d.contact:
            for g in (c.geom1,c.geom2):
                if g in geometry_index:penalties[geometry_index[g]]=max(penalties[geometry_index[g]],.0005-c.dist)
        r.extend(10000*penalties)
        return r
    rng=np.random.default_rng(20261016);best=None
    for attempt in range(starts):
        start=seed.copy() if attempt==0 else seed+rng.normal(0,.65,len(seed))
        if pulp:start[[3,7,11]]-=1.4
        result=least_squares(residual,np.clip(start,bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0],bounds[:,1]),max_nfev=350,diff_step=1e-5,ftol=1e-7,gtol=1e-7,xtol=1e-7)
        if best is None or result.cost<best.cost:best=result
        print('IK_START',attempt,'cost',result.cost,'best',best.cost,flush=True)
    result=best
    residual(result.x);m.geom_margin[:]=original_margin;mujoco.mj_forward(m,d)
    errors=[float(np.linalg.norm(d.xpos[b]+d.xmat[b].reshape(3,3)@loc[i]-targets[i])*1000)for i,b in enumerate(bids)]
    normalerr=[float(np.degrees(np.arccos(np.clip((d.xmat[b].reshape(3,3)@surface_normal)@normal[i],-1,1))))for i,b in enumerate(bids)]
    collisions=[dict(names=[m.geom(c.geom1).name,m.geom(c.geom2).name],penetration_mm=float(-c.dist*1000))for c in d.contact if c.dist<0]
    for i,k in [(0,0),(2,1)]:records[k]={**records[k],'target':targets[i].tolist(),'position_error_mm':errors[i],'local':loc[i].tolist(),'pad_normal_local':surface_normal.tolist()}
    report=dict(index_local=loc[1].tolist(),pad_normal_local=surface_normal.tolist(),position_error_mm=errors,normal_error_deg=normalerr,joint_margin_rad=margin,initial_collisions=collisions,palm_position=d.xpos[palm].tolist(),finger_joint_positions=result.x[:20].tolist())
    return d.qpos.copy(),records,report

if __name__=='__main__':
    from .initialization import initial_state
    from .geometry import ART
    xml,q,records=initial_state(ART/'precision-slotted.xml',True)
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--horizontal-cap',action='store_true');p.add_argument('--pulp',action='store_true');a=p.parse_args()
    q,records,r=balanced_grasp(xml,q,records,horizontal_cap=a.horizontal_cap,pulp=a.pulp,starts=2)
    np.savez_compressed(ART/('balanced-pulp.npz'if a.pulp else'balanced-horizontal.npz'if a.horizontal_cap else'balanced-initial.npz'),qpos=q[None,:]);(ART/('balanced-pulp.json'if a.pulp else'balanced-horizontal.json'if a.horizontal_cap else'balanced-initial.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
