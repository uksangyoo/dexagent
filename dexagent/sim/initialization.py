"""Collision-aware initial configurations for isolated manipulation tests.

These are prepared-pose tests, not evidence of a successful pickup. All state
assignment ends before integration starts. Full-task runs require a new pickup.
"""
import json
import numpy as np,mujoco
from scipy.optimize import least_squares
from dexagent.sim.model import SOURCE,OUT,PREFIX,reached_state,source_model

def fit_grasp(xml,gap=.00015,offset=0,tilt_deg=None):
    # Prepared tests begin with 50 micrometres of bit clearance above the
    # head floor. Contact and any descent must develop under motor actuation.
    old=source_model();od=mujoco.MjData(old)
    od.qpos[:]=np.load(PREFIX)['qpos'][-1];mujoco.mj_forward(old,od)
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=reached_state(xml);mujoco.mj_forward(m,d)
    tool=m.body('screwdriver').id
    if tilt_deg is not None:
        from scipy.spatial.transform import Rotation
        qa=m.joint('tool_free').qposadr[0];screw=m.body('screw').id
        yaw=np.arctan2(d.xmat[tool].reshape(3,3)[1,0],d.xmat[tool].reshape(3,3)[0,0])
        R=Rotation.from_euler('y',tilt_deg,degrees=True)*Rotation.from_euler('z',yaw)
        axis=R.as_matrix()[:,2];d.qpos[qa:qa+3]=d.xpos[screw]+.12305*axis
        d.qpos[qa+3:qa+7]=R.as_quat(scalar_first=True);mujoco.mj_forward(m,d)
    center=d.xpos[tool].copy();axis=d.xmat[tool].reshape(3,3)[:,2];records=[]
    for i,targetname in [(1,'handle'),(2,'cap'),(3,'handle')]:
        oldbid=old.body(f'link_{i}_4').id;bid=m.body(f'link_{i}_4').id
        points=[c.pos for c in od.contact if {old.geom(c.geom1).name,old.geom(c.geom2).name}=={f'finger_tip_{i}',targetname}]
        if not points:raise RuntimeError(f'No reference contact for finger {i}')
        point=np.mean(points,axis=0);local=od.xmat[oldbid].reshape(3,3).T@(point-od.xpos[oldbid])
        radial=point-center;radial-=axis*np.dot(radial,axis);radial/=np.linalg.norm(radial)
        target=center+axis*.043+gap*axis if i==2 else center+axis*np.dot(point-center,axis)+(m.geom('handle').size[0]+gap)*radial
        ids=np.array([m.joint(f'joint_{i}_{j}').id for j in range(1,5)]);qi=m.jnt_qposadr[ids];home=d.qpos[qi].copy()
        def residual(q):
            d.qpos[qi]=q;mujoco.mj_kinematics(m,d)
            p=d.xpos[bid]+d.xmat[bid].reshape(3,3)@local
            return np.r_[(p-target)*1000,.03*(q-home)]
        opt=least_squares(residual,home,bounds=(m.jnt_range[ids,0]+1e-5,m.jnt_range[ids,1]-1e-5),max_nfev=250,xtol=1e-12,ftol=1e-12,gtol=1e-12)
        residual(opt.x);records.append(dict(finger=i,local=local.tolist(),target=target.tolist(),position_error_mm=float(np.linalg.norm(residual(opt.x)[:3]))))
    # Perturb tool and screw together relative to the robot for the matched
    # engagement tests; the workpiece moves with its screw in scenario XML.
    if offset:d.qpos[m.joint('tool_free').qposadr[0]+1]+=offset
    mujoco.mj_forward(m,d)
    records.append(dict(initial_contacts=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,depth_m=float(-c.dist))for c in d.contact if c.dist<0 and any(m.geom(x).name in ['handle','cap']for x in [c.geom1,c.geom2])]))
    return d.qpos.copy(),records

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
