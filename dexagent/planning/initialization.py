"""Develop a physical horizontal-tool pickup for the corrected model.

Only initial robot/tool placement uses inverse kinematics. Dynamic rollouts
must use HydroPlant motor efforts, with no subsequent object state assignment.
The historical pickup is used as an arm configuration seed, never replayed as
an object trajectory and never credited as a new hydroelastic task success.
"""
import json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import least_squares
from dexagent.sim.model import OUT,SOURCE,PREFIX,prepare,source_model


def initial_state(xml_override=None, index_prepose=False, arm_shift=None, output_stem=None,pickup_y=.1,pickup_x=-.22,reject_collisions=True):
    xml=xml_override or prepare(True,48);root=ET.parse(xml).getroot()
    for name,x,radius in [('cap',-.181,.006),('shaft',-.29,.001)]:
        g=next(g for g in root.iter('geom')if g.get('name')=='pickup_'+name)
        x+=pickup_x+.22;top=.43-radius;bottom=.12
        g.set('pos',f'{x} {pickup_y} {(top+bottom)/2}');g.set('size',f'.002 .025 {(top-bottom)/2}')
    xml=Path(xml_override).with_name((output_stem or'pickup-sequence-initial')+'.xml') if xml_override else OUT/'pickup-development.xml';xml.write_text(ET.tostring(root,encoding='unicode'))
    old=source_model();od=mujoco.MjData(old)
    oldq=np.load(PREFIX)['qpos'][45]
    od.qpos[:]=oldq;mujoco.mj_forward(old,od)
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
    for j in range(m.njnt):
        a=m.jnt_qposadr[j]
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            bid=old.body(m.body(int(m.jnt_bodyid[j])).name).id
            d.qpos[a:a+3]=od.xpos[bid];mujoco.mju_mat2Quat(d.qpos[a+3:a+7],od.xmat[bid])
        else:d.qpos[a]=od.qpos[old.joint(m.joint(j).name).qposadr[0]]
    if xml_override:
        si=m.joint('screw_free').qposadr[0];d.qpos[si:si+7]=m.qpos0[si:si+7]
    # The physical support heights match the new cap and shaft radii.
    qa=m.joint('tool_free').qposadr[0];d.qpos[qa:qa+3]=[pickup_x,pickup_y,.43]
    d.qpos[qa+3:qa+7]=[np.sqrt(.5),0,np.sqrt(.5),0]
    mujoco.mj_forward(m,d)
    # A more extended initial reach avoids starting the distal joints at their
    # limits. Move the pickup fixture away from the screw workpiece and solve
    # the arm pose at t=0; the subsequent lift is motor-driven.
    from scipy.spatial.transform import Rotation
    palm=m.body('palm').id;R0=d.xmat[palm].reshape(3,3).copy();p0=d.xpos[palm].copy()
    goal=p0+np.array([pickup_x+.22,pickup_y-.1,.08] if arm_shift is None else np.asarray(arm_shift)+[pickup_x+.22,pickup_y-.1,0])
    arm=np.array([m.joint(n).qposadr[0]for n in ['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']]);seed=d.qpos[arm].copy()
    def arm_error(q):
        d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
        return np.r_[100*(d.xpos[palm]-goal),Rotation.from_matrix(R0@d.xmat[palm].reshape(3,3).T).as_rotvec(),.001*(q-seed)]
    arm_opt=least_squares(arm_error,seed,bounds=(seed-2.5,seed+2.5),max_nfev=200);arm_error(arm_opt.x)
    tool=m.body('screwdriver').id;center=d.xpos[tool].copy();records=[]
    for i,sign in [(1,-1),(3,1)]:
        bid=m.body(f'link_{i}_4').id;local=np.array([.0466 if i==1 else .0357,-.0029,.0016])
        normal=np.array([0.,sign,0.]);target=center+.008*normal
        ids=np.array([m.joint(f'joint_{i}_{j}').id for j in range(1,5)]);qi=m.jnt_qposadr[ids];home=np.clip(d.qpos[qi],m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4)
        def residual(q):
            d.qpos[qi]=q;mujoco.mj_kinematics(m,d);R=d.xmat[bid].reshape(3,3)
            point=d.xpos[bid]+R@local
            return np.r_[(point-target)*1000,3*(R[:,0]+normal),.01*(q-home)]
        opt=least_squares(residual,home,bounds=(m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4),max_nfev=500)
        residual(opt.x);records.append(dict(finger=i,local=local.tolist(),target=target.tolist(),position_error_mm=float(np.linalg.norm(residual(opt.x)[:3]))))
    if index_prepose:
        i=2;bid=m.body('link_2_4').id;local=np.array([.0357,-.0029,.0016]);target=(center+np.array([.039,0,.0065]) if index_prepose=='horizontal_cap' else center+np.array([0,0,.048]))
        ids=np.array([m.joint(f'joint_2_{j}').id for j in range(1,5)]);qi=m.jnt_qposadr[ids];seed=d.qpos[qi].copy()
        def index_error(q):
            d.qpos[qi]=q;mujoco.mj_kinematics(m,d);R=d.xmat[bid].reshape(3,3);point=d.xpos[bid]+R@local
            return np.r_[1000*(point-target),3*(R[:,0]-np.array([0,0,-1])),.01*(q-seed)]
        opt=least_squares(index_error,np.clip(seed,m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4),bounds=(m.jnt_range[ids,0]+1e-4,m.jnt_range[ids,1]-1e-4),max_nfev=500)
        print('INDEX_PREPOSE_ERROR_MM',np.linalg.norm(index_error(opt.x)[:3]))
    unused=np.array([m.joint(f'joint_{i}_{j}').qposadr[0]for i in [4,5]for j in range(1,5)])
    d.qpos[unused]=[.827758,.66805968,1.36045791,.6022931,.01,1.33998422,1.46749509,1.5608]
    # Initialization-only clearance optimization for the unused fingers.
    unused_ids=[m.joint(f'joint_{i}_{j}').id for i in [4,5]for j in range(1,5)]
    geom_ids=[g for g in range(m.ngeom)if m.body(m.geom_bodyid[g]).name.startswith(('link_4_','link_5_'))]
    original_margin=m.geom_margin.copy();m.geom_margin[geom_ids]=.001
    seed=d.qpos[unused].copy()
    def clearance(q):
        d.qpos[unused]=q;mujoco.mj_forward(m,d);pen=np.zeros(len(geom_ids))
        for c in d.contact:
            for k,g in enumerate(geom_ids):
                if g in [c.geom1,c.geom2]:pen[k]=max(pen[k],(.0005-c.dist)*1000)
        return np.r_[pen,.01*(q-seed)]
    opt=least_squares(clearance,np.clip(seed,m.jnt_range[unused_ids,0]+1e-4,m.jnt_range[unused_ids,1]-1e-4),bounds=(m.jnt_range[unused_ids,0]+1e-4,m.jnt_range[unused_ids,1]-1e-4),max_nfev=250,diff_step=1e-4)
    clearance(opt.x);m.geom_margin[:]=original_margin
    mujoco.mj_forward(m,d)
    penetrations=[dict(names=[m.geom(c.geom1).name,m.geom(c.geom2).name],depth_mm=float(-c.dist*1000))for c in d.contact if c.dist<-.000001]
    result=dict(scope='Initialization only; not a successful pickup',contacts=records,initial_penetrations=penetrations)
    destination=Path(xml).parent
    np.savez_compressed(destination/((output_stem or'pickup-initial')+'.npz'),qpos=d.qpos[None,:]);(destination/((output_stem or'pickup-initial')+'.json')).write_text(json.dumps(result,indent=2))
    if reject_collisions and penetrations:raise RuntimeError('Reject penetrating initial state: '+str(penetrations[:5]))
    print('INITIAL_AUDIT',json.dumps(dict(finger_fit_mm=[r['position_error_mm']for r in records],penetration_count=len(penetrations),max_penetration_mm=max([r['depth_mm']for r in penetrations]+[0]))));return xml,d.qpos.copy(),records


if __name__=='__main__':initial_state()
