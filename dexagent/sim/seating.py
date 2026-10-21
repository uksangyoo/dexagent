"""Observe the modeled head/workpiece clearance; never assign an object pose."""
import numpy as np
import mujoco


def geometry(model,data):
    floor=model.geom('slot_floor').id;screw=model.body('screw').id
    assert model.geom_bodyid[floor]==screw
    assert model.geom_type[floor]==mujoco.mjtGeom.mjGEOM_CYLINDER
    # Kinematics may have been updated without geom frames in the scratch
    # model. Compose directly from the measured rigid-body pose instead.
    R=data.xmat[screw].reshape(3,3)
    local_rotation=np.empty(9);mujoco.mju_quat2Mat(local_rotation,model.geom_quat[floor])
    axis=(R@local_rotation.reshape(3,3))[:,2]
    center=data.xpos[screw]+R@model.geom_pos[floor]
    radius,half_height=model.geom_size[floor,:2]
    top=[]
    for name in ['workpiece_0_-1','workpiece_0_1','workpiece_1_-1','workpiece_1_1']:
        g=model.geom(name).id;b=int(model.geom_bodyid[g]);B=data.xmat[b].reshape(3,3)
        G=np.empty(9);mujoco.mju_quat2Mat(G,model.geom_quat[g]);G=B@G.reshape(3,3)
        p=data.xpos[b]+B@model.geom_pos[g]
        top.append(float(p[2]+abs(G[2])@model.geom_size[g]))
    assert max(top)-min(top)<1e-9
    plane=top[0]
    center_gap=float(center[2]-half_height*axis[2]-plane)
    rim_extent=float(radius*np.sqrt(max(0.,1-axis[2]**2)))
    # The expected final screw-origin height assumes its head is flat.
    seated_origin_z=float(plane-model.geom_pos[floor,2]+half_height)
    return dict(head_gap_min_m=center_gap-rim_extent,
                head_gap_max_m=center_gap+rim_extent,
                flat_seat_remaining_mm=float((data.xpos[screw,2]-seated_origin_z)*1000),
                seated_origin_z_m=seated_origin_z)


def contact_reached(observation,seat_load_N):
    return seat_load_N>.1 and observation['head_gap_max_m']<50e-6 and observation['head_gap_min_m']>-10e-6


def flush(observation):
    return observation['head_gap_max_m']<20e-6 and observation['head_gap_min_m']>-10e-6


from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
