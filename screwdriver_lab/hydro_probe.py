"""Newton hydroelastic backend feasibility probe, restricted to physical GPU 1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json,numpy as np,mujoco,warp as wp,newton
from pathlib import Path
from screwdriver_lab.screw_sim import ScrewSim
from newton.geometry import HydroelasticSDF
OUT=Path(__file__).resolve().parent/'artifacts/full_motion'
def main():
 import warp._src.marching_cubes as mc
 if not hasattr(mc,'mc_cube_corner_offsets'):mc.mc_cube_corner_offsets=mc.MC_CUBE_CORNER_OFFSETS
 wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
 from screwdriver_lab.full_motion import FullSim
 s=FullSim()
 path=OUT/'hydro-probe.xml';path.write_text(s.xml)
 b=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(b)
 b.add_mjcf(str(path),parse_visuals=False,collapse_fixed_joints=False,enable_self_collisions=True)
 print('JOINTS',list(zip(b.joint_label,b.joint_type,b.joint_q_start)),flush=True)
 print('EQ',b.mujoco.equality_constraint_count if hasattr(b,'mujoco') else 'builder attrs',flush=True)
 others=[];selected=[]
 for i,label in enumerate(b.shape_label):
  name=label.split('/')[-1]
  if name.startswith('finger_tip_') or name in ('handle','cap'):
   if b.shape_type[i]==newton.GeoType.MESH:
    mesh=b.shape_source[i];scale=np.asarray(b.shape_scale[i]);mesh=mesh.copy(vertices=mesh.vertices*scale,recompute_inertia=True);b.shape_source[i]=mesh;b.shape_scale[i]=(1.,1.,1.)
    mesh.build_sdf(max_resolution=64,narrow_band_range=(-.003,.003),margin=.001,cache_dir=OUT/'sdf-cache')
   else:
    b.shape_sdf_max_resolution[i]=64;b.shape_sdf_narrow_band_range[i]=(-.003,.003)
   b.shape_flags[i]|=newton.ShapeFlags.HYDROELASTIC;b.shape_material_kh[i]=1e9;b.shape_gap[i]=.0005;selected.append(i)
  else:others.append(i)
 from scipy.spatial import ConvexHull
 for i in others:
  if b.shape_type[i]==newton.GeoType.MESH:
   mesh=b.shape_source[i];h=ConvexHull(mesh.vertices);b.shape_source[i]=newton.Mesh(mesh.vertices,h.simplices.flatten());b.shape_type[i]=newton.GeoType.CONVEX_MESH
 for i in range(b.shape_count):
  for j in range(i+1,b.shape_count):
   if b.shape_body[i]==b.shape_body[j]:b.shape_collision_filter_pairs.append((i,j))
 print('HYDRO_SHAPES',[(i,b.shape_label[i]) for i in selected],flush=True)
 model=b.finalize(device='cuda:0');state=model.state();other=model.state();ctrl=model.control()
 print('EQUALITY',model.mujoco.equality_constraint_count,model.mujoco.equality_constraint_polycoef.numpy(),flush=True)
 # Build named scalar/free joint maps; current authored model may combine screw DOFs.
 print('MAPS',model.joint_label,model.joint_q_start.numpy(),flush=True)
 solver=newton.solvers.SolverMuJoCo(model,use_mujoco_contacts=False,iterations=80,ls_iterations=15,solver='newton',integrator='implicitfast',cone='elliptic',njmax=8192,nconmax=4096)
 print('MJCF_NATIVE',solver.mj_model.nq,solver.mj_model.nv,solver.mj_model.neq,flush=True)
 pipeline=newton.CollisionPipeline(model,broad_phase='explicit',sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_fraction=.1,margin_contact_area=1e-8));contacts=pipeline.contacts()
 q=state.joint_q.numpy();target=ctrl.joint_target_q.numpy()
 for j,label in enumerate(model.joint_label):
  name=label.split('/')[-1];mid=mujoco.mj_name2id(s.m,mujoco.mjtObj.mjOBJ_JOINT,name)
  if mid<0:continue
  a=s.m.jnt_qposadr[mid];b0=int(model.joint_q_start.numpy()[j])
  if s.m.jnt_type[mid]==mujoco.mjtJoint.mjJNT_FREE:q[b0:b0+7]=np.r_[s.d.qpos[a:a+3],s.d.qpos[a+4:a+7],s.d.qpos[a+3]]
  else:q[b0]=s.d.qpos[a];target[b0]=q[b0]
 state.joint_q.assign(q);ctrl.joint_target_q.assign(target)
 newton.eval_fk(model,state.joint_q,state.joint_qd,state);pipeline.collide(state,contacts)
 print('POSE_ERROR',max(np.linalg.norm(state.body_q.numpy()[i,:3]-s.d.xpos[s.m.body(label.split('/')[-1]).id]) for i,label in enumerate(model.body_label)),flush=True)
 print('CONTACTS',contacts.rigid_contact_count.numpy(),flush=True)
 from collections import Counter
 n=int(contacts.rigid_contact_count.numpy()[0]);aa=contacts.rigid_contact_shape0.numpy()[:n];bb=contacts.rigid_contact_shape1.numpy()[:n]
 print('PAIRS',Counter((b.shape_label[int(a)],b.shape_label[int(z)])for a,z in zip(aa,bb)).most_common(20),flush=True)
 print('GAPS',model.shape_gap.numpy().min(),model.shape_gap.numpy().max(),flush=True)
 solver.step(state,other,ctrl,contacts,.001);wp.synchronize();print('STEP_OK',other.joint_q.numpy(),flush=True)
if __name__=='__main__':main()
