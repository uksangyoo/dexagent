"""Distributed SDF contact patches solved by Newton/MuJoCo-Warp on GPU 1.

MuJoCo in the controller is used only for named kinematics. All forces below
come from Newton's solved constraints, sampled at the preceding step state.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np,mujoco,warp as wp,newton
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot
from newton.geometry import HydroelasticSDF
from screwdriver_lab.full_scene import OUT

class HydroBackend:
 def __init__(self,s,resolution=64,kh=5e10):
  import warp._src.marching_cubes as mc
  if not hasattr(mc,'mc_cube_corner_offsets'):mc.mc_cube_corner_offsets=mc.MC_CUBE_CORNER_OFFSETS
  wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True;self.s=s;self.kh=kh
  import xml.etree.ElementTree as E
  root=E.fromstring(s.xml.replace('condim="6"','condim="3"'))
  for i,act in enumerate(root.find('actuator')):
   act.set('kp',str(s.m.actuator_gainprm[i,0]));act.set('kv',str(-s.m.actuator_biasprm[i,2]))
  path=OUT/'hydro-active.xml';path.write_text(E.tostring(root,encoding='unicode'))
  b=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(b);b.add_mjcf(str(path),parse_visuals=False,collapse_fixed_joints=False,enable_self_collisions=True)
  for i,label in enumerate(b.shape_label):
   name=label.split('/')[-1];b.shape_gap[i]=.0003
   if name.startswith('finger_tip_') or name in ('handle','cap'):
    if b.shape_type[i]==newton.GeoType.MESH:
     mesh=b.shape_source[i];mesh=mesh.copy(vertices=mesh.vertices*np.asarray(b.shape_scale[i]),recompute_inertia=True);b.shape_source[i]=mesh;b.shape_scale[i]=(1.,1.,1.);mesh.build_sdf(max_resolution=resolution,narrow_band_range=(-.003,.003),margin=.0005,cache_dir=OUT/'sdf-cache')
    else:b.shape_sdf_max_resolution[i]=resolution;b.shape_sdf_narrow_band_range[i]=(-.003,.003)
    b.shape_flags[i]|=newton.ShapeFlags.HYDROELASTIC;b.shape_material_kh[i]=kh
   elif b.shape_type[i]==newton.GeoType.MESH:
    mesh=b.shape_source[i];h=ConvexHull(mesh.vertices);b.shape_source[i]=newton.Mesh(mesh.vertices,h.simplices.flatten());b.shape_type[i]=newton.GeoType.CONVEX_MESH
  for a in range(b.shape_count):
   for z in range(a+1,b.shape_count):
    if b.shape_body[a]==b.shape_body[z]:b.shape_collision_filter_pairs.append((a,z))
  self.model=b.finalize(device='cuda:0');self.state=self.model.state();self.other=self.model.state();self.ctrl=self.model.control();self.body_names=[n.split('/')[-1]for n in self.model.body_label];self.shape_names=[n.split('/')[-1]for n in self.model.shape_label];self.sb=self.model.shape_body.numpy();self.com=self.model.body_com.numpy()
  self.solver=newton.solvers.SolverMuJoCo(self.model,use_mujoco_contacts=False,iterations=80,ls_iterations=15,solver='newton',integrator='implicitfast',cone='elliptic',njmax=4096,nconmax=1024)
  self.pipeline=newton.CollisionPipeline(self.model,broad_phase='explicit',sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_fraction=1.0,margin_contact_area=1e-8));self.contacts=self.pipeline.contacts();self.solved=newton.Contacts(max(self.solver.mjw_data.naconmax,self.contacts.rigid_contact_max),0,device='cuda:0',requested_attributes={'force'})
  self.qmap=[];self.vmap=[];qs=self.model.joint_q_start.numpy();vs=self.model.joint_qd_start.numpy()
  for j,label in enumerate(self.model.joint_label):
   mid=mujoco.mj_name2id(s.m,mujoco.mjtObj.mjOBJ_JOINT,label.split('/')[-1])
   if mid<0:continue
   a,v=int(s.m.jnt_qposadr[mid]),int(s.m.jnt_dofadr[mid]);b0,w=int(qs[j]),int(vs[j]);free=s.m.jnt_type[mid]==mujoco.mjtJoint.mjJNT_FREE
   if free:self.qmap.extend([(a,b0),(a+1,b0+1),(a+2,b0+2),(a+3,b0+6),(a+4,b0+3),(a+5,b0+4),(a+6,b0+5)])
   else:self.qmap.append((a,b0))
   self.vmap.extend(zip(range(v,v+(6 if free else 1)),range(w,w+(6 if free else 1))))
  self.ma,self.na=np.array(self.qmap).T;self.mv,self.nv=np.array(self.vmap).T
  q=self.state.joint_q.numpy();q[self.na]=s.d.qpos[self.ma];self.state.joint_q.assign(q);self.ctrl.joint_target_q.assign(q);newton.eval_fk(self.model,self.state.joint_q,self.state.joint_qd,self.state)
  mapping=dict(self.qmap);self.actn=np.array([mapping[int(s.m.jnt_qposadr[j])]for j in s.m.actuator_trnid[:,0]])
  self.stats=dict(backend='Newton SDF hydroelastic + MuJoCo-Warp',gpu_physical=1,sdf_resolution=resolution,kh_Pa_m=kh,max_contact_count=0,steps=0,patch_area_mm2=[0]*5,contact_capacity=1024);self.sample_q=self.state.body_q.numpy();self.sample_v=self.state.body_qd.numpy()
 def set_hand_impedance(self,kp,kv):
  import newton
  ke=self.model.joint_target_ke.numpy();kd=self.model.joint_target_kd.numpy()
  # Coordinate-layout targets: scalar finger coordinate indices are actuator indices.
  ke[self.actn[:20]]=kp;kd[self.actn[:20]]=kv;self.model.joint_target_ke.assign(ke);self.model.joint_target_kd.assign(kd)
  self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
 def prepare_graph(self):
  from screwdriver_lab.hydro_watchdog import clear_loads,contacts_watch,record,capacity_watch
  self.guard=wp.zeros(5,dtype=float,device='cuda:0');self.gloads=wp.zeros(5,dtype=float,device='cuda:0')
  fingers=[];targets=[]
  for i,name in enumerate(self.shape_names):
   bn=self.body_names[self.sb[i]] if self.sb[i]>=0 else ''
   fingers.append(int(bn.split('_')[1])-1 if bn.startswith('link_') else -1);targets.append(int(bn in ['screwdriver','swivel_cap']))
  self.finger=wp.array(fingers,dtype=int,device='cuda:0');self.target_shape=wp.array(targets,dtype=int,device='cuda:0')
  self.sample=self.model.state()
  # Warm up compilation and allocations without advancing the retained state.
  self.pipeline.collide(self.state,self.contacts);self.solver.step(self.state,self.other,self.ctrl,self.contacts,.001);self.solver.update_contacts(self.solved)
  with wp.ScopedCapture(device='cuda:0') as cap:
   for k in range(20):
    if k==19:
     wp.copy(self.sample.body_q,self.state.body_q);wp.copy(self.sample.body_qd,self.state.body_qd)
    self.state.clear_forces();self.pipeline.collide(self.state,self.contacts);self.solver.step(self.state,self.other,self.ctrl,self.contacts,.001);self.solver.update_contacts(self.solved)
    wp.launch(clear_loads,5,[self.gloads],device='cuda:0')
    c=self.solved
    wp.launch(contacts_watch,c.rigid_contact_max,[c.rigid_contact_count,c.rigid_contact_shape0,c.rigid_contact_shape1,c.rigid_contact_point0,c.rigid_contact_point1,c.rigid_contact_normal,c.force,self.model.shape_body,self.state.body_q,self.finger,self.target_shape,self.gloads,self.guard],device='cuda:0')
    wp.launch(record,5,[self.gloads,self.contacts.rigid_contact_count,self.guard],device='cuda:0')
    h=self.pipeline.hydroelastic_sdf
    for count,capacity in zip(h.iso_buffer_counts,[h.max_num_blocks_broad,*h.iso_max_dims]):wp.launch(capacity_watch,1,[count,capacity,self.guard],device='cuda:0')
    wp.launch(capacity_watch,1,[h.contact_reduction.contact_count,h.max_num_face_contacts,self.guard],device='cuda:0')
    self.state,self.other=self.other,self.state
  self.graph=cap.graph
 def step_batch(self):
  if not hasattr(self,'graph'):self.prepare_graph()
  s=self.s;target=self.ctrl.joint_target_q.numpy();target[self.actn]=s.d.ctrl;self.ctrl.joint_target_q.assign(target);self.guard.zero_();wp.capture_launch(self.graph)
  guard=self.guard.numpy();self.stats['max_contact_count']=max(int(guard[2]),self.stats['max_contact_count'])
  if guard[2]>=1024 or guard[3]>0:raise RuntimeError('Hydro contact buffer overflow: trial invalid')
  self.stats['max_internal_buffer_fraction']=max(self.stats.get('max_internal_buffer_fraction',0),float(guard[4]))
  s.d.qpos[self.ma]=self.state.joint_q.numpy()[self.na];s.d.qvel[self.mv]=self.state.joint_qd.numpy()[self.nv];s.d.time+=.020;mujoco.mj_forward(s.m,s.d);self.stats['steps']+=20
  self.sample_q=self.sample.body_q.numpy();self.sample_v=self.sample.body_qd.numpy();self.measure();s.peak=max(s.peak,float(guard[0]));s.peakpen=max(s.peakpen,float(guard[1]));return guard
 def step(self):
  s=self.s;target=self.ctrl.joint_target_q.numpy();target[self.actn]=s.d.ctrl;self.ctrl.joint_target_q.assign(target)
  self.sample_q=self.state.body_q.numpy();self.sample_v=self.state.body_qd.numpy();self.state.clear_forces();self.pipeline.collide(self.state,self.contacts)
  count=int(self.contacts.rigid_contact_count.numpy()[0]);self.stats['max_contact_count']=max(count,self.stats['max_contact_count'])
  if count>=1024 or count>=self.contacts.rigid_contact_max:raise RuntimeError('Hydro contact overflow')
  self.solver.step(self.state,self.other,self.ctrl,self.contacts,.001);self.solver.update_contacts(self.solved);self.state,self.other=self.other,self.state
  s.d.qpos[self.ma]=self.state.joint_q.numpy()[self.na];s.d.qvel[self.mv]=self.state.joint_qd.numpy()[self.nv];s.d.time+=.001;mujoco.mj_forward(s.m,s.d);self.stats['steps']+=1
 def measure(self):
  s=self.s;s.loads[:]=0;s.axial=0;net_axial=0.;s.cap_axial=0.;s.slot_force=0;s.seat=0;s.pen=0;s.forbidden=0;s.slip=[0.,0.];s.handleloads=[0.,0.];weights=[0.,0.]
  c=self.solved;n=int(c.rigid_contact_count.numpy()[0]);a=c.rigid_contact_shape0.numpy()[:n];b=c.rigid_contact_shape1.numpy()[:n];p0=c.rigid_contact_point0.numpy()[:n];p1=c.rigid_contact_point1.numpy()[:n];norm=c.rigid_contact_normal.numpy()[:n];forces=c.force.numpy()[:n,:3];Rs=Rot.from_quat(self.sample_q[:,3:]).as_matrix();positions=self.sample_q[:,:3]
  for k,(a0,b0)in enumerate(zip(a,b)):
   if min(a0,b0)<0:continue
   ba,bb=int(self.sb[a0]),int(self.sb[b0]);bn=[self.body_names[x] if x>=0 else 'world'for x in [ba,bb]];gn=[self.shape_names[a0],self.shape_names[b0]]
   pa=positions[ba]+Rs[ba]@p0[k]if ba>=0 else p0[k];pb=positions[bb]+Rs[bb]@p1[k]if bb>=0 else p1[k];distance=float(np.dot(pb-pa,norm[k]));s.pen=max(s.pen,-distance);load=abs(float(np.dot(forces[k],norm[k])))
   for target in ['screwdriver','swivel_cap']:
    if target not in bn:continue
    obj=bn.index(target);oi=1-obj
    if bn[oi].startswith('link_'):
     finger=int(bn[oi].split('_')[1])-1;s.loads[finger]+=load
     if gn[obj] not in ['handle','cap']:s.forbidden+=load
     net_axial-=float((forces[k] if obj==0 else -forces[k])[2])
     if gn[obj]=='cap':s.cap_axial+=max(0.,-float((forces[k] if obj==0 else -forces[k])[2]))
     if finger in [1,2] and gn[obj]=='handle':s.handleloads[finger-1]+=load
     if finger in [1,2] and gn[obj]=='handle' and load>.001:
      point=(pa+pb)/2;vel=[]
      for body in [ba,bb]:vel.append(self.sample_v[body,:3]+np.cross(self.sample_v[body,3:],point-positions[body]-Rs[body]@self.com[body]))
      v=vel[0]-vel[1];s.slip[finger-1]+=load*np.linalg.norm(v-norm[k]*np.dot(v,norm[k]))*1000;weights[finger-1]+=load
    if 'screw' in bn:s.slot_force+=load
   if 'screw' in bn and any(x.startswith('workpiece')for x in gn):s.seat+=load
  area=np.zeros(5);pc=self.contacts;pn=int(pc.rigid_contact_count.numpy()[0]);sa=pc.rigid_contact_shape0.numpy()[:pn];sb=pc.rigid_contact_shape1.numpy()[:pn];ke=pc.rigid_contact_stiffness.numpy()[:pn]
  for a,b,k in zip(sa,sb,ke):
   if min(a,b)<0 or k<=0:continue
   for f,obj in [(a,b),(b,a)]:
    name=self.body_names[self.sb[f]] if self.sb[f]>=0 else ''
    if name.startswith('link_') and self.shape_names[obj] in ['handle','cap']:area[int(name.split('_')[1])-1]+=2*float(k)/self.kh*1e6
  self.stats['patch_area_mm2']=area.tolist();self.stats['peak_patch_area_mm2']=np.maximum(self.stats.get('peak_patch_area_mm2',[0]*5),area).tolist();s.patch_area=area.tolist()
  s.hand_axial=max(0.,net_axial);s.axial=s.cap_axial
  s.slip=[float(v/max(w,1e-9))for v,w in zip(s.slip,weights)]
  R=s.d.xmat[s.tool].reshape(3,3);yaw=-np.arctan2(R[1,0],R[0,0]);s.tool_angle+=float(np.arctan2(np.sin(yaw-s.last_yaw),np.cos(yaw-s.last_yaw)));s.last_yaw=yaw;s.peak=max(s.peak,max(s.loads));s.peakpen=max(s.peakpen,s.pen)
