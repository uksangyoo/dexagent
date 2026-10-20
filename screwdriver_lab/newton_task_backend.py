"""GPU 1 task dynamics with native convex contact and explicit robot motors."""
import os,time
os.environ['CUDA_VISIBLE_DEVICES']='1'
import xml.etree.ElementTree as E
import numpy as np
import mujoco,warp as wp,newton
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from screwdriver_lab.physical_task import MotorControl,OUT
from screwdriver_lab import soft_finger

class NewtonTaskBackend:
    def __init__(self,s,torque=.11):
        wp.init();wp.set_device('cuda:0')
        assert wp.get_device().pci_bus_id=='00000000:61:00'
        self.s=s;self.motor=MotorControl(s,torque);self.dt=.00025
        self.started=time.monotonic();self.collision_hz=getattr(s,"gpu_collision_hz",1000)
        root=E.fromstring(s.xml)
        # Native motor actuators retain their limits and exactly the same
        # ordering as the controller. Do not apply generalized assistance.
        for a in root.find('actuator'):
            name,joint,limit=a.get('name'),a.get('joint'),a.get('forcerange')
            a.clear();a.tag='motor';a.set('name',name);a.set('joint',joint);a.set('gear','1');a.set('forcerange',limit)
        root.find('option').set('timestep',str(self.dt))
        path=OUT/'newton-active.xml';path.write_text(E.tostring(root,encoding='unicode'))
        b=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_mjcf(str(path),parse_visuals=False,collapse_fixed_joints=False,enable_self_collisions=True,ctrl_direct=True)
        # Newton's builder defaults to a 10 cm speculative contact gap. That
        # overwhelms narrow millimetre threads; the source MJCF has gap=0.
        for i in range(b.shape_count):b.shape_gap[i]=0.
        for i in range(b.shape_count):
            if b.shape_type[i]==newton.GeoType.MESH:
                mesh=b.shape_source[i];hull=ConvexHull(mesh.vertices)
                b.shape_source[i]=newton.Mesh(mesh.vertices,hull.simplices.flatten())
                b.shape_type[i]=newton.GeoType.CONVEX_MESH
        for i in range(b.shape_count):
            for j in range(i+1,b.shape_count):
                if b.shape_body[i]==b.shape_body[j]:b.shape_collision_filter_pairs.append((i,j))
        self.model=b.finalize(device='cuda:0');self.state=self.model.state();self.other=self.model.state();self.ctrl=self.model.control()
        native=getattr(s,"gpu_contact_source","newton")=="native"
        self.solver=newton.solvers.SolverMuJoCo(self.model,use_mujoco_contacts=native,
            iterations=200,ls_iterations=100,tolerance=1e-6,solver='newton',integrator='implicitfast',cone='elliptic',njmax=8192,nconmax=2048)
        self.pipeline=newton.CollisionPipeline(self.model,broad_phase='sap',rigid_contact_max=2048)
        self.contacts=self.pipeline.contacts()
        self.capacity=self.solver.mjw_data.naconmax
        self.solved=newton.Contacts(self.capacity,0,device='cuda:0',requested_attributes={'force'})
        self.body_names=[n.split('/')[-1]for n in self.model.body_label]
        self.source_body_ids=np.array([s.m.body(n).id for n in self.body_names])
        self.shape_names=[n.split('/')[-1]for n in self.model.shape_label]
        self.sb=self.model.shape_body.numpy();self.com=self.model.body_com.numpy()
        self.observed_shapes=np.array([i for i,n in enumerate(self.shape_names)if n in ['handle','cap','shaft','driver_bit','driver_bit_cross'] or n.startswith('workpiece_')])
        self.free_maps=[];self.qmap=[];self.vmap=[];qs=self.model.joint_q_start.numpy();vs=self.model.joint_qd_start.numpy()
        # MJCF importer combines co-located scalar joints. The native solver
        # mappings below resolve each source coordinate without assuming order.
        mj=self.solver.mj_model
        self.sj=[];self.mj=[]
        for j in range(s.m.njnt):
            name=s.m.joint(j).name
            # Imported scalar names may carry their parent scope.
            candidates=[i for i in range(mj.njnt)if mj.joint(i).name.split('/')[-1]==name]
            if not candidates:continue
            k=candidates[0];n=7 if s.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
            self.sj.extend(range(s.m.jnt_qposadr[j],s.m.jnt_qposadr[j]+n))
            self.mj.extend(range(mj.jnt_qposadr[k],mj.jnt_qposadr[k]+n))
        # For the ordinary articulated robot and free screwdriver each source
        # joint maps directly to Newton. Additional thread DOFs are checked.
        for j,label in enumerate(self.model.joint_label):
            name=label.split('/')[-1];mid=mujoco.mj_name2id(s.m,mujoco.mjtObj.mjOBJ_JOINT,name)
            if name=='screw_depth_screw_x_screw_y_screw_tilt_x_screw_tilt_y':
                mid=s.m.joint('screw_depth').id
            if mid<0:continue
            a,v=int(s.m.jnt_qposadr[mid]),int(s.m.jnt_dofadr[mid]);b0,w=int(qs[j]),int(vs[j])
            nq=int(qs[j+1]-qs[j]);nv=int(vs[j+1]-vs[j])
            if s.m.jnt_type[mid]==mujoco.mjtJoint.mjJNT_FREE:
                self.free_maps.append((mid,j,int(self.model.joint_child.numpy()[j])))
                self.qmap.extend([(a,b0),(a+1,b0+1),(a+2,b0+2),(a+3,b0+6),(a+4,b0+3),(a+5,b0+4),(a+6,b0+5)])
            elif nq==1:self.qmap.append((a,b0))
            elif nq==5 and name=='screw_depth_screw_x_screw_y_screw_tilt_x_screw_tilt_y':
                self.qmap.extend(zip(range(a,a+5),range(b0,b0+5)))
            else:raise RuntimeError(f'Combined joint needs explicit mapping: {label} nq={nq} nv={nv}')
            self.vmap.extend(zip(range(v,v+nv),range(w,w+nv)))
        self.ma,self.na=np.array(self.qmap).T;self.mv,self.nv=np.array(self.vmap).T
        if len(self.ma)!=s.m.nq:raise RuntimeError(f'Incomplete source pose mapping {len(self.ma)} / {s.m.nq}: {self.model.joint_label}')
        q=self.state.joint_q.numpy();q[self.na]=s.d.qpos[self.ma];self.state.joint_q.assign(q)
        qd=self.state.joint_qd.numpy();qd[self.nv]=s.d.qvel[self.mv]
        for mid,j,body in self.free_maps:
            a,v=int(s.m.jnt_qposadr[mid]),int(s.m.jnt_dofadr[mid]);w=int(vs[j])
            R=Rotation.from_quat(s.d.qpos[a+3:a+7],scalar_first=True).as_matrix()
            Rp=Rotation.from_quat(self.model.joint_X_p.numpy()[j,3:]).as_matrix()
            omega=R@s.d.qvel[v+3:v+6]
            qd[w:w+3]=Rp.T@(s.d.qvel[v:v+3]+np.cross(omega,R@self.com[body]))
            qd[w+3:w+6]=Rp.T@omega
        self.state.joint_qd.assign(qd)
        newton.eval_fk(self.model,self.state.joint_q,self.state.joint_qd,self.state)
        self.stats=dict(backend=('Newton / MuJoCo-Warp native convex contact'if native else 'Newton convex manifold / MuJoCo-Warp'),gpu_physical_index=1,
            gpu_pci=wp.get_device().pci_bus_id,timestep_s=self.dt,max_contacts=0,steps=0,
            hydroelastic=False,soft_finger_torsion=True,collision_refresh_hz=4000 if native else self.collision_hz)
        self.sample_q=self.state.body_q.numpy();self.sample_v=self.state.body_qd.numpy()
        native_shapes=self.solver.mjc_geom_to_newton_shape.numpy()[0]
        self.native_pads=[]
        for finger in range(1,6):
            shape=self.shape_names.index(f'finger_tip_{finger}')
            self.native_pads.append(int(np.flatnonzero(native_shapes==shape)[0]))
        self.native_friction=self.solver.mjw_model.geom_friction.numpy()
        self.ctrl.mujoco.ctrl.zero_()
        if not native:self.pipeline.collide(self.state,self.contacts)
        self.solver.step(self.state,self.other,self.ctrl,self.contacts,self.dt)
        # Capture four 0.25 ms substeps per 1 ms observer tick. Held motor
        # effort updates every other observer tick, exactly 500 Hz.
        self.sample=self.model.state()
        with wp.ScopedCapture(device='cuda:0') as cap:
            if not native and self.collision_hz==1000:self.pipeline.collide(self.state,self.contacts)
            for i in range(4):
                if not native and self.collision_hz==4000:self.pipeline.collide(self.state,self.contacts)
                if i==3:
                    wp.copy(self.sample.body_q,self.state.body_q);wp.copy(self.sample.body_qd,self.state.body_qd)
                # Refresh geometry at 1 kHz; the solver updates separation at
                # each 0.25 ms dynamics substep using the same local patch.
                self.state.clear_forces()
                self.solver.step(self.state,self.other,self.ctrl,self.contacts,self.dt)
                self.state,self.other=self.other,self.state
            self.solver.update_contacts(self.solved)
        self.graph=cap.graph
    def set_hand_impedance(self,*args):pass
    def step(self):
        s=self.s;soft_finger.update(s)
        for native,pad in zip(self.native_pads,s.tipids):self.native_friction[0,native,1]=s.m.geom_friction[pad,1]
        self.solver.mjw_model.geom_friction.assign(self.native_friction)
        self.ctrl.mujoco.ctrl.assign(self.motor.command().astype(np.float32))
        wp.capture_launch(self.graph)
        s.d.qpos[self.ma]=self.state.joint_q.numpy()[self.na];s.d.qvel[self.mv]=self.state.joint_qd.numpy()[self.nv]
        # Newton FREE joint twists use the COM and parent/world frame;
        # MuJoCo FREE qvel uses origin-linear and body-angular velocity.
        body_q=self.state.body_q.numpy();body_qd=self.state.body_qd.numpy()
        for mid,j,body in self.free_maps:
            a,v=int(s.m.jnt_qposadr[mid]),int(s.m.jnt_dofadr[mid])
            R=Rotation.from_quat(body_q[body,3:]).as_matrix();omega=body_qd[body,3:]
            s.d.qpos[a:a+3]=body_q[body,:3]
            s.d.qpos[a+3:a+7]=body_q[body,[6,3,4,5]]
            s.d.qvel[v:v+3]=body_qd[body,:3]-np.cross(omega,R@self.com[body])
            s.d.qvel[v+3:v+6]=R.T@omega
        s.d.time+=.001
        # Read-only robot kinematics and bias terms for the controller. Avoid a
        # second CPU contact/dynamics solve after the GPU already integrated it.
        mujoco.mj_kinematics(s.m,s.d);mujoco.mj_comPos(s.m,s.d)
        mujoco.mj_comVel(s.m,s.d);mujoco.mj_passive(s.m,s.d)
        mujoco.mj_rne(s.m,s.d,0,s.d.qfrc_bias)
        s.d.actuator_force[:]=self.motor.torque
        if self.stats['steps']%4000==0:
            error=float(np.max(np.linalg.norm(s.d.xpos[self.source_body_ids]-body_q[:,:3],axis=1)))
            self.stats['max_kinematic_position_error_mm']=max(self.stats.get('max_kinematic_position_error_mm',0),1000*error)
            if error>.001:raise RuntimeError(f'Newton/controller body-frame mismatch: {error} m')
        self.sample_q=self.sample.body_q.numpy();self.sample_v=self.sample.body_qd.numpy();self.stats['steps']+=4
        if self.stats['steps']%4000==0:
            print('GPU_PROFILE',round(s.d.time,3),'wall',round(time.monotonic()-self.started,2),
                  'solver_iterations',self.solver.mjw_data.solver_niter.numpy().tolist(),
                  'max_contacts',self.stats['max_contacts'],flush=True)
    def measure(self):
        s=self.s;c=self.solved;n=int(c.rigid_contact_count.numpy()[0])
        self.stats['max_contacts']=max(n,self.stats['max_contacts'])
        if n>=self.capacity:raise RuntimeError('Contact capacity exhausted')
        s.loads[:]=0;s.handleloads=[0.,0.];s.slip=[0.,0.];s.signed_slip=[0.,0.]
        s.index_cap_load=s.rigid_tool_load=s.axial=s.slot_force=s.seat=s.pen=s.forbidden=0.
        s.cap_contact_points=[]
        s.pad_points=np.zeros((2,3));s.pad_normals=np.zeros((2,3));s.pad_weights=np.zeros(2)
        aa=c.rigid_contact_shape0.numpy()[:n];bb=c.rigid_contact_shape1.numpy()[:n]
        p0=c.rigid_contact_point0.numpy()[:n];p1=c.rigid_contact_point1.numpy()[:n]
        normals=c.rigid_contact_normal.numpy()[:n];forces=c.force.numpy()[:n,:3]
        R=Rotation.from_quat(self.sample_q[:,3:]).as_matrix();pos=self.sample_q[:,:3]
        axis=s.d.xmat[s.tool].reshape(3,3)[:,2]
        valid=(aa>=0)&(bb>=0)
        ia=np.maximum(self.sb[np.maximum(aa,0)],0);ib=np.maximum(self.sb[np.maximum(bb,0)],0)
        wa=pos[ia]+np.einsum('nij,nj->ni',R[ia],p0)
        wb=pos[ib]+np.einsum('nij,nj->ni',R[ib],p1)
        wa[self.sb[np.maximum(aa,0)]<0]=p0[self.sb[np.maximum(aa,0)]<0]
        wb[self.sb[np.maximum(bb,0)]<0]=p1[self.sb[np.maximum(bb,0)]<0]
        if np.any(valid):
            depths=-np.einsum('ij,ij->i',wb-wa,normals);depths[~valid]=-np.inf
            worst=int(np.argmax(depths));s.pen=max(0.,float(depths[worst]))
            if s.pen>self.stats.get('max_penetration_m',0):
                self.stats['max_penetration_m']=s.pen
                self.stats['max_penetration_pair']=[self.shape_names[aa[worst]],self.shape_names[bb[worst]]]
                self.stats['max_penetration_time_s']=float(s.d.time)
        relevant=valid & (np.isin(aa,self.observed_shapes)|np.isin(bb,self.observed_shapes))
        for k in np.flatnonzero(relevant):
            a,b=aa[k],bb[k]
            if min(a,b)<0:continue
            bodies=[int(self.sb[a]),int(self.sb[b])];gn=[self.shape_names[a],self.shape_names[b]]
            bn=[self.body_names[i]if i>=0 else 'world'for i in bodies]
            pa=pos[bodies[0]]+R[bodies[0]]@p0[k]if bodies[0]>=0 else p0[k]
            pb=pos[bodies[1]]+R[bodies[1]]@p1[k]if bodies[1]>=0 else p1[k]
            s.pen=max(s.pen,-float(np.dot(pb-pa,normals[k])))
            load=abs(float(np.dot(forces[k],normals[k])));point=(pa+pb)/2
            if 'screw'in bn and any(g.startswith('workpiece_')for g in gn):s.seat+=load
            obj=next((i for i,g in enumerate(gn)if g in ['handle','cap','shaft','driver_bit','driver_bit_cross']),None)
            if obj is None:continue
            other=1-obj
            if 'screw'in bn:s.slot_force+=load
            if not bn[other].startswith('link_'):continue
            finger=int(bn[other].split('_')[1])-1;s.loads[finger]+=load
            if gn[obj]not in ['handle','cap']:s.forbidden+=load;continue
            if not gn[other].startswith('finger_tip_'):s.rigid_tool_load+=load
            if gn[obj]=='cap':
                s.axial+=max(0.,-float((forces[k]if obj==0 else -forces[k])[2]))
                if finger==1:
                    s.index_cap_load+=load
                    if load>.001:s.cap_contact_points.append(point.copy())
                continue
            if finger not in [0,2]:continue
            slot=0 if finger==0 else 1;s.handleloads[slot]+=load
            radial=point-s.d.xpos[s.tool];radial-=axis*np.dot(radial,axis)
            normal=normals[k].copy()
            if np.dot(normal,radial)<0:normal=-normal
            s.pad_points[slot]+=load*point;s.pad_normals[slot]+=load*normal;s.pad_weights[slot]+=load
            vel=[self.sample_v[i,:3]+np.cross(self.sample_v[i,3:],point-pos[i]-R[i]@self.com[i])for i in bodies]
            v=vel[other]-vel[obj];vt=v-normal*np.dot(v,normal)
            tangent=-np.cross(axis,radial/max(1e-9,np.linalg.norm(radial)))
            s.slip[slot]+=load*np.linalg.norm(vt)*1000;s.signed_slip[slot]+=load*np.dot(v,tangent)*1000
        for i in range(2):
            w=max(1e-9,s.pad_weights[i]);s.pad_points[i]/=w
            s.pad_normals[i]/=max(1e-9,np.linalg.norm(s.pad_normals[i]))
            s.slip[i]/=w;s.signed_slip[i]/=w
        rot=s.d.xmat[s.tool].reshape(3,3);yaw=-np.arctan2(rot[1,0],rot[0,0])
        s.tool_angle+=float(np.arctan2(np.sin(yaw-s.last_yaw),np.cos(yaw-s.last_yaw)));s.last_yaw=yaw
        s.peak=max(s.peak,max(s.loads));s.peakpen=max(s.peakpen,s.pen)
    def step_batch(self):
        for _ in range(20):self.step();self.measure()
        return [self.s.peak,self.s.peakpen,0,0,0]
