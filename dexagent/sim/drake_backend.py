"""Independent hydroelastic dynamics; MuJoCo is used only for frame auditing.

No object force, position controller or helical constraint is installed.
Hydroelastic patch tractions supply frictional forces and moments directly.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json
from pathlib import Path
import numpy as np
import mujoco
from pydrake.all import (DiagramBuilder,AddMultibodyPlantSceneGraph,Parser,
    RigidTransform,RotationMatrix,JointIndex,BodyIndex,Role,RoleAssign,
    ProximityProperties,AddCompliantHydroelasticProperties,AddRigidHydroelasticProperties,
    AddContactMaterial,CoulombFriction,ContactModel,DiscreteContactApproximation,
    SpatialVelocity,Simulator,Mesh,Convex,CollisionFilterDeclaration,GeometrySet)
from dexagent.sim.volume_mesh import tetrahedral_pad
from dexagent.sim.model import OUT,prepare,reached_state

class HydroPlant:
    def __init__(self,xml=None,dt=.001,hydro=True,resolution=.001,modulus=1.7e6,steel_model='point', explicit_steel=False, beta=1.,grip_friction=.6, normal_force_limit=1.5, null_posture_kp=.1,null_posture_kd=.01,tangential_stiffness=None,tangential_damping=3.):
        self.xml=Path(xml or prepare());self.m=mujoco.MjModel.from_xml_path(str(self.xml));self.d=mujoco.MjData(self.m)
        self.builder=DiagramBuilder();self.plant,self.scene=AddMultibodyPlantSceneGraph(self.builder,dt)
        p=self.plant;Parser(p,self.scene).AddModels(str(self.xml))
        self.geometry={};inspector=self.scene.model_inspector();self.properties=[]
        for gid in inspector.GetAllGeometryIds():
            if inspector.GetProximityProperties(gid)is None:continue
            name=inspector.GetName(gid).split('::')[-1];self.geometry[name]=gid
            if hydro:
                props=ProximityProperties()
                is_pad=name.startswith('finger_tip_')
                is_thread=name.startswith('male_thread_') or name=='screw_core'
                shape=inspector.GetShape(gid)
                if is_pad:
                    source=shape.source().path()
                    self.scene.ChangeShape(p.get_source_id(),gid,Mesh(str(tetrahedral_pad(source,resolution))))
                elif is_thread and isinstance(shape,Mesh):
                    # These wedge assets are explicit convex polyhedra.
                    self.scene.ChangeShape(p.get_source_id(),gid,Convex(shape.source().path()))
                friction=grip_friction if is_pad or name in ['handle','cap']else .15 if 'thread' in name or name=='screw_core'else .5
                AddContactMaterial(dissipation=10.,friction=CoulombFriction(friction,friction),properties=props)
                # SAP uses relaxation time, not Hunt-Crossley dissipation.
                props.AddProperty('material','relaxation_time',.001)
                if is_pad or (is_thread and steel_model=='hydroelastic'):
                    AddCompliantHydroelasticProperties(resolution,modulus if is_pad else 1e8,props)
                else:AddRigidHydroelasticProperties(min(resolution,.0005)if name in ['handle','cap']else resolution,props)
                if (is_thread or explicit_steel) and steel_model=='point':props.AddProperty('material','point_contact_stiffness',1e8)
                self.scene.AssignRole(p.get_source_id(),gid,props,RoleAssign.kReplace)
                self.properties.append(dict(name=name,model='compliant'if is_pad or (is_thread and steel_model=='hydroelastic')else'rigid',modulus_Pa=modulus if is_pad else 1e8 if is_thread and steel_model=='hydroelastic'else None,mu=friction))
        # Drake's MJCF parser does not implement contype/conaffinity masks.
        # Reapply their exact semantics, rather than silently adding pairs.
        groups={}
        for i in range(self.m.ngeom):
            name=self.m.geom(i).name
            if name not in self.geometry:continue
            key=(int(self.m.geom_contype[i]),int(self.m.geom_conaffinity[i]))
            groups.setdefault(key,[]).append(self.geometry[name])
        declaration=CollisionFilterDeclaration();keys=list(groups)
        for i,a in enumerate(keys):
            for b in keys[i:]:
                if not ((a[0]&b[1])or(b[0]&a[1])):
                    declaration.ExcludeBetween(GeometrySet(groups[a]),GeometrySet(groups[b]))
        # Geometry queries can include overlapping shapes attached to the
        # same rigid assembly. Those are material inside one rigid body,
        # not contacting bodies. In particular, the elastomer/backing and
        # neighboring thread sectors must not contact their own assembly.
        welded={}
        for name,gid in self.geometry.items():
            mid=self.m.geom(name).id
            key=int(self.m.body_weldid[self.m.geom_bodyid[mid]])
            welded.setdefault(key,[]).append(gid)
        for ids in welded.values():declaration.ExcludeWithin(GeometrySet(ids))
        self.scene.collision_filter_manager().Apply(declaration)
        for index in p.GetJointActuatorIndices():
            actuator=p.get_joint_actuator(index)
            jid=self.m.joint(actuator.joint().name()).id
            actuator.set_default_gear_ratio(1.)
            actuator.set_default_rotor_inertia(float(self.m.dof_armature[self.m.jnt_dofadr[jid]]))
        p.set_contact_model(ContactModel.kHydroelasticWithFallback if hydro else ContactModel.kPoint)
        p.set_sap_near_rigid_threshold(beta)
        p.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        p.Finalize();self.diagram=self.builder.Build();self.context=self.diagram.CreateDefaultContext();self.pc=p.GetMyMutableContextFromRoot(self.context)
        self.scalar=[]
        for j in range(self.m.njnt):
            if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
            dj=p.GetJointByName(self.m.joint(j).name)
            assert dj.num_positions()==1
            self.scalar.append((j,dj.position_start(),dj.velocity_start()))
        self.body_map={i:p.GetBodyByName(self.m.body(i).name)for i in range(1,self.m.nbody)if p.HasBodyNamed(self.m.body(i).name)}
        self.free=[j for j in range(self.m.njnt)if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
        self.actuator_names=[p.get_joint_actuator(i).name() for i in p.GetJointActuatorIndices()]
        self.input=p.get_actuation_input_port().FixValue(self.pc,np.zeros(p.num_actuated_dofs()))
        self.simulator=Simulator(self.diagram,self.context)
        self.control_joints=[p.get_joint_actuator(i).joint()for i in p.GetJointActuatorIndices()]
        self.qi=np.array([self.m.joint(j.name()).qposadr[0]for j in self.control_joints])
        self.vi=np.array([self.m.joint(j.name()).dofadr[0]for j in self.control_joints])
        self.dqi=np.array([j.position_start()for j in self.control_joints])
        self.dvi=np.array([j.velocity_start()for j in self.control_joints])
        self.limit=np.r_[np.full(20,.11),[150]*3,[28]*3]
        self.last_effort=np.zeros(26)
        self.normal_force_limit=float(normal_force_limit)
        self.null_posture_kp=float(null_posture_kp)
        self.null_posture_kd=float(null_posture_kd)
        self.tangential_stiffness=tangential_stiffness
        self.tangential_damping=float(tangential_damping)
        self.finger_impedances={} # Optional motor-servo overrides, never contact material.
    def sync(self):
        """Read dynamics into a kinematic scratch model; never step MuJoCo."""
        p=self.plant;q=p.GetPositions(self.pc);v=p.GetVelocities(self.pc)
        for j,iq,iv in self.scalar:
            self.d.qpos[self.m.jnt_qposadr[j]]=q[iq];self.d.qvel[self.m.jnt_dofadr[j]]=v[iv]
        for j in self.free:
            body=self.body_map[int(self.m.jnt_bodyid[j])];X=body.EvalPoseInWorld(self.pc);V=body.EvalSpatialVelocityInWorld(self.pc)
            qa=self.m.jnt_qposadr[j];va=self.m.jnt_dofadr[j]
            self.d.qpos[qa:qa+3]=X.translation();self.d.qpos[qa+3:qa+7]=X.rotation().ToQuaternion().wxyz()
            self.d.qvel[va:va+3]=V.translational();self.d.qvel[va+3:va+6]=X.rotation().matrix().T@V.rotational()
        self.d.time=self.context.get_time()
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d);mujoco.mj_comVel(self.m,self.d)
        return self.d
    def contacts(self):
        r=self.plant.get_contact_results_output_port().Eval(self.pc);inspector=self.scene.model_inspector();records=[]
        for k in range(r.num_hydroelastic_contacts()):
            c=r.hydroelastic_contact_info(k);s=c.contact_surface();F=c.F_Ac_W()
            names=[inspector.GetName(g).split('::')[-1]for g in [s.id_M(),s.id_N()]]
            records.append(dict(names=names,point=s.centroid(),force=F.translational(),moment=F.rotational(),area=s.total_area(),hydro=True))
        for k in range(r.num_point_pair_contacts()):
            c=r.point_pair_contact_info(k);pair=c.point_pair()
            names=[inspector.GetName(g).split('::')[-1]for g in [pair.id_A,pair.id_B]]
            records.append(dict(names=names,point=c.contact_point(),force=c.contact_force(),moment=np.zeros(3),area=0.,hydro=False))
        return records
    def motor_step(self,qref,vref=None,dt=.002,normal_tasks=None,damping_compensation=0.,spin_tasks=None):
        p=self.plant;self.sync();vref=np.zeros(26)if vref is None else vref
        kp=np.r_[np.full(20,3.),[1200]*3,[400]*3];kd=np.r_[np.full(20,.12),[40]*3,[5]*3]
        bias=p.CalcBiasTerm(self.pc)-p.CalcGravityGeneralizedForces(self.pc)
        tau=kp*(qref-self.d.qpos[self.qi])+kd*(vref-self.d.qvel[self.vi])+bias[self.dvi]
        if normal_tasks is not None:
            for i,task in enumerate(normal_tasks):
                J,n,target,measured=task[:4];i=task[4]if len(task)>4 else i;sl=slice(4*i,4*i+4);cols=self.vi[sl];Ji=J[:,cols]
                normal_reference=float(task[5])if len(task)>5 else 0.
                fn=np.clip(target+.5*(target-measured)+3*(np.dot(n,J@self.d.qvel)-normal_reference),0,self.normal_force_limit)
                # Common hybrid inner servo: tangential position/velocity,
                # normal pressure through robot joint torque. No object force.
                inverse=np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),Ji)
                equivalent_force=inverse@(tau[sl]-bias[self.dvi[sl]])
                tangent_kp,tangent_kd=self.finger_impedances.get(i,(self.tangential_stiffness,self.tangential_damping))
                if tangent_kp is not None:
                    # Task-space impedance avoids multiplying joint damping
                    # by the inverse squared finger lever arm near singularity.
                    error=Ji@(qref[sl]-self.d.qpos[self.qi[sl]])
                    velocity_goal=Ji@vref[sl]+J[:,self.vi[20:]]@vref[20:]
                    equivalent_force=tangent_kp*error+tangent_kd*(velocity_goal-J@self.d.qvel)
                tangent=equivalent_force-n*np.dot(n,equivalent_force)
                # Limit tangential requests to a modest friction overdrive;
                # otherwise an infeasible position error consumes the entire
                # motor budget and destroys normal pressure regulation.
                budget=1.25*.6*max(target,measured,.02)
                tangent*=min(1.,budget/max(1e-12,np.linalg.norm(tangent)))
                null=np.eye(4)-Ji.T@inverse
                posture=self.null_posture_kp*(qref[sl]-self.d.qpos[self.qi[sl]])-self.null_posture_kd*self.d.qvel[cols]
                tau[sl]=Ji.T@(tangent-n*fn)+null@posture+bias[self.dvi[sl]]
                # Optional pickup assistance: an angular-velocity servo on
                # the fingertip, projected into the contact-point nullspace.
                # This remains a bounded robot-motor effort. No wrench is
                # applied to the tool, and the prepared-grasp study omits it.
                for finger,Jr,axis,goal,gain in spin_tasks or []:
                    if finger!=i:continue
                    error=goal-float(axis@Jr@self.d.qvel)
                    tau[sl]+=null@Jr[:,cols].T@(axis*gain*error)
        # Feedforward through bounded motors; the plant damping is unchanged.
        for sl in [slice(0,4),slice(8,12)]:tau[sl]+=damping_compensation*self.m.dof_damping[self.vi[sl]]*self.d.qvel[self.vi[sl]]
        available=self.limit.copy();available[:20]=np.minimum(available[:20],.65*np.maximum(0,1-abs(self.d.qvel[self.vi[:20]])/(85*2*np.pi/60)))
        self.last_effort=np.clip(tau,-available,available)
        self.input.GetMutableData().get_mutable_value().SetFromVector(self.last_effort)
        self.simulator.AdvanceTo(round(self.context.get_time()+dt,10));self.sync()
    def initialize_from_mujoco(self,qpos,qvel=None):
        """Initialization only; this is never called inside a dynamic rollout."""
        if self.context.get_time()!=0:raise RuntimeError('State initialization after time zero is prohibited')
        self.d.qpos[:]=qpos;self.d.qvel[:]=0 if qvel is None else qvel;mujoco.mj_forward(self.m,self.d)
        p=self.plant;q=p.GetPositions(self.pc).copy();v=p.GetVelocities(self.pc).copy()
        for j,iq,iv in self.scalar:q[iq]=qpos[self.m.jnt_qposadr[j]];v[iv]=self.d.qvel[self.m.jnt_dofadr[j]]
        p.SetPositions(self.pc,q);p.SetVelocities(self.pc,v)
        for j in self.free:
            bodyid=int(self.m.jnt_bodyid[j]);body=self.body_map[bodyid];qadr=self.m.jnt_qposadr[j];vadr=self.m.jnt_dofadr[j]
            R=self.d.xmat[bodyid].reshape(3,3)
            p.SetFreeBodyPose(self.pc,body,RigidTransform(RotationMatrix(R),self.d.xpos[bodyid]))
            p.SetFreeBodySpatialVelocity(self.pc,body,SpatialVelocity(R@self.d.qvel[vadr+3:vadr+6],self.d.qvel[vadr:vadr+3]))
    def audit(self):
        errors=[];mass=[];inertia=[];p=self.plant
        for i,b in self.body_map.items():
            X=b.EvalPoseInWorld(self.pc);errors.append(dict(body=b.name(),translation_m=float(np.linalg.norm(X.translation()-self.d.xpos[i])),rotation=float(np.linalg.norm(X.rotation().matrix()-self.d.xmat[i].reshape(3,3)))))
            mass.append(abs(b.default_mass()-self.m.body_mass[i]))
            R=np.empty(9);mujoco.mju_quat2Mat(R,self.m.body_iquat[i]);R=R.reshape(3,3);c=self.m.body_ipos[i]
            expected=R@np.diag(self.m.body_inertia[i])@R.T+self.m.body_mass[i]*(np.dot(c,c)*np.eye(3)-np.outer(c,c))
            inertia.append(float(np.linalg.norm(b.default_spatial_inertia().CalcRotationalInertia().CopyToFullMatrix3()-expected)))
        missing=[self.m.geom(i).name for i in range(self.m.ngeom)if (self.m.geom_contype[i] or self.m.geom_conaffinity[i]) and self.m.geom(i).name not in self.geometry]
        assert not missing,missing
        assert max(mass)<1e-10 and max(inertia)<1e-10
        assert max(e['translation_m']for e in errors)<1e-8 and max(e['rotation']for e in errors)<1e-8,sorted(errors,key=lambda e:e['translation_m'])[-4:]
        mujoco.mj_forward(self.m,self.d);M=np.zeros((self.m.nv,self.m.nv));mujoco.mj_fullM(self.m,self.d,M)
        D=p.CalcMassMatrixViaInverseDynamics(self.pc)
        mass_matrix_error=float(np.max(abs(M[np.ix_(self.vi,self.vi)]-D[np.ix_(self.dvi,self.dvi)])))
        assert mass_matrix_error<1e-9,mass_matrix_error
        B=np.zeros((p.num_velocities(),self.m.nv))
        for j,iq,iv in self.scalar:B[iv,self.m.jnt_dofadr[j]]=1
        for j in self.free:
            bid=int(self.m.jnt_bodyid[j]);b=self.body_map[bid];iv=b.floating_velocities_start_in_v();mv=self.m.jnt_dofadr[j]
            B[iv:iv+3,mv+3:mv+6]=self.d.xmat[bid].reshape(3,3);B[iv+3:iv+6,mv:mv+3]=np.eye(3)
        full_mass_error=float(np.max(abs(M-B.T@D@B)))
        passive_armature={self.m.joint(j).name:float(self.m.dof_armature[self.m.jnt_dofadr[j]])for j,_,_ in self.scalar if self.m.jnt_dofadr[j]not in self.vi and self.m.dof_armature[self.m.jnt_dofadr[j]]>0}
        if not passive_armature:assert full_mass_error<1e-9,full_mass_error
        return dict(body_count=len(errors),max_mass_error_kg=max(mass),max_inertia_error_kg_m2=max(inertia),actuated_mass_matrix_max_error=mass_matrix_error,full_mass_matrix_max_error=full_mass_error,unsupported_passive_armature=passive_armature,max_position_error_m=max(e['translation_m']for e in errors),max_rotation_error=max(e['rotation']for e in errors),missing_collision_geometries=missing,geometry_count=len(self.geometry),actuators=self.actuator_names,properties=self.properties)

if __name__=='__main__':
    h=HydroPlant(hydro=False)
    q=reached_state(h.xml)
    h.initialize_from_mujoco(q)
    r=h.audit();(OUT/'drake-conversion-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k!='properties'},indent=2))

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
