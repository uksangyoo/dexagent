"""Independent free-screw mechanics test, separate from manipulation rollouts.

Only this test applies an external wrench to a fastener. Task rollouts have
no such input. Check the helical contact law against pitch and step refinement.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse, json, time, xml.etree.ElementTree as ET
import numpy as np
import trimesh
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, SpatialInertia,
    UnitInertia, RigidTransform, Convex, Cylinder, ProximityProperties,
    AddRigidHydroelasticProperties, AddContactMaterial, CoulombFriction,
    ContactModel, DiscreteContactApproximation, ExternallyAppliedSpatialForce,
    SpatialForce, Simulator, CollisionFilterDeclaration, GeometrySet)
from screwdriver_lab.thread_geometry import add_helix, PITCH
from screwdriver_lab.hom.model import OUT


def run(segments=24, dt=.001, duration=1., load=.6, speed=90, beta=0.):
    folder=OUT/'thread-validation';folder.mkdir(exist_ok=True)
    b=DiagramBuilder();p,sg=AddMultibodyPlantSceneGraph(b,dt)
    screw=p.AddRigidBody('screw',SpatialInertia(.008,[0,0,-.004],UnitInertia(2.1e-7/.008,2.1e-7/.008,6e-8/.008)))
    root=ET.Element('mujoco');asset=ET.SubElement(root,'asset');male=ET.SubElement(root,'male');female=ET.SubElement(root,'female')
    add_helix(asset,male,'male',12,-.014,.0027,.00325,.00042,.00010,segments)
    add_helix(asset,female,'female',6,-.012,.00285,.0034,.00010,.00042,segments,PITCH/2)
    props={}
    for name in ['male','female']:
        prop=ProximityProperties();AddRigidHydroelasticProperties(.002,prop)
        AddContactMaterial(friction=CoulombFriction(.15,.15),properties=prop)
        prop.AddProperty('material','relaxation_time',.001)
        prop.AddProperty('material','point_contact_stiffness',1e8)
        props[name]=prop
    ids={'male':[],'female':[]}
    for mesh in asset:
        name=mesh.get('name');kind=name.split('_')[0]
        path=folder/f's{segments}-{name}.obj'
        if not path.exists():
            vertices=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3)
            temp=path.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(trimesh.Trimesh(vertices=vertices,process=False).convex_hull.export(file_type='obj'));temp.replace(path)
        ids[kind].append(p.RegisterCollisionGeometry(screw if kind=='male' else p.world_body(),RigidTransform(),Convex(str(path)),name,props[kind]))
    ids['male'].append(p.RegisterCollisionGeometry(screw,RigidTransform([0,0,-.008]),Cylinder(.0027,.016),'core',props['male']))
    declaration=CollisionFilterDeclaration()
    for gs in ids.values():declaration.ExcludeWithin(GeometrySet(gs))
    sg.collision_filter_manager().Apply(declaration)
    p.set_sap_near_rigid_threshold(beta);p.set_contact_model(ContactModel.kHydroelasticWithFallback);p.set_discrete_contact_approximation(DiscreteContactApproximation.kSap);p.Finalize()
    diagram=b.Build();ctx=diagram.CreateDefaultContext();pc=p.GetMyMutableContextFromRoot(ctx)
    wrench=ExternallyAppliedSpatialForce();wrench.body_index=screw.index();wrench.p_BoBq_B=np.zeros(3)
    port=p.get_applied_spatial_force_input_port();sim=Simulator(diagram,ctx)
    rows=[];previous=0.;angle=0.;start=time.monotonic()
    for k in range(round(duration/dt)):
        V=screw.EvalSpatialVelocityInWorld(pc);torque=np.clip(-.0004+.00002*(-np.radians(speed)-V.rotational()[2]),-.001,.001) if k*dt>.1 else 0.
        wrench.F_Bq_W=SpatialForce([0,0,torque],[0,0,-load*min(1.,k*dt/.1)]);port.FixValue(pc,[wrench]);sim.AdvanceTo((k+1)*dt)
        X=screw.EvalPoseInWorld(pc);R=X.rotation().matrix();yaw=np.arctan2(R[1,0],R[0,0]);angle-=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous));previous=yaw
        if k%10==0:
            contacts=p.get_contact_results_output_port().Eval(pc)
            rows.append([ctx.get_time(),np.degrees(angle),-X.translation()[2]*1000,torque,contacts.num_point_pair_contacts(),np.degrees(np.arccos(np.clip(R[2,2],-1,1)))])
    rows=np.array(rows);error=rows[:,2]-rows[:,1]/360
    report=dict(scope='Independent mechanics validation; external wrench only in this bench',segments=segments,dt=dt,load_N=load,command_deg_s=speed,elapsed_s=time.monotonic()-start,rotation_deg=float(rows[-1,1]),depth_mm=float(rows[-1,2]),pitch_residual_range_mm=[float(min(error)),float(max(error))],max_tilt_deg=float(max(rows[:,5])),trace=rows.tolist(),beta=beta,axial_load_ramp_s=.1,female_point_stiffness_N_m=1e8,torque_controller='clip(-0.0004 + 0.00002*(-desired_speed-omega_z), +/-0.001) Nm',male_point_stiffness_N_m=1e8)
    (OUT/f'thread-validation-filtered-s{segments}-dt{dt}-beta{beta}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='trace'}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--segments',type=int,default=24);a.add_argument('--dt',type=float,default=.001);a.add_argument('--duration',type=float,default=1.);a.add_argument('--beta',type=float,default=0.);v=a.parse_args();run(v.segments,v.dt,v.duration,beta=v.beta)
