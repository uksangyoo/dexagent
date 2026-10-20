"""Mechanics-only M3 thread convergence check; not a manipulation rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,time,xml.etree.ElementTree as E
import numpy as np,mujoco
from pydrake.all import (DiagramBuilder,AddMultibodyPlantSceneGraph,SpatialInertia,UnitInertia,RigidTransform,Convex,Cylinder,ProximityProperties,AddRigidHydroelasticProperties,AddContactMaterial,CoulombFriction,ContactModel,DiscreteContactApproximation,ExternallyAppliedSpatialForce,SpatialForce,Simulator,CollisionFilterDeclaration,GeometrySet)
from .geometry import ART

def run(dt=.00025,segments=48,passive=False,load=.15,duration=1.):
 model=ART/('precision-slotted.xml'if segments==48 else f'precision-slotted-s{segments}.xml')
 root=E.parse(model).getroot();m=mujoco.MjModel.from_xml_path(str(model));bid=m.body('screw').id;mass=m.body_mass[bid];R=np.empty(9);mujoco.mju_quat2Mat(R,m.body_iquat[bid]);R=R.reshape(3,3);I=R@np.diag(m.body_inertia[bid])@R.T/mass
 b=DiagramBuilder();p,sg=AddMultibodyPlantSceneGraph(b,dt);screw=p.AddRigidBody('screw',SpatialInertia(mass,m.body_ipos[bid],UnitInertia(I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2])))
 prop=ProximityProperties();AddRigidHydroelasticProperties(.002,prop);AddContactMaterial(friction=CoulombFriction(.15,.15),properties=prop);prop.AddProperty('material','relaxation_time',.001);prop.AddProperty('material','point_contact_stiffness',1e8)
 ids={'male':[],'female':[]}
 for asset in root.findall('asset/mesh'):
  name=asset.get('name','');kind='male'if name.startswith('seq_male_thread')else'female'if name.startswith('seq_female_thread')else None
  if kind:ids[kind].append(p.RegisterCollisionGeometry(screw if kind=='male'else p.world_body(),RigidTransform(),Convex(asset.get('file')),name,prop))
 ids['male'].append(p.RegisterCollisionGeometry(screw,RigidTransform([0,0,-.0038]),Cylinder(.00118,.0076),'core',prop))
 declaration=CollisionFilterDeclaration()
 for group in ids.values():declaration.ExcludeWithin(GeometrySet(group))
 sg.collision_filter_manager().Apply(declaration);p.set_sap_near_rigid_threshold(0);p.set_contact_model(ContactModel.kHydroelasticWithFallback);p.set_discrete_contact_approximation(DiscreteContactApproximation.kSap);p.Finalize()
 diagram=b.Build();ctx=diagram.CreateDefaultContext();pc=p.GetMyMutableContextFromRoot(ctx);sim=Simulator(diagram,ctx);w=ExternallyAppliedSpatialForce();w.body_index=screw.index();w.p_BoBq_B=np.zeros(3);port=p.get_applied_spatial_force_input_port();rows=[];previous=0;angle=0;start=time.monotonic()
 for k in range(round(duration/dt)):
  V=screw.EvalSpatialVelocityInWorld(pc);tau=np.clip(-.000025+.00002*(-np.pi/2-V.rotational()[2]),-.0002,.0002)if k*dt>.1 and not passive else 0
  w.F_Bq_W=SpatialForce([0,0,tau],[0,0,-load*min(1,k*dt/.1)]);port.FixValue(pc,[w]);sim.AdvanceTo((k+1)*dt);X=screw.EvalPoseInWorld(pc);R=X.rotation().matrix();yaw=np.arctan2(R[1,0],R[0,0]);angle-=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous));previous=yaw
  if k%10==0:rows.append([ctx.get_time(),np.degrees(angle),-X.translation()[2]*1000,np.degrees(np.arccos(np.clip(R[2,2],-1,1)))])
 rows=np.array(rows);error=rows[:,2]-rows[:,1]*.5/360
 result=dict(scope='Independent thread bench; external wrench ONLY here, not in hand tasks',dt=dt,mass_kg=mass,pitch_mm=.5,load_N=load,passive=passive,torque_controller='zero torque'if passive else'clip(-0.000025+0.00002*(-pi/2-omega_z), +/-0.0002) Nm after 0.1 s',rotation_deg=float(rows[-1,1]),depth_mm=float(rows[-1,2]),pitch_residual_mm=[float(error.min()),float(error.max())],max_tilt_deg=float(rows[:,3].max()),wall_s=time.monotonic()-start,trace=rows.tolist())
 (ART/(f'thread-bench-passive-load{load}-s{segments}-dt{dt}.json'if passive else f'thread-bench-slow-s{segments}-dt{dt}.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='trace'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.00025);p.add_argument('--segments',type=int,default=48);p.add_argument('--passive',action='store_true');p.add_argument('--load',type=float,default=.15);p.add_argument('--duration',type=float,default=1.);a=p.parse_args();run(a.dt,a.segments,a.passive,a.load,a.duration)
