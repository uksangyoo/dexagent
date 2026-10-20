"""Independent hydroelastic pressure tests, with analytic sphere/plane references.

Prescribed geometry is used only for this contact-law test, not in task runs.
The reference is the radial hydroelastic pressure law, not Hertz elasticity.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json
import numpy as np
from pydrake.all import *
from screwdriver_lab.hom.model import OUT

def static_sphere(resolution,indentation,R=.01,E=1.7e6):
    builder=DiagramBuilder();p,sg=AddMultibodyPlantSceneGraph(builder,.001)
    body=p.AddRigidBody('test_pad',SpatialInertia(.1,[0,0,0],UnitInertia.SolidSphere(R)))
    soft=ProximityProperties();AddCompliantHydroelasticProperties(resolution,E,soft);AddContactMaterial(friction=CoulombFriction(.6,.6),properties=soft)
    rigid=ProximityProperties();AddRigidHydroelasticProperties(resolution,rigid);AddContactMaterial(friction=CoulombFriction(.6,.6),properties=rigid)
    p.RegisterCollisionGeometry(body,RigidTransform(),Sphere(R),'sphere',soft)
    p.RegisterCollisionGeometry(p.world_body(),RigidTransform(),HalfSpace(),'plane',rigid)
    p.set_contact_model(ContactModel.kHydroelastic);p.Finalize();diagram=builder.Build();c=diagram.CreateDefaultContext();pc=p.GetMyMutableContextFromRoot(c)
    p.SetFreeBodyPose(pc,body,RigidTransform([0,0,R-indentation]));query=sg.get_query_output_port().Eval(sg.GetMyContextFromRoot(c))
    surfaces=query.ComputeContactSurfaces(HydroelasticContactRepresentation.kTriangle)
    force=0.;area=0.;torque_bound=0.;triangles=0
    for surface in surfaces:
        mesh=surface.tri_mesh_W();field=surface.tri_e_MN()
        for i in range(mesh.num_triangles()):
            a=mesh.area(i);center=mesh.element_centroid(i);pressure=field.EvaluateCartesian(i,center)
            force+=a*pressure;area+=a;torque_bound+=.6*a*pressure*np.linalg.norm(center[:2]);triangles+=1
    expected=np.pi*E*indentation**2*(1-2*indentation/(3*R))
    return dict(resolution_m=resolution,indentation_m=indentation,normal_force_N=force,analytic_force_N=expected,relative_force_error=abs(force-expected)/expected,patch_area_mm2=area*1e6,triangles=triangles,frictional_torque_bound_Nm=torque_bound)

def main():
    OUT.mkdir(exist_ok=True,parents=True);records=[]
    for h in [.004,.002,.001,.0005]:
        for d in [.0001,.0003,.0006]:
            r=static_sphere(h,d);records.append(r);print(r,flush=True)
    (OUT/'hydrostatic-validation.json').write_text(json.dumps(dict(reference='F=pi*E*delta^2*(1-2*delta/(3R)); radial hydroelastic sphere against a rigid plane',hardware_calibrated=False,records=records),indent=2))
if __name__=='__main__':main()
