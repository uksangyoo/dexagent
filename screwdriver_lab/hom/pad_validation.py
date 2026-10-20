"""Hydroelastic pressure convergence using the actual remeshed Tesollo pad.

An independent quasistatic geometry test, not a scripted task trajectory.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json,numpy as np
from pydrake.all import (DiagramBuilder,AddMultibodyPlantSceneGraph,SpatialInertia,UnitInertia,ProximityProperties,AddCompliantHydroelasticProperties,AddRigidHydroelasticProperties,AddContactMaterial,CoulombFriction,RigidTransform,Mesh,HalfSpace,ContactModel,HydroelasticContactRepresentation,RotationMatrix)
from screwdriver_lab.hom.model import OUT
from screwdriver_lab.hom.volume_mesh import tetrahedral_pad

def test(h,indentations):
    source=OUT/'meshes/5d9ee0d505_link_1_tip.obj';path=tetrahedral_pad(source,h)
    b=DiagramBuilder();p,sg=AddMultibodyPlantSceneGraph(b,.001)
    pad=p.AddRigidBody('pad',SpatialInertia(.03,[0,0,0],UnitInertia.SolidSphere(.01)))
    soft=ProximityProperties();AddCompliantHydroelasticProperties(h,1.7e6,soft);AddContactMaterial(friction=CoulombFriction(.6,.6),properties=soft)
    hard=ProximityProperties();AddRigidHydroelasticProperties(h,hard);AddContactMaterial(friction=CoulombFriction(.6,.6),properties=hard)
    p.RegisterCollisionGeometry(pad,RigidTransform(),Mesh(str(path)),'pad',soft)
    # Pad nose points +x; the rigid halfspace occupies x >= max_x-delta.
    plane=p.RegisterCollisionGeometry(p.world_body(),RigidTransform(RotationMatrix.MakeYRotation(-np.pi/2),[.01790028,0,0]),HalfSpace(),'indenter',hard)
    p.set_contact_model(ContactModel.kHydroelastic);p.Finalize();diagram=b.Build();c=diagram.CreateDefaultContext();pc=p.GetMyMutableContextFromRoot(c);records=[]
    for delta in indentations:
        p.SetFreeBodyPose(pc,pad,RigidTransform([delta,0,0]));query=sg.get_query_output_port().Eval(sg.GetMyContextFromRoot(c));F=0.;A=0.;N=0
        for surface in query.ComputeContactSurfaces(HydroelasticContactRepresentation.kTriangle):
            mesh=surface.tri_mesh_W();field=surface.tri_e_MN()
            for i in range(mesh.num_triangles()):
                area=mesh.area(i);pressure=field.EvaluateCartesian(i,mesh.element_centroid(i));F+=area*pressure;A+=area;N+=1
        records.append(dict(resolution_m=h,indentation_m=delta,force_N=F,area_mm2=A*1e6,triangles=N))
    return records
if __name__=='__main__':
    records=[]
    for h in [.002,.001]:
        rows=test(h,[.00005,.0001,.0002,.0004]);records+=rows;print(json.dumps(rows),flush=True)
    (OUT/'pad-pressure-convergence.json').write_text(json.dumps(dict(geometry='Actual Tesollo pad against plane at distal nose',hardware_calibrated=False,records=records),indent=2))
