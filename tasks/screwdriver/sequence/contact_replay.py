"""Measure contact-relative motion from immutable recorded states.

This is geometry/kinematics replay, not a new rollout: the simulator is never
advanced. Hydroelastic patch centroids are queried at each saved physical pose.
Original measured forces remain in the original recording, not reconstructed.
"""
import argparse,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np,mujoco
from pydrake.all import HydroelasticContactRepresentation
from dexagent.sim.drake_backend import HydroPlant
from dexagent.sim.model import OUT


def measure(name):
 path=OUT/(name+'.npz');a=np.load(path);report=json.loads((OUT/(name+'.json')).read_text())
 h=HydroPlant(OUT/(name+'.xml'),dt=report['parameters']['dt'],resolution=.002,explicit_steel=True,beta=0)
 m=h.m;tool=m.body('screwdriver').id;fingers=[m.body(f'link_{i}_4').id for i in [1,3]];rows=[]
 for t,q,v in zip(a['time'],a['qpos'],a['qvel']):
  h.initialize_from_mujoco(q,v);h.sync();d=h.d
  query=h.scene.get_query_output_port().Eval(h.scene.GetMyContextFromRoot(h.context));ins=query.inspector()
  surfaces,_=query.ComputeContactSurfacesWithFallback(HydroelasticContactRepresentation.kTriangle)
  axis=d.xmat[tool].reshape(3,3)[:,2];center=d.xpos[tool];contacts=[]
  for finger,bid in zip([1,3],fingers):
   found=[]
   for surface in surfaces:
    names={ins.GetName(g).split('::')[-1]for g in [surface.id_M(),surface.id_N()]}
    if names=={'handle',f'finger_tip_{finger}'}:
     point=surface.centroid();radial=point-center;normal=radial-axis*np.dot(radial,axis);normal/=np.linalg.norm(normal);tangent=-np.cross(axis,normal)
     J=np.zeros((3,m.nv));K=J.copy();Jr=J.copy();Kr=J.copy();mujoco.mj_jac(m,d,J,Jr,point,bid);mujoco.mj_jac(m,d,K,Kr,point,tool)
     relative_v=(J-K)@d.qvel;relative_w=(Jr-Kr)@d.qvel
     found.append(dict(point_world_m=point.tolist(),circumferential_slip_mm_s=float(1000*tangent@relative_v),axial_slip_mm_s=float(1000*axis@relative_v),normal_spin_deg_s=float(np.degrees(normal@relative_w)),relative_angular_speed_deg_s=float(np.degrees(np.linalg.norm(relative_w)))))
   contacts.append(found)
  rows.append(dict(t=float(t),thumb=contacts[0],middle=contacts[1]))
 result=dict(scope='Geometry and kinematics replay of recorded states; no simulation advance, object motion commands, or reconstructed force claims',source_recording_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_xml_sha256=hashlib.sha256((OUT/(name+'.xml')).read_bytes()).hexdigest(),sampling_hz=50,interpretation='Centroid linear slip and relative normal-axis spin are distinct. During a gravity pivot the latter can dominate. Empty contact arrays mean no reconstructed hydroelastic patch at that saved pose.',frames=rows)
 output=OUT/(name+'-contact-motion.json');output.write_text(json.dumps(result,separators=(',',':')));print(output)
 return output

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();measure(a.name)
