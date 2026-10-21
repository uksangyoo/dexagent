"""Check the prepared scene before accepting any manipulation score."""
import json
import numpy as np
from pydrake.all import HydroelasticContactRepresentation
from dexagent.sim.model import prepare,OUT
from dexagent.sim.initialization import fit_grasp
from dexagent.sim.drake_backend import HydroPlant


def audit():
    xml=prepare(True,48);q,initial=fit_grasp(xml,tilt_deg=0)
    h=HydroPlant(xml,dt=.00025,resolution=.002,explicit_steel=True,beta=0)
    h.initialize_from_mujoco(q);h.sync()
    query=h.scene.get_query_output_port().Eval(h.scene.GetMyContextFromRoot(h.context));ins=query.inspector()
    surfaces,pairs=query.ComputeContactSurfacesWithFallback(HydroelasticContactRepresentation.kTriangle)
    entries=[]
    for pair in pairs:
        entries.append(dict(names=[ins.GetName(g).split('::')[-1]for g in [pair.id_A,pair.id_B]],penetration_m=float(pair.depth)))
    same_body=[]
    for s in surfaces:
        names=[ins.GetName(g).split('::')[-1]for g in [s.id_M(),s.id_N()]]
        bids=[h.m.geom_bodyid[h.m.geom(name).id]for name in names]
        if h.m.body_weldid[bids[0]]==h.m.body_weldid[bids[1]]:same_body.append(names)
    audit=h.audit()
    result=dict(initialization=initial,initial_hydro_surfaces=len(surfaces),initial_point_pairs=entries,
                largest_initial_penetration_m=max([r['penetration_m']for r in entries]+[0]),
                redundant_rigid_assembly_hydro_pairs=same_body,
                fingertip_tool_initial_overlap=any(set([ins.GetName(g).split('::')[-1]for g in [s.id_M(),s.id_N()]])&{'handle','cap'}for s in surfaces),
                wrist_collision='Convex envelope of supplied wrist3 CAD, replacing the capsule that extended beyond the flange',
                unactuated_objects=['screwdriver','screw'],physical_actuator_count=len(h.qi),conversion_audit=audit)
    assert not same_body
    assert not result['fingertip_tool_initial_overlap']
    assert result['largest_initial_penetration_m']<1e-6,entries
    (OUT/'corrected-scene-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items()if k not in ['initialization','conversion_audit']},indent=2))


if __name__=='__main__':audit()
