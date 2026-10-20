"""Check the generated pressure law independently of the dynamics solver.

Each prescribed pose is a separate static geometry query, not a task rollout.
For two equal linear materials and a flat patch, F = A kh delta / 2.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import newton
import warp as wp
from newton.geometry import HydroelasticSDF

def main():
    wp.init()
    wp.set_device('cuda:0')
    assert wp.get_device().pci_bus_id == '00000000:61:00'
    rows = []
    for resolution in [32,64,128]:
        b = newton.ModelBuilder()
        cfg = b.ShapeConfig(density=1000, mu=.6, ke=1e4, kd=100, gap=0, margin=0,
                            mu_torsional=0, mu_rolling=0)
        cfg.configure_sdf(max_resolution=resolution, is_hydroelastic=True, kh=1e8)
        cfg.sdf_narrow_band_range=(-.003,.003)
        body = b.add_body(xform=wp.transform((0,0,.003),wp.quat_identity()))
        b.add_shape_box(body,hx=.006,hy=.006,hz=.003,cfg=cfg)
        b.add_shape_box(-1,xform=wp.transform((0,0,-.005),wp.quat_identity()),hx=.03,hy=.02,hz=.005,cfg=cfg)
        m=b.finalize(device='cuda:0');s=m.state()
        pipeline=newton.CollisionPipeline(m,broad_phase='explicit',rigid_contact_max=2048,
            sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_fraction=1.,buffer_mult_broad=8,buffer_mult_iso=8))
        c=pipeline.contacts()
        for delta in [.00005,.0001,.0002,.0004]:
            pose=s.body_q.numpy();pose[body,:3]=[0,0,.003-delta];pose[body,3:]=[0,0,0,1];s.body_q.assign(pose)
            pipeline.collide(s,c)
            n=int(c.rigid_contact_count.numpy()[0]);assert n<2048
            sh0=c.rigid_contact_shape0.numpy()[:n];sh1=c.rigid_contact_shape1.numpy()[:n]
            sb=m.shape_body.numpy();p0=c.rigid_contact_point0.numpy()[:n];p1=c.rigid_contact_point1.numpy()[:n]
            p0=p0+np.where((sb[sh0]>=0)[:,None],pose[body,:3],0)
            p1=p1+np.where((sb[sh1]>=0)[:,None],pose[body,:3],0)
            normals=c.rigid_contact_normal.numpy()[:n];k=c.rigid_contact_stiffness.numpy()[:n]
            depth=np.maximum(0,-np.einsum('ij,ij->i',p1-p0,normals))
            sign=np.where(sb[sh0]==body,-1.,1.)
            force=float(np.sum(sign*normals[:,2]*k*depth));expected=.012*.012*1e8*delta/2
            rows.append(dict(resolution=resolution,indentation_mm=delta*1000,contacts=n,
                generated_force_N=force,analytic_force_N=expected,relative_error=abs(force-expected)/expected))
    result=dict(rows=rows,newton_commit=subprocess.check_output(['git','-C',str(Path(newton.__file__).parents[1]),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu_pci=wp.get_device().pci_bus_id,
        scope='Static collision/pressure integration only. Does not validate dynamics, friction, or actual pad material.')
    out=Path(__file__).parent/'artifacts/physics_validation/pressure-probe.json'
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
