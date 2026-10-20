"""Independent Newton contact checks; no task controller or success-based tuning.

A free, uniformly massive 12 x 12 x 6 mm coupon rests on a fixed platen.
Gravity supplies normal load. A declared external test load probes static and
sliding friction. These are numerical checks, not measured Tesollo materials.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
import newton
from newton.geometry import HydroelasticSDF

OUT = Path(__file__).parent / 'artifacts/physics_validation'
SOURCE_BYTES = Path(__file__).read_bytes()


@wp.kernel
def test_load(f: wp.array(dtype=wp.spatial_vector), force: float, torque: float):
    f[0] = wp.spatial_vector(force, 0.0, 0.0, 0.0, 0.0, torque)


def run(mode='hydro', resolution=64, dt=.001, mass=.1, kh=1e8, platen_ratio=1., gap=0., loading="slide", buffer_mult=8, duration=1.0):
    # Compatibility alias for this installed Warp / Newton pair, no math change.
    try:
        import warp._src.marching_cubes as mc
    except ModuleNotFoundError:
        mc = None  # Newer Newton imports the public Warp implementation.
    if mc is not None and not hasattr(mc, 'mc_cube_corner_offsets'):
        mc.mc_cube_corner_offsets = mc.MC_CUBE_CORNER_OFFSETS
    wp.init()
    wp.set_device('cuda:0')
    assert wp.get_device().pci_bus_id.upper() == '00000000:61:00', wp.get_device().pci_bus_id
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    cfg = b.ShapeConfig(density=0, mu=.6, ke=1e4, kd=100,
                        mu_torsional=0, mu_rolling=0, margin=0, gap=gap)
    if mode == 'hydro':
        cfg.configure_sdf(max_resolution=resolution, is_hydroelastic=True, kh=kh)
        cfg.sdf_narrow_band_range = (-.003, .003)
    half = np.array([.006, .006, .003])
    inertia = mass / 3 * np.array([half[1]**2 + half[2]**2,
                                  half[0]**2 + half[2]**2,
                                  half[0]**2 + half[1]**2])
    body = b.add_body(xform=wp.transform((0, 0, .0031), wp.quat_identity()),
                      mass=mass, inertia=wp.mat33(np.diag(inertia)), lock_inertia=True,
                      label='coupon')
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label='coupon')
    platen_cfg = cfg.copy()
    platen_cfg.kh = kh*platen_ratio
    b.add_shape_box(-1, xform=wp.transform((0, 0, -.005), wp.quat_identity()),
                    hx=.03, hy=.02, hz=.005, cfg=platen_cfg, label='platen')
    model = b.finalize(device='cuda:0')
    s, nxt, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, s.joint_q, s.joint_qd, s)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=mode == 'native',
            iterations=200, ls_iterations=50, cone='elliptic', solver='newton',
            integrator='implicitfast', njmax=4096, nconmax=1024)
    pipeline = None if mode == 'native' else newton.CollisionPipeline(model,
            broad_phase='explicit', rigid_contact_max=1024,
            sdf_hydroelastic_config=HydroelasticSDF.Config(buffer_fraction=1.0, buffer_mult_broad=buffer_mult, buffer_mult_iso=buffer_mult))
    contacts = pipeline.contacts() if pipeline else None
    solved = newton.Contacts(1024, 0, device='cuda:0', requested_attributes={'force'})
    records = []
    max_contacts = 0
    for step in range(round(duration/dt)):
        t = step*dt
        ratio = 0 if t < .35 or t >= .60 else .5 if t < .55 else 1.5
        load = ratio*.6*mass*9.81
        force = load if loading=='slide' else 0.
        mean_radius = .006/3*(np.sqrt(2)+np.log(1+np.sqrt(2)))
        torque = load*mean_radius if loading=='twist' else 0.
        previous_vz = float(s.body_qd.numpy()[0,2])
        s.clear_forces()
        wp.launch(test_load, 1, [s.body_f, force, torque], device='cuda:0')
        if pipeline:
            pipeline.collide(s, contacts)
        solver.step(s, nxt, ctrl, contacts, dt)
        solver.update_contacts(solved)
        n = int(solved.rigid_contact_count.numpy()[0])
        max_contacts = max(max_contacts, n)
        if n >= 1024:
            raise RuntimeError('Contact capacity exhausted: invalid trial')
        if pipeline and mode == 'hydro':
            h = pipeline.hydroelastic_sdf
            for count, cap in zip(h.iso_buffer_counts, [h.max_num_blocks_broad, *h.iso_max_dims]):
                if int(count.numpy()[0]) > cap or (cap > 0 and int(count.numpy()[0]) == cap):
                    raise RuntimeError(f'Hydro internal buffer exhausted: count={int(count.numpy()[0])}, capacity={cap}')
            if int(h.contact_reduction.contact_count.numpy()[0]) >= h.max_num_face_contacts:
                raise RuntimeError('Hydro face buffer exhausted: invalid trial')
        pose = nxt.body_q.numpy()[0]
        vel = nxt.body_qd.numpy()[0]
        f = solved.force.numpy()[:n, :3]
        normals = solved.rigid_contact_normal.numpy()[:n]
        normal = float(np.sum(np.abs(np.einsum('ij,ij->i', f, normals))))
        ids0 = solved.rigid_contact_shape0.numpy()[:n]
        sign = np.where(model.shape_body.numpy()[ids0]==body, 1., -1.)
        net_z = float(np.sum(sign*f[:,2]))
        balance_error = net_z-mass*(9.81+(float(vel[2])-previous_vz)/dt)
        if not np.all(np.isfinite(np.r_[pose, vel, normal])):
            raise RuntimeError('Nonfinite dynamics')
        records.append([t+dt, *pose, *vel, force, normal, n, torque, net_z, balance_error])
        s, nxt = nxt, s
    a = np.asarray(records)
    def window(lo, hi):
        return a[(a[:,0]>=lo) & (a[:,0]<hi)]
    rest, sub, slide, end = window(.25,.35), window(.45,.55), window(.58,.60), window(.95,1.01)
    # Column layout: t, xyz, quaternion xyzw, linear velocity xyz, angular xyz,
    # applied Fx, sum positive contact normal, contact count.
    result = dict(mode=mode, resolution=resolution if mode=='hydro' else None,
        dt_s=dt, mass_kg=mass, kh_N_m3=kh if mode=='hydro' else None, gap_m=gap, loading=loading,
        solver_iterations=200, linesearch_iterations=50, buffer_mult=buffer_mult,
        platen_stiffness_ratio=platen_ratio, mu=.6, ke_N_m=1e4, kd_Ns_m=100, normal_load_N=mass*9.81,
        vertical_force_balance_rms_N=float(np.sqrt(np.mean(a[:,19]**2))),
        settled_vertical_speed_rms_mm_s=float(np.sqrt(np.mean(rest[:,10]**2))*1000),
        subthreshold_angular_speed_rad_s=float(np.max(abs(sub[:,13]))),
        driven_angular_speed_rad_s=float(slide[:,13].mean()),
        final_angular_speed_rad_s=float(end[:,13].mean()),
        settled_indentation_mm=float((.003-rest[:,3].mean())*1000),
        normal_force_relative_error=float(abs(rest[:,18].mean()-mass*9.81)/(mass*9.81)),
        subthreshold_displacement_mm=float((sub[-1,1]-rest[-1,1])*1000),
        subthreshold_speed_mm_s=float(np.max(abs(sub[:,8]))*1000),
        sliding_speed_mm_s=float(slide[:,8].mean()*1000),
        final_speed_mm_s=float(end[:,8].mean()*1000), max_contacts=max_contacts,
        analytic_flat_patch_indentation_mm=(1+1/platen_ratio)*mass*9.81/(kh*.012*.012)*1000 if mode=='hydro' else None,
        notes=['Flat coupon is a numerical test fixture, not the Tesollo pad.',
               'Zero extra point torsion/rolling friction; hydro patch torque arises from distributed friction.',
               'All material parameters are declared uncalibrated assumptions.'])
    return result, a


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['native','point','hydro'], default='hydro')
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--mass', type=float, default=.1)
    p.add_argument('--kh', type=float, default=1e8)
    p.add_argument('--platen-ratio', type=float, default=1.)
    p.add_argument('--gap', type=float, default=0.)
    p.add_argument('--loading', choices=['slide','twist'], default='slide')
    p.add_argument('--buffer-mult', type=int, default=8)
    args = p.parse_args()
    root=Path(newton.__file__).parents[1]
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    OUT=OUT/commit[:12]
    OUT.mkdir(parents=True, exist_ok=True)
    name=f'{args.mode}-r{args.resolution}-dt{args.dt:g}-m{args.mass:g}-kh{args.kh:g}-ratio{args.platen_ratio:g}-gap{args.gap:g}-{args.loading}'
    result, trace = run(**vars(args))
    result.update(newton_commit=commit,
        source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),
        warp_version=wp.__version__,
        gpu_physical_index=1, gpu_pci=wp.get_device().pci_bus_id)
    (OUT/(result['source_sha256']+'_contact_bench.py')).write_bytes(SOURCE_BYTES)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/(name+'.npz'),trace=trace,
        columns=['t','x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz','force_x','normal_force','contact_count','torque_z','contact_force_z','force_balance_error'])
    print(json.dumps(result),flush=True)
