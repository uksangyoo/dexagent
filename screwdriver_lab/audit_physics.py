"""Separate numerical task completion from physical-model validity."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'screwdriver_lab/artifacts/physics_validation'
MANUAL = 'https://cdn.tesollo.com/site/docs/2026/08/10a30190-9e90-4732-8254-378badcdc4b0.pdf'


def main():
    xml = ROOT/'screwdriver_lab/artifacts/full_motion/pivot-captured-long.xml'
    m = mujoco.MjModel.from_xml_path(str(xml))
    hand = [i for i in range(m.nbody) if m.body(i).name == 'palm' or m.body(i).name.startswith('link_')]
    issues = [
        dict(id='thread', observed='An equality enforces screw angle = 2π × depth / 1 mm; the screw has only axial/rotation joints.',
             consequence='Cannot validate thread engagement, cross-threading, or contact-generated helical motion.',
             required='Use explicit mating thread geometry and free screw motion for the realistic task; retain the ideal joint only as a labeled reduced-model control.'),
        dict(id='gravity', observed=f'{int(np.sum(m.body_gravcomp>0))} bodies have ideal gravity compensation.',
             consequence='Compensation bypasses actuator effort budgets.',
             required='Set body gravcomp to zero. Apply estimated compensation through the same torque/speed/current limits as control.'),
        dict(id='actuation', observed='Finger limits ±0.3 N m; wrench/velocity control updated at 1 kHz.',
             consequence='Exceeds the published 0.11 N m continuous rating and 500 Hz communication cycle; no thermal or torque-speed model.',
             required='500 Hz command updates, 0.11 N m continuous limits; model documented peak duration and torque-speed dependence before using peak torque.'),
        dict(id='material', observed='μ=0.9 and MuJoCo time constant 4 ms; hydro pilot used equal fingertip/tool kh=5e12 N/m³.',
             consequence='These were not identified from material data. Equal softness of silicone and handle is unjustified.',
             required='Use silicone pads and a much stiffer handle with separate parameters; validate normal force, dissipation, sliding and torsional resistance before task tuning.'),
        dict(id='geometry', observed='24 mm diameter synthetic handle, ideal cap bearing, engineered pickup supports, finite friction patches.',
             consequence='A strategy prototype, not a reconstruction of the reference tool or test fixture.',
             required='Derive tool size ratios from the reference, retain uncertainty in absolute scale, use material densities, and audit real pad/handle collision surfaces.'),
        dict(id='collision', observed='Native convex collision detection is disabled; fallback libccd remains enabled.',
             consequence='This is a collision-algorithm switch, not continuous collision detection. Neither path has been validated on all task pairs.',
             required='Validate bit/slot and pad/handle signed distances and contacts against geometry; never suppress a physical collision to complete the task.'),
        dict(id='sensors', observed='Exact simulator contact wrenches and tool pose are available to the controller.',
             consequence='The result assumes sensing beyond the standard hand; tactile sensors are optional.',
             required='Declare the observation contract; evaluate noise, delay and contact-estimation errors separately.'),
    ]
    profile = dict(status='physical_validation_required', prior_task_completion_is_physical_validation=False,
        prior_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        manufacturer=dict(source=MANUAL, page=12, model='DG-5F-S 20 DoF',
            pad_material='silicone', pad_hardness_shore_A=40, hand_mass_kg=.880,
            rated_joint_torque_Nm=.11, stall_joint_torque_Nm=.65,
            no_load_joint_speed_rpm=85, control_cycle_Hz=500,
            note='Hardness does not determine friction, damping, or a unique pressure-depth law.'),
        tool_analogue=dict(model='Wiha PicoFinish 42391 / 260P slotted', source='https://wiha.com/au/en/tools/screwdrivers/precision-screwdrivers/picofinish/slotted/picofinish-fine-screwdriver/42391', overall_length_mm=150, blade_length_mm=50, blade_diameter_mm=2.5, bit_width_mm=2.5, bit_thickness_mm=.4, maximum_handle_diameter_mm=18, published_weight_g=16, identity='Documented commercial analogue; not an identification of the tool in the video.'),
        model_audit=dict(hand_mass_kg=sum(float(m.body_mass[i]) for i in hand),
            tool_and_cap_mass_kg=float(m.body_mass[m.body('screwdriver').id]+m.body_mass[m.body('swivel_cap').id]),
            gravity_compensated_bodies=int(np.sum(m.body_gravcomp>0)),
            equalities=[m.equality(i).name for i in range(m.neq)]),
        issues=issues,
        contact_candidates=[
            dict(name='Native MuJoCo contact through Newton', purpose='Rigid contact baseline; test friction regularization and manifold behavior.'),
            dict(name='Newton point/manifold contact + MuJoCo-Warp', purpose='Separate collision pipeline effects from solver effects.'),
            dict(name='Newton SDF hydroelastic + MuJoCo-Warp', purpose='Distributed compliant pad patches; finite-patch torsion from resolved tangential forces, no extra point torsion by default.'),
            dict(name='Deformable pad model', purpose='Escalation if large strain or hysteresis cannot be represented by a calibrated hydroelastic law; not yet implemented in this study.')],
        material_uncertainty=dict(kh_N_m3=[1e8,1e9,1e10], mu=[.3,.6,1.0],
            note='Exploratory ranges, not a fit to the DG-5F-S. Never select by task success. Pad backing and thickness must be accounted for.'),
        acceptance=dict(load_balance_relative_error_max=.01,
            timestep_and_sdf_refinement_relative_change_max=.05,
            no_contact_buffer_overflow=True, no_nonfinite_states=True,
            no_tunneling=True, sliding_friction_matches_declared_law=True,
            contact_torque_and_passivity_checks_required=True,
            measurement_based_material_validation=False),
        next_controller_protocol='Freeze the validated physical profile first, then compare controller abstractions with identical geometry/materials/actuation/initial conditions; preserve all failures.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'physics-audit.json').write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile['model_audit']))


if __name__ == '__main__':
    main()
