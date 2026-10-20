# UR5e–Tesollo full screwdriver task

This study implements a manually authored Graph-as-Policy-style task graph over an articulated UR5e and a 20-joint Tesollo DG5F-S right hand. It follows the function of the supplied reference: acquire a horizontal screwdriver, lift and reorient it, engage the bit, transfer to a thumb-supported driving grasp, execute repeated turns, and verify seating. It does not recover human joints or run an LLM in the servo loop.

The live website compares five hand interfaces and five paired conditions. It retains failures as well as successful runs. One deterministic run per cell is a controlled engineering comparison, not a statistical success-rate estimate. The results characterize these implementations, not the theoretical capacity of each abstraction.

## Policy structure

`graph/workflow.json` uses the local Graph-as-Policy project's general structure: modular skill subgraphs, typed observation/interface inputs, explicit exits, conditional routing, and backward edges. Its source reference is commit `c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea`. `graph_policy.py` is an independent executor; the GaP harness is not used, and compatibility with its parser is not claimed.

The executed sequence is approach → grasp → lift → reorient → align → press → acquire cap → reposition → turn/renew → verify seat. Each completion exit records the observations that justified it. Failed guards exit to abort. A reacquisition edge exists in the graph schema but is not exercised by this benchmark; automatic retry recovery is future work. The executor uses the top-level conditional edges; the nested skill-loop descriptions are implemented as Python control branches in `full_motion.py`, with hand commands dispatched through `abstractions.py`.

All five interfaces share graph sequencing, arm feedback, motor limits, impedance scheduling, and phase completion checks:

| Interface | Hand command |
|---|---|
| Contact skills | Current contact roles, force targets, geometric contact points, and measured turn/release/regrasp exits |
| Joint trajectories | Nominal per-node 20-joint trajectory replay |
| Motor synergies | Rank-3 PCA projection of that same trajectory |
| Fingertip poses | Nominal three fingertip poses in the palm frame, tracked through current joint IK |
| Object motion | The same grasp calibration plus bounded tool-pose error correction |

The calibration comes from the successful nominal simulated contact rollout, not the human video. Nominal replay is therefore an in-distribution reconstruction test. If driving outlasts its nominal calibration, the replay-based interfaces repeat the last complete driving cycle rather than freezing at the final joint target. The perturbations test transfer from that calibration. Arm feedback remains live for all methods; “joint trajectories” does not mean the entire robot is open loop. Synergy variance retained is reported in `abstraction-bank.json`.

## Physical model and scope

- 26 robot actuators: 20 finger joints and 6 actual UR5e joints. No Cartesian arm stage remains in the evaluated embodiment. The functional grasp uses thumb, index and middle fingers; ring and little fingers have no assigned tool-contact role.
- Free screwdriver and passive swivel cap. No tool weld, injected tool torque, or tool pose reset after time zero.
- Horizontal tool starts on two visible narrow cradle supports, with the hand open above it.
- Authored 24 mm diameter handle, simplified cruciform bit/socket, 1 mm screw pitch, and 2 mm insertion travel. These dimensions and head geometry are not measured from the human video.
- Screw rotation and translation are coupled by an ideal helical constraint. Bit/socket and screw-head/workpiece interactions are physical contacts. The screw is pre-threaded: no cutting or cross-threading is modeled.
- Full-task sweep: MuJoCo 3.10.1, 1 ms dynamics, 50 Hz control, elliptic finite-patch friction (`condim=6`), convex hand collision. Shared contact tangential μ=0.9 nominal, torsional coefficient 0.006 m, rolling coefficient 0.00002 m.
- Finger position gains change from 8 N·m/rad for pickup to 2 N·m/rad for driving, with damping 0.18 and 0.08 respectively. Finger torque limit is ±0.3 N·m. Arm gains and limits are in `arm_scene.py`.
- Ideal gravity compensation and privileged simulator pose/contact observations. This is not a hardware controller certification.
- Full-task guards: per-finger load >12 N, penetration >0.6 mm, unintended finger/shaft load >0.3 N, and driving alignment bounds. A failed terminal sample can exceed the guard because it triggers the stop; it is retained.
- Success requires depth ≥1.98 mm, engaged bit, screw speed <0.15 rad/s, 0.15–5 N cap load, and >0.1 N seating contact held for 0.4 s. Depth alone is insufficient.

The five conditions are nominal, +1.5 mm lateral pickup misregistration, +20° initial axial tool roll, shared contact μ=0.55 (not a handle-only material change), and doubled thread resistance (0.8 → 1.6 mN·m). The pickup perturbation is deliberately not corrected before approach; it tests transfer from the nominal pickup plan. The shared arm controller uses measured tool pose after grasp acquisition.

## Hydroelastic ablation

Separate 15 s drive-only trials start from the same aligned hand/arm/tool grasp. They cross two contact models with release/regrasp and lightly loaded alternating return. They exclude pickup, reorientation, and grasp transfer, so they are not full-task success claims. Contact model and numerical solver backend change together; the experiment does not isolate hydroelasticity from solver/precision differences.

The hydroelastic backend uses Newton SDF contact patches for fingertips, handle and cap, and solved MuJoCo-Warp contact forces. Other contacts use convex geometry. SDF resolution is 64 and the hydroelastic gradient parameter is 5×10¹⁰ Pa/m; these are simulation choices, not measured Tesollo elastomer parameters. The local Newton commit is `b24dc255392126fe42a609ea6c4e4c8dca8009cc`, with Warp `1.16.0.dev20260720`. A process-local compatibility alias handles Warp's marching-cubes constant rename.

All hydroelastic substeps execute on physical GPU 1. Twenty 1 ms steps are captured per CUDA graph; force, penetration and all internal contact-buffer capacities are checked every substep. State/force observations return at 50 Hz. A guard stops at the end of the current 20 ms CUDA batch, so terminal overshoot can be much larger than the threshold; the threshold is not a force clamp. The hydroelastic deformation guard is 1 mm rather than the baseline's 0.6 mm; compare the measured penetration values, not only pass/fail status. Capacity overflow invalidates a run. Earlier overflowed development trials are excluded from the paired dataset.

In the paired drive-only ablation, “loaded return slip” means a return-mode finger sample with ≥0.02 N handle-contact load and normal-force-weighted tangential handle-contact speed >2 mm/s, divided by all return-mode finger samples. Both backends exclude individual contact contributions ≤0.001 N. The full-task sweep retains an unweighted handle-contact slip diagnostic and is not pooled with this ablation. It does not imply productive screw rotation. Screw advance, backward rotation, forces and penetration are also retained. Hydroelastic patches alone do not solve grasp planning or prevent a returning finger from undoing progress.

## Reproduction

From the extracted repository root, install `screwdriver_lab/requirements.txt` for CPU simulation and the render-queue helpers. The optional hydroelastic backend additionally requires the Newton commit and compatible Warp development build listed above; these third-party packages are not bundled. RTX rendering requires a separate Blender `bpy==4.5.10` environment, FFmpeg and a compatible NVIDIA OptiX driver. Arm and right-hand mesh assets, licenses, nominal grasp calibration, graph, and controller calibration are included. Extract every trace pack into the same directory for full recorded data. Videos have direct download links; optional duplicate offline media ZIPs can be generated with `python -m screwdriver_lab.package_full --media-archives` but are omitted from hosting to stay within its archive limit. The full-task dynamics run on CPU; GPU workloads exclusively select physical GPU 1 on this workstation.

```bash
# Reconstruct the calibration if desired; requires the nominal full trace pack.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.abstractions nominal-contact

# Paired sweep. Existing JSON records are retained; move them aside to rerun.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.run_full

# A new single full-task run.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.full_motion \
  --arm --mode regrasp --abstraction contact --timeout 190 --name new-contact

# Paired drive-only experiments, 15 s each.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.drive_experiment --mode regrasp --timeout 15
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.drive_experiment --mode slide --timeout 15
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.drive_experiment --hydro --mode regrasp --timeout 15
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.drive_experiment --hydro --mode slide --timeout 15
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.drive_experiment --mode hold --timeout 15

python -m screwdriver_lab.analyze_full
python -m screwdriver_lab.export_full nominal-contact --scale 6
CUDA_VISIBLE_DEVICES=1 /path/to/bpy-python -m screwdriver_lab.render_full \
  --bundle screwdriver_lab/artifacts/full_motion/rtx/nominal-contact \
  --output screwdriver_lab/artifacts/full_motion/rtx/nominal-contact.mp4
```

The RTX renderer uses Blender 4.5.10 Cycles, NVIDIA OptiX, 24 samples with OptiX denoising, 1280×960, and recorded geometry/transforms. No physics pose is corrected in rendering. Display-only changes are PBR materials, studio lights, cameras, a floor, and a pedestal under the fixed arm base. Videos identify simulation time and their time compression. The website compensates its playback speed and seeking for that compression. An extra second holds the final frame; it does not extend the simulated trial.

`analyze_full.py` checks actual robot actuators, free-tool topology, initial hand clearance, paired state hashes, pitch consistency, successful seating tails, and a pressure-only falsification. Finger-force and penetration peaks are sampled every physics step; arm torque, tracking-error and speed summaries use the recorded frames. The legacy `tool_angle_deg` field is unwrapped world yaw and becomes ill-conditioned for the initial horizontal tool; use `screw_angle_deg` for driving progress. Full records retain 30 Hz poses/observations and 50 Hz commands; hydroelastic observations are sampled from its 50 Hz outputs. The website uses a smaller time-series sample, with the final observation retained. Source hashes and rendering provenance are downloadable.

## Recorded outcome matrix

Budget: 190 s. Each cell is one deterministic run; values are final insertion depths in millimeters.

| Interface | Nominal | Pickup +1.5 mm | Tool roll +20° | Contact μ = 0.55 | 2× thread resistance |
|---|---|---|---|---|---|
| Contact skills | 2.001; seated | 0.000; penetration | 2.001; seated | 0.116; force_limit | 1.429; timeout |
| Joint trajectories | 2.001; seated | 0.001; cap_acquisition_failed | 2.002; seated | 1.902; timeout | 1.857; timeout |
| Motor synergies | -0.026; handoff_failed | -0.088; penetration | -0.029; handoff_failed | -0.012; penetration | -0.027; handoff_failed |
| Fingertip poses | 2.001; seated | 0.001; cap_acquisition_failed | 2.001; seated | 1.898; timeout | 1.853; timeout |
| Object motion | 2.002; seated | -0.126; handoff_failed | 1.725; timeout | 0.010; cap_acquisition_failed | 2.003; seated |

Eight of 25 runs meet the complete seating criterion. Each of joint replay, fingertip poses, object feedback and contact skills completes two conditions; rank-3 synergies complete none. The identities differ: object feedback completes doubled resistance while contact, joint and fingertip interfaces complete the initial-roll perturbation. Neither pickup misregistration nor reduced shared friction is solved within the budget.

The nominal contact run seats in 145.08 s, with peak finger load 5.682 N and peak contact penetration 0.076 mm. Initial hand clearance is 24.368 mm from the handle, 50.439 mm from the shaft, and 23.164 mm from the cap. Minor initial finger–finger contact penetration (0.016 mm) is retained in the validation record.

Matched 15 s drive-only depth: finite-patch regrasp 0.444 mm, hydroelastic regrasp 0.464 mm, finite-patch sliding 0.169 mm, hydroelastic sliding -0.005378 mm. Hydroelastic sliding has loaded handle slip in 15.5% of return samples, but almost no net screw advance. A separate full-task hydroelastic probe fails during lift at 5.80 s with a 95.1 N terminal finger load; its buffer occupancy remains valid. This failure is not part of the finite-patch success count.

Additional nominal synergy probes retain 99.751% variance at rank 6 and 99.987% at rank 10, but fail cap acquisition and bit engagement respectively. Variance retained is not a contact-preservation criterion. These are diagnostic probes, not independently tuned low-dimensional controllers.
