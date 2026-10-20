# Screwdriver control laboratory

**Current task:** [full UR5e–Tesollo screwdriver motion](FULL_MOTION.md): horizontal pickup, reorientation, grasp transfer, repeated driving and verified seating. See the paired control-interface sweep and separate hydroelastic contact ablation. [Earlier pre-aligned insertion](SCREW_IN.md) and the free-rotation proxy below are retained as historical studies.

## Historical rotation proxy

This is a runnable MuJoCo simulation and RTX rendering pipeline and a measured website comparison for the supplied human screwdriver video. It adds a standalone Tesollo scene without changing the existing Newton/GaP implementation.

**Result:** none of the 50 paired +90° trials completed the full guarded task. Five low-friction free-body starts were rejected, leaving 45 valid +90° trials and zero successes. In a separate nominal +15° motor test, fingertip, object-angle and contact-aware controllers passed in both free-body and fixture modes (6/10 total). The four object/contact local successes also passed at a 1 ms timestep. This establishes limited local capability, not repeated screwdriver spinning, robust generalization, thread engagement, or hardware readiness.

The reference's human panel is cropped using `crop=640:1048:0:0` and encoded at 30 fps. Task phases were interpreted manually. No vision model, human joint retargeter, live LLM, or trained motor policy was run. The target angle and proxy dimensions are explicit experiment choices, not recovered measurements.

## Run

From the repository root, with Python 3.12+, the dependencies in `requirements.txt`, the included Tesollo meshes, and FFmpeg:

```bash
# Optional: regenerate the nominal collision-refined reset (diagnostic still only).
MUJOCO_GL=osmesa python -m screwdriver_lab.scene

# Five methods × five paired conditions × free/fixture = 50 main trials.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.run \
  --output screwdriver_lab/artifacts/revision_rtx/main

# Separate local motor gate; nominal case only.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.run \
  --output screwdriver_lab/artifacts/revision_rtx/local_motor --cases nominal --goal 15

# Four numerical sensitivity runs.
OPENBLAS_NUM_THREADS=1 python -m screwdriver_lab.run \
  --output screwdriver_lab/artifacts/revision_rtx/timestep_check \
  --cases nominal --methods object contact --goal 15 --timestep .001

python -m unittest screwdriver_lab.test_pipeline -v
```

The working physics interpreter on this workstation is `/home/uyoo/miniconda3/envs/newton/bin/python`. RTX rendering uses a separate Python environment with `bpy==4.5.10`, NumPy and Pillow at `/home/uyoo/miniconda3/envs/blender-render/bin/python`. `batch_rtx.py` exports all 60 recordings, runs at most two render workers on GPU 1, and writes the website manifests after all render jobs finish. On another workstation, set `DEXAGENT_PHYSICS_PYTHON` and `DEXAGENT_RENDER_PYTHON` to the two interpreters (the batch and queue scripts read them) and adjust the selected GPU. On this host GPU 0 belongs to another workload and was left untouched.

```bash
# Resource guard checks that the selected GPU is idle before starting.
OPENBLAS_NUM_THREADS=1 python -m dextrain.resource_guard --gpu 1 \
  --output screwdriver_lab/artifacts/revision_rtx/render_guard --seconds 7200 \
  -- python -m screwdriver_lab.batch_rtx

# Or export and render a single recorded trial:
python -m screwdriver_lab.export_rtx \
  --input screwdriver_lab/artifacts/revision_rtx/main \
  --id free-nominal-object --out /tmp/tesollo-rtx
CUDA_VISIBLE_DEVICES=1 /home/uyoo/miniconda3/envs/blender-render/bin/python \
  screwdriver_lab/render_rtx.py --bundle /tmp/tesollo-rtx --output /tmp/rollout.mp4

python -m screwdriver_lab.validate
python -m http.server 8766 --bind 127.0.0.1 --directory screwdriver_site/dist
```

The production renderer is Blender Cycles with NVIDIA OptiX on an RTX 4090, 1280 × 960, 30 fps, 24 path-tracing samples and OptiX denoising. It refuses a CPU fallback. This is RTX hardware rendering through OptiX, not Isaac Sim/Omniverse. Exact compiled MuJoCo visual meshes and recorded world transforms are exported without pose edits, rescaling, resimulation or animation interpolation. PBR materials, area lights, camera, studio floor and a display-only wrist mount improve visibility. The mount cannot support the screwdriver in physics. Identical geometry frames may reuse a cached RTX path trace before method/time labels are applied; cache keys include exact world transforms, all mesh arrays, geometry, rendering settings and renderer source. This avoids repeatedly rendering identical acquisition states across paired methods. `render.py` is the legacy OSMesa renderer and is not used for this revision.

## Initialization correction

The first published revision hid three seconds of settling and accepted about 35 mm of acquisition slip with proximal-link/shaft loading. Its videos and results have been replaced. The upright hand now grasps the handle in front of the palm. Nominal acquisition drifts 1.46 mm, with minimum shaft clearance 14.31 mm (18.22 mm at acquisition end), peak force 4.39 N and maximum penetration 0.088 mm. These are measured simulated values, not a hardware calibration.

Every video includes reset and the entire acquisition. Low-friction acquisition failures remain visible as short rejected episodes, with a labeled final-frame hold. The old results remain in the local historical artifact directories but are excluded from the site and corrected downloadable bundle.

## Architecture

`scene.py` builds the actual 20-joint right Tesollo hand with a fixed palm. The screwdriver is a rigid cylinder, cap and shaft, with total mass 85.1 g and a 36 mm nominal handle diameter. It has either an unactuated free joint or an explicitly labeled passive hinge. The hand is oriented so the tool axis is approximately vertical. No UR5e asset or historical checkpoint is required.

`policy.py` is an inspectable feedback-driven Python policy. The runner calls `decide(observation, elapsed_seconds)` at 5 Hz. The decision selects a phase, angle and optional moving finger. `controllers.py` converts this intent into one of five abstractions, updated at 50 Hz:

| Interface | Low-level implementation | Main limitation |
| --- | --- | --- |
| Joint targets | Replay a nominal IK posture trajectory through a rate-limited servo | No task-space compensation for contact slip |
| Three PCA coordinates | Project that same nominal trajectory through a rank-three posture basis | Restricts independent finger motion; no learned semantics |
| Fingertip targets | Track material contact targets with damped Jacobian IK | Fingertip tracking does not guarantee object tracking |
| Object-angle goal | Bounded angle PI correction, material targets, the same IK | Fixed contacts reach a workspace limit |
| Contact-aware policy | Small turns, sequential finger resets, receiving-contact checks and bounded patch replanning | The current receiving contacts and support transfer are unreliable |

The same `Simulator` integrates the hand's position servos in MuJoCo at 500 Hz. It observes solved contact forces, object motion and penetration. All methods share actuator/joint bounds and physical stop guards. Only the 20 hand servos are actuated; object motion is produced by contacts, never direct object torque or pose writes during a rollout. Gravity and a physical floor are enabled. The palm/first-link and palm/thumb-pivot exclusions match the local mesh model's mechanical overlap; other self-collisions remain enabled.

## Agent integration

The delivered policy is scripted, not a measured LLM benchmark. A generated or externally driven policy can be injected with:

```bash
python -m screwdriver_lab.run --policy my_agent_policy:Policy
```

The class constructor takes `(method, goal_deg)` and its `decide(observation, elapsed)` returns `screwdriver_lab.policy.Decision`. The simulator, measurements, rendering and shared guards remain outside the agent. Observations contain angle (degrees), angular speed (rad/s), drift (mm), tilt (degrees), all 20 joint positions (radians), and five summed finger forces (N). Decisions and their reasons are preserved in the trace. Agent calls are synchronous; an external model adapter must enforce its own latency/token budget. This is not an untrusted-code sandbox: only load trusted Python policy modules.

An outer code-improvement agent can generate a candidate policy, run it on designated **training** cases, inspect typed failure codes, and revise it. Freeze the code hash before evaluating separate held-out cases. The published five-condition sweep was used for engineering diagnosis, so it is not a held-out scientific generalization benchmark. There is no hidden live-model repair loop in the reported numbers.

## Initial conditions and acceptance

Cases are nominal, friction 0.35 instead of 0.8, 3 mm lateral offset, +30° starting phase, and 20 mm instead of 18 mm handle radius. Both contacting surfaces use the case friction. A deterministic collision-distance refinement fits offset and handle-size reset geometry before dynamics; nominal, friction and phase cases reuse the nominal joint configuration. This initialization aid uses privileged geometry, and is not a learned grasp acquisition policy. Cases and physical initialization hashes are paired across all methods. The starting phase on the largely cylindrical tool is a weak phase/symmetry test, not a hard geometric generalization test.

Every episode runs the same acquisition controller for up to 3 s. Acquisition requires at least two **distal fingers contacting the handle**, speed <0.2 rad/s, tilt <10°, drift <5 mm, finger loads <8 N, penetration <0.5 mm, shaft clearance ≥3 mm, no forbidden contact load ≥0.05 N and no external support. Cap, shaft, palm or proximal-link contact cannot count as grasp support. Any acquisition guard event is latched; the run cannot recover into an accepted start after violating a bound. The commanded 2 mm preload is a compliant servo target, not a permitted physical penetration.

The rotation reference is set to the acquired pose **without changing the object state**. The task then has 12 s. Success requires angle error ≤3°, speed <0.2 rad/s, drift ≤20 mm, tilt ≤20°, at least two loaded distal-handle contacts, loads ≤8 N, penetration ≤0.5 mm, shaft clearance ≥3 mm and no forbidden or external support, continuously at every physics step for 0.4 s. Force, penetration, shaft clearance and forbidden-contact guards run after every 2 ms step and stop at the first observed violation. Drift/tilt/external-support stop checks run at 50 Hz. Guard detection does not prevent instantaneous overshoot; recorded peaks retain it. Hand self-contact penetration has a separate 1.5 mm guard.

Contact-aware finger reset uses the remaining fingers' held actuator targets and checks the moved finger after 0.6 s. This is a prototype, not a proven force-closure or receiver-first transfer controller. It can still lose support. The video trace does not claim otherwise.

Fixture mode constrains tool translation and tilt with a passive hinge. It is only a rotation diagnostic. It does not simulate screw engagement, thread pitch, axial advance, insertion, or loaded fastening. Free and fixture results must not be pooled as evidence of free-body success.

## Evidence and validation

- `artifacts/revision_rtx/main/`: 50 main outcomes, recorded qpos, scene XML, traces and source hashes.
- `artifacts/revision_rtx/local_motor/`: ten nominal +15° outcomes.
- `artifacts/revision_rtx/timestep_check/`: four +15° object/contact checks at 1 ms.
- `artifacts/evaluation/`: earlier development sweep before observation synchronization was corrected; not the website's reported final sweep.
- `screwdriver_site/dist/data/`: website manifests and validation summary.

Rendered videos show the actual simulated trajectories, including failures. The final frame is held for a labeled 1 s end card; this is not additional simulated motion. Video sampling is 30 fps, including the entire acquisition phase; task time is reported separately from video/simulation time. Every render has a `.render.json` record of backend, device and settings. Full qpos and metrics are downloadable in the pipeline bundle. For offline website playback, also download all media packs from the website and extract them into the same directory. The source host limits individual file sizes, so media is packaged separately. The browser compares recordings; it does not simulate arbitrary slider settings.

Tests check real hand/object DOFs, absence of object actuation, rejection of invalid success states, physical falling without contacts, zero fixture rotation without contacts, joint limits, policy recovery termination, strict support classification, clear reset geometry, latched acquisition slip, and exact renderer export coordinates. All 11 regression tests pass. Source and grasp hashes, matched initial-state hashes, finite values and media references are checked by `validate.py`. The nominal +15° object/contact successes persisted in both modes at a 1 ms step, although final angles and loads changed. This sensitivity check does not validate the +90° task or calibrate the simulator against hardware.

Next motor milestones are contact-force regulation, a receiver-confirmed support transfer, and sustained repeated turning. Next agent milestones are matched LLM-generated policy trials, held-out seeds/conditions, measured model budgets, and a frozen primitive bank for clean abstraction ablations.

Tesollo assets retain their BSD-3-Clause license in `third-party/tesollo_model/LICENSE`.
