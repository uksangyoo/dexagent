# Dexterous action abstractions for UR5e + Tesollo DG5F-S

## Repository layout

```bash
git clone --recursive <repo-url>    # or: git submodule update --init --recursive
```

External code and assets live in `third-party/`:

| Path | Source |
| --- | --- |
| `third-party/IsaacLab` | submodule, [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) |
| `third-party/newton` | submodule, [newton-physics/newton](https://github.com/newton-physics/newton) (used by `screwdriver_lab/run_contact_bench.py`) |
| `third-party/robosimstudio` | submodule, [robosimstudio/robosimstudio](https://github.com/robosimstudio/robosimstudio) (used by `crate_lab`) |
| `third-party/tesollo_model` | submodule, [tesollodelto/tesollo_model](https://github.com/tesollodelto/tesollo_model) |
| `third-party/universal_robots_ur5e` | vendored UR5e MJCF and meshes (BSD-3-Clause, see `LICENSE`) |
| `third-party/vab_crate_washing` | vendored crate-washing scene assets (MIT, see `LICENSE`) |

Generated outputs (`runs/`, `screwdriver_lab/artifacts/`, `crate_lab/results/`), local virtualenvs (`screwdriver_lab/vendor/`) and the published site (`screwdriver_site/`) are not tracked.

## Goal-to-control workflow (October 16)

The executable authoring layer is in [screwdriver_lab/agentic](screwdriver_lab/agentic/README.md): language/video request → agent-authored graph and Python nodes → selected hand abstraction → 20 hand + 6 arm motor commands → measured feedback for revision. It includes an external-generator command protocol and bounded revision loop with a fixed physical goal contract. The current examples were authored in this conversation; a separate live model provider has not been configured or evaluated. The website exposes actual policy transitions, RTX replays and actuator traces. Full pickup-to-seat execution remains incomplete. All GPU work is restricted to physical GPU 1.

## Screwdriver control laboratory (October 15)

The October 16 cube extension now includes one continuous pickup → 30° in-hand
goal → stable hold result: 3.94° final error, 77.52 mm lift and 0.312 s accepted
hold, with only 0.014° palm rotation. A matched rerun reproduced the trajectory;
fingertip-path and rigid-object-twist adapters collided from the same initial
state. Perturbed-start and quarter-turn results do not yet pass. The website
includes synchronized RTX movies, the frozen comparison, sensitivity cases,
and the request → graph → bounded motor-command → measured-feedback artifacts.
The screwdriver has reached engagement and partial driving in a prepared
diagnostic; the continuous pickup-to-seat task remains incomplete.

The current revision is in `screwdriver_lab/hom/`. It uses Drake SAP with hydroelastic fingertip pressure patches, explicit mating thread contact, and 26 bounded robot motors. Screwdriver and screw are independent unactuated rigid bodies. MuJoCo supplies kinematics only; dynamics runs on CPU and RTX rendering uses physical GPU 1.

A frozen six-second, three-interface comparison produces 39.47° / 0.116 mm for fingertip paths, 26.10° / 0.079 mm for rigid object twist, and 154.10° / 0.434 mm for HOM-inspired relative-contact control. Three starting finger clearances preserve this ordering. These are prepared-grasp manipulation trials, not full pickup-to-seating successes. Material assumptions remain uncalibrated; numerical and friction sensitivity checks are retained with the raw evidence in `screwdriver_lab/artifacts/hom_revision`.

Earlier full-task claims from `artifacts/physical_task` are superseded: inspection found a wrist collision capsule overlapping the thumb base, an oversized tool surrogate, and condition-specific tuning. Those solver-generated recordings are historical diagnostics, not credible evidence of realistic manipulation or an abstraction ranking. The corrected wrist uses the supplied CAD envelope. See the [interactive comparison](https://dexagent-screwdriver-lab.comet-basil-0008.chatgpt.site) and `screwdriver_lab/HOM_REVISION.txt` for methods, limitations, and reproduction.

The contact-aware implementation is in `dexagent/`. The existing three-seed PPO curriculum remains in `dextrain/`; it is a separate experiment. The design and earlier literature study are in [research/action_abstraction_study.md](research/action_abstraction_study.md).

## Visual review

Published privately at [Dexagent Simulation Review](https://dexagent-simulation-review.comet-basil-0008.chatgpt.site). Deployment identity and the exact source commit are recorded in [runs/review_site_publication.json](runs/review_site_publication.json).

The browser review is in [review_site/dist/index.html](review_site/dist/index.html), with an [offline ZIP](review_site/dist/assets/dexagent-review.zip). It now contains **14 actual Newton diagnostic rollouts**, each rendered from three cameras: seven free-body/controller recordings and the seven historical fixture recordings. Contact forces, role assignments, penetration, pose error, and raw states are inspectable. Geometry comparison figures show the visible meshes and their convex collision approximation. Newton generates trajectories; MuJoCo OpenGL renders recorded coordinates. RTX is not used.

The new physical checks pass three-second holds for nut, screw, and wrench proxies. They require two loaded fingers at every 2 ms sample, bounded force/penetration, low final speed, accurate pose, and no external support. A deliberate release makes the nut fall onto a physical workbench. A support-transfer graph also passes through the real GaP executor and repeats at measured contact friction 0.5, 0.8, and 1.2. Precise manipulation and the full finger-gait cycle still fail; these are controller development results, **not a completed agentic A–F study**. See [research/physics_revision_20261012.md](research/physics_revision_20261012.md) and the [complete development ledger](review_site/dist/data/development_ledger.json).

The old floating-nut recordings used a passive hinge and are archived as fixture experiments. They do not validate a free grasp. Those three precision training jobs were stopped by resource guards around 04:15 UTC: one for GPU-memory usage and two for GPU-monitoring timeouts. This revision used Newton's Warp solver on CPU while all GPUs were occupied.

The hosted/offline review contains a dated snapshot. For a local view that refreshes training progress every 30 seconds:

```bash
cd ~/Projects/dexagent
.venv-newton/bin/python -m scripts.review.serve --port 8765
```

It binds only to `127.0.0.1`. From another computer, forward it with `ssh -L 8765:127.0.0.1:8765 uyoo@bajcsy`, then open `http://localhost:8765`. Video scrubbing is supported by HTTP byte-range responses. Refreshing progress only reads logs; it does not launch simulations. The recorded videos remain fixed until explicitly recaptured.

`scripts/review/capture_rollouts.py` records Newton states; `render_rollouts.py` renders them without resimulation; `encode_webm.py` supplies browser-compatible previews. These compute jobs should run through `dextrain.resource_guard`. `export_data.py` rebuilds the historical snapshot; `finalize_revision.py` combines the new free-body evidence with that archive, and `package_review.py` rebuilds the portable ZIP. Rendered assets and sanitized data are the only experiment artifacts in the website; checkpoint weights and local tooling are excluded. Sites identity and deployment configuration are retained in `review_site/.openai/hosting.json`.

## October 12 experiment and server audit

The original three eight-hour training jobs completed normally (1.382 billion environment steps combined). The host has not rebooted since August 17. Available experiment logs contain no out-of-memory or disk-full errors. Three early diagnostics did fail with CUDA illegal memory access before the switch to Newton collision detection. Kernel logs are inaccessible, so this is not a comprehensive host crash audit. At inspection, about 1.3 TiB of RAM and 129 GiB on the output filesystem were available. Evidence: [health audit](runs/health_audit_20261012.json).

Completed held-out evaluation: **7,680 rollouts**, three training seeds, five control regimes, 512 episodes per seed/regime. Each trained seed's deterministic policy crossed 30°, 60°, and 90° in 100% of episodes, but **0% stopped and held any requested angle**. Settled capture requires ±3°, speed below 0.2 rad/s, and a continuous 0.2-second hold. This is a passive fixture experiment with the original geometry/friction, not free-body success or an A–F comparison.

| Regime | 90° crossing, range across seeds | 90° settled capture, range across seeds | Unstable episodes |
|---|---:|---:|---:|
| Trained policy, mean action | 100% | 0% | 0 / 1,536 |
| Trained policy, sampled action | 100% | 0% | 0 / 1,536 |
| Zero action | 0% | 0% | 2 / 1,536 |
| Random action | 22.66–23.05% | 0.78–0.98% | 0 / 1,536 |
| Trained policy, object contacts disabled | 0% | 0% | 0 / 1,536 |

The two unstable controls share reset case `reset_20000_182` across two seeds and remain in the results. Removing object contacts produces exactly zero object motion. [Aggregate and per-seed evidence](runs/motor_eval_20261012T015042Z/aggregate.json) retain these limitations.

The next three-seed curriculum explicitly conditions on ±30°/60°/90° goals and rewards precise stopping and holding. It runs on GPUs **1, 6, and 7** from a frozen Python/scene snapshot, with original spin checkpoints as warm starts. The initial precision pilot was stopped because PPO updates diverged; the replacement uses a lower learning rate and target-KL early stopping. The current batch is recorded in [latest_precision_campaign.json](runs/latest_precision_campaign.json). Each job automatically evaluates fresh reset seeds after training; maximum training time is 7.5 hours and the supervisor allows 8 hours total. This remains prerequisite motor training.

All new training launchers use `dextrain.resource_guard`. Stop thresholds: process-group RSS 8 GiB, selected GPU memory 12 GiB, per-job output 2 GiB, available host RAM below 64 GiB, or free space below 20 GiB on the output filesystem, `/tmp`, or `/`. Monitoring runs every five seconds; these are polling stop conditions, not reserved memory or a guarantee against OOM. Core dumps are disabled and each child output file has an OS-enforced 256 MiB limit. Occupied GPUs are refused. Only the job's own process group is signaled, including surviving descendants. Limits and samples are retained beside checkpoints.

Free-body bootstrap remains unsuccessful. A three-finger fixture grasp produced load-bearing contacts but excessive penetration and failed after removing the fixture. A new 1 mm object-penetration guard rejects that state. Optional force-space stiffness probes at 3,000 and 10,000 N/m also fail this gate; they are experimental profiles, not calibrated materials. Raw probes are under [contact_calibration_20261012T021629Z](runs/contact_calibration_20261012T021629Z). The task runner now latches task failures at every 2 ms physics step, including failures inside long agent actions.

The current physical milestone is **contact establishment through the real GaP executor**: an index-finger contact was established in Newton in 40 control steps (0.8 simulated seconds), with about 0.066 N summed normal force and 0.00018 m/s tangential slip, independently accepted by a GaP checkpoint. This is a component smoke test, not a manipulation success rate. The result is in [runs/contact_grammar_edge_validated/result.json](runs/contact_grammar_edge_validated/result.json). The complete establish → maintain → release sequence also passed in two separate GPU runs. A second GPU test validates the free-body nut's pose conversion between Newton and the kinematics model, without claiming a stable grasp.

## Implementation and validation status

| Component | Implemented | Validation / remaining work |
|---|---|---|
| Structured state | Five named fingers, four joint positions/velocities each, fingertip poses, object-specific surface patches/normals, solved forces, object 6D pose and twist relative to the moving palm | Newton GPU observation smoke; frame, COM-velocity, force-sign, slip and rolling tests |
| Roles and grammar | Independent SUPPORT/DRIVE/REPOSITION/RELEASE/FREE roles; strict Pydantic action/transition contracts; explicit units and frames | Invalid/nonfinite values, mismatched targets, absent support and force-limit tests |
| Contact controller | Establish, maintain, release, receiver-first support transfer; bounded target integration and structured failure reports | Establish → maintain → release passed in Newton; establishment passed through GaP; free-body receiver-confirmation → thumb release → one-second hold passed through GaP; complete finger-gait cycle still fails |
| Object-motion compiler | Object/fixture effects → material contact-point targets → bounded hand/wrist IK, joint limits, sampled collision rejection, per-physics-step force/support checks | Frame and rejection tests; sustained physical 30°/60°/90° rotation remains unvalidated |
| Composite skills | GaP graph builders for pinch, gait, rotate/translate, regrasp, ratchet, and cable-feed composition | Ordinary GaP routing/checkpoint tests; compositions inherit controller/backend limitations |
| GaP integration | Connector, eight `hand.*` tools, complete JSON schemas, privileged World snapshots, repair feedback, verified workflow preservation | Real executor success/failure routing and checkpoint enforcement tested; no separate graph executor |
| Benchmarks | Randomized **free-body nut** scene generator; nut, wrench, cable and screw measurement predicates | CPU model and Newton GPU free-body pose checks; three rigid-object single-case hold gates pass; randomized grasps, precise motion, threaded fastening, and deformable cable/tension remain to implement/validate |
| A–F ablations | Interface adapters, training-only PCA fitting, paired cases, shared physical limits, budget/provenance validation, learning-curve hooks | Manifest is pending; no comparative task success results or live LLM repair runs yet |

Sliding/rolling controllers and cable motion deliberately return `unsupported`. The schema accepts those intentions, but rigid-object IK cannot execute a deformable task. The wrench/screw predicates require actual thread advancement, not just angular motion. These gates prevent substituting the passive nut curriculum for the requested industrial tasks.

## Run

The prepared `.venv-newton` inherits the existing Conda `newton` environment. It contains local pins `mujoco==3.10.0`, `warp-lang==1.16.0`, editable `dexagent`, and editable sibling GaP/core packages. It does not change the Conda base installation or the sibling GaP source.

```bash
cd ~/Projects/dexagent

# CPU contract, geometry, task and real-GaP integration tests.
.venv-newton/bin/python -m unittest discover -s tests -v

# Optional real-physics component test; choose an idle GPU and fresh output path.
.venv-newton/bin/python -m dextrain.resource_guard --gpu 1 \
  --output runs/contact_test_new --seconds 600 -- \
  env DEXAGENT_GPU_TESTS=1 .venv-newton/bin/python \
  -m unittest discover -s tests -p test_newton_gpu.py -v

# Contact establishment through Newton + GaP + a physical checkpoint.
# Select an appropriate GPU; this is a small single-world diagnostic.
.venv-newton/bin/python -m dextrain.resource_guard --gpu 1 \
  --output runs/contact_demo_guard --seconds 600 -- \
  .venv-newton/bin/python examples/contact_grammar/run.py \
  --output runs/contact_demo --finger index --timeout 4

# Generate the paired A–F manifest: 2,880 episodes per evaluation round.
.venv-newton/bin/python -m dexagent.experiments.abstraction_ablation manifest \
  --cases 20 --output study/artifacts/contact_grammar_v2_manifest.jsonl

# Inspect the independently running PPO motor curriculum.
.venv-newton/bin/python -m dextrain.status
```

The schema is [dexagent/grammar/schema.py](dexagent/grammar/schema.py). A GaP bundle is provided at [bundles/skills/dexterous-hand/SKILL.md](bundles/skills/dexterous-hand/SKILL.md); load `bundles/` as a GaP registry root, and use `DexterousConnector` for runtime tools. Call `hand.schema` to retrieve full machine-readable contracts. `examples/contact_grammar/run.py` demonstrates a concrete workflow and checkpoint sidecar. Its default is an explicit index-finger contact point on the fixture nut; `--object-point X Y Z` changes that task input. Automatic nearest-surface proposals are available from the environment but can choose an unreachable or poorly controlled facet, so they are not a validated contact planner.

Current validation provenance is recorded in [runs/progress_validation_20261012.json](runs/progress_validation_20261012.json): 41 CPU tests passed, two optional GPU tests passed separately. The October 10 record [runs/representation_validation.json](runs/representation_validation.json) describes earlier source and is retained as historical evidence. Failed diagnostic runs remain under `runs/`. In particular, rounding a target near a nut edge previously selected the wrong face normal and timed out; exterior-feature projection now uses the finger's approach direction to disambiguate faces within 0.1 mm. A CPU regression covers that failure. The earlier GaP component is preserved, with its checkpoint and scoped provenance, in [runs/verified_skills/fixture_index_contact_v1](runs/verified_skills/fixture_index_contact_v1).

## Contact and controller conventions

All lengths are meters and forces are newtons. Quaternions are `wxyz`, matching GaP; the adapter converts Newton's `xyzw`. Contact normals point **out of the object toward the finger**. Forces are forces **on the finger**, recovered from the solved MuJoCo-Warp constraints, not inferred from penetration. The force cap is summed over each finger's target-object patches.

The adapter keeps collision-input and solved-observation buffers separate. State, contact points and relative velocities refer to the beginning of the last solved 2 ms step, so contact geometry is paired with its solved forces rather than with a mismatched post-integration pose. The moving-palm twist subtracts both palm translation and rotational transport. Self/fixture contacts never count as target-object support.

Classification uses force/proximity thresholds and slip hysteresis. Rolling additionally requires persistent low slip, relative tangential angular velocity, and material patch migration on both bodies. It is a heuristic estimate with confidence/evidence; contact-patch discretization can still cause ambiguity. Spin about the normal alone does not count as rolling. Repeated reads of one snapshot cannot manufacture temporal confirmation.

The controller is a guarded, receding-horizon IK prototype, not a calibrated force MPC. Count/force/slip thresholds do not constitute a force-closure or object-wrench feasibility certificate. IK uses the reference MuJoCo model for kinematics and collision checks, while Newton executes the motion. Convex-hull differences can affect accuracy. Runtime force/support constraints are checked every physics substep and execution stops on a violation; this detects limits being crossed and does not prove a physical force can never overshoot. Actuator target integration retains gravity/contact load error with a bounded windup limit.

The hand mapping is explicit (`joint_1_*` thumb through `joint_5_*` pinky) for this local standard right 20-DOF model. Mount geometry, collision approximations, friction, drives, contact thresholds, and all force limits still require calibration. This package exposes simulation tools only.

The current checkout uses Newton's `SolverMuJoCo.update_contacts()` force export. [Newer Newton documentation](https://newton-physics.github.io/newton/latest/concepts/solver_observables.html) describes a different solver-observables API; changing the installed Newton revision requires adapting and revalidating the observation adapter.

## Experiment discipline

The new manifest has six representations × four tasks × two held-out splits × three agent seeds × twenty physical cases = **2,880 episodes per evaluation round**. Angle targets cycle through 30°, 60°, and 90°. Physical cases, dimensions, friction, initial pose and grasp perturbations are paired across methods. The out-of-distribution size/friction ranges are disjoint from training ranges. Invalid initializations must be repaired for the entire paired case rather than silently counted as manipulation failures or discarded by method.

A uses normalized joint rates integrated into position targets; B decodes a training-only PCA posture basis; C specifies palm-frame fingertip positions; D selects frozen object-motion primitives; E freezes functional roles and does not permit support reassignment; F permits dynamic roles and contact transitions. E can release and re-establish its fixed fingers, but cannot transfer a support role. All methods retain the same structured feedback, so E/F focuses on role reassignment. A separate feedback-removal ablation is needed to attribute gains specifically to diagnostic feedback.

All representations share the position servo, physical limits, scene, observation regime and agent budgets. Cartesian/grammar methods also share the IK implementation. The current A–F adapter requires a **fixed wrist**; the standalone motion controller supports wrist IK, but a common arm-command interface for every baseline is pending. A/B/C accept `duration_s` (default 0.2 seconds) and repeat their guarded servo increments within the same maximum action horizon as D/E/F (default 5 seconds). Thus low-level methods are not restricted to one 20 ms step per agent call. Logs retain physical time and tool-call counts; equal calls alone are not equal motion budgets. Simulated-time budgets cap actions, and wall-time limits are checked before and after agent calls. An agent/provider adapter must separately cap model latency and token spending before costly LLM studies.

Policy improvement rounds are distinct from decisions within a rollout. `learning_curve()` evaluates a frozen snapshot and sends **training feedback only** to repair. No API calls or model-generated policies are launched by importing this package. Bind the runner to the selected GaP agent and record model/prompt/controller/scene provenance before starting the study.

`run_trial()` uses the hand observation for nut metrics. Other tasks require an explicit `measure_task(state)` callback returning their measurement type; capability flags alone are insufficient. The callback is invoked at every physics step, and the first terminal result immediately halts an ongoing action. Before full experiments, task backends still need calibrated measurements (including tension and thread advancement), verified grasps, and a frozen controller operating region. The current predicates and rollout hooks are not a completed industrial evaluation backend.

Results are stratified by task, split and improvement iteration. The summarizer rejects pending outcomes, duplicate trials, incomplete A–F pairs, changed physical cases/budgets and mismatched controllers/agents/observations. Next-action recovery and eventual task recovery are reported separately. The earlier study's crossed seed/case bootstrap should be applied to completed paired outcomes before drawing performance conclusions.

RTX remains unconfigured. These experiments currently use state/contact observations; A100 GPUs are [not supported for Isaac Sim RTX rendering](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). Renderer integration is a distinct pending validation step.

## Historical controlled-slip prototype (2026-10-15; invalidated)

The standalone `screwdriver_lab` experiment now implements an index-cap,
thumb/middle controlled-slip gait on an articulated UR5e + Tesollo DG5F-S.
Its executable policy graph routes acquire → drive → unload → sliding return,
with measured contact gates and seating verification. It uses the general
Graph-as-Policy structure without the GaP harness or a runtime LLM.

The old prototype reported 720° / 2 mm and a fifteen-case comparison.
Those records used the overlapping wrist capsule and are **invalid as evidence
of realistic task completion or abstraction superiority**. They are retained
only as development history. The corrected hydroelastic comparison at the top
of this README supersedes them; full pickup-to-seating remains unverified.

This lab has its own RTX replay pipeline (Blender Cycles / OptiX, physical GPU 1)
and does not configure the separate contact-grammar experiments above.

- Implementation: `screwdriver_lab/controlled_slip.py`, `slip_graph.py`, `slip_compare.py`.
- Reproduction and limits: `screwdriver_lab/artifacts/reference_strategy/CONTROLLED_SLIP_REPRODUCE.txt`.
- Raw experiments and frozen source snapshots: `screwdriver_lab/artifacts/full_motion/`.
- Website: `screwdriver_site/dist/`; preview with `python -m screwdriver_lab.serve_screw`.
- Mechanical/provenance checks: `python -m screwdriver_lab.slip_validate`.

The contact-based stopping branch now passes its 0.5 s hold test after
695.8° / 1.941 mm of screw motion. This is an explicitly **checkpoint-resumed**
validation from the original long trial, whose incorrect distance guard let
it lose alignment after reaching the seat. No object pose or velocity is
adjusted at the checkpoint. A fresh complete pickup-to-seat rollout remains
unverified. See `screwdriver_lab/HOM_REVISION.txt` for the criteria and provenance.
