# Physics-first revision — 12 October 2026

The archived learned policies rotate a fixture-supported nut. The passive hinge prevents translation and dropping. These rollouts cannot support a claim about free in-hand dexterity; their visual quality and physical scope were inadequate for the proposed study.

## What changed

- Added unactuated six-DOF nut, closed-end wrench, and hex-head screw proxy scenes with a physical workbench. No object-pose assignments, object actuators, or hidden welds occur during a recorded rollout.
- Settled candidate grasps through free-body dynamics and validated loaded servo targets in separate hold episodes. The revised contact profile is `solref=.004,1`, `solimp=.95,.99,.001`, with a 2 ms timestep. These are provisional simulation settings, not measured hardware compliance.
- Kept zero/low-force penetrating contacts in observations so safety cannot discard geometry just because its solved load is small. Load-bearing contact classification remains separate.
- Distinguished table/external support from legitimate palm contact. A free joint alone does not establish an unsupported grasp.
- Added a friction-pyramid linear program for static gravity/disturbance support, exposed as `hand.check_support_wrench`. Free-body releases check the remaining SUPPORT contacts. This is a conditional static certificate, not a proof of dynamic stability, joint-torque feasibility, or reachability.
- Fixed receiver-first transfer to preserve an already loaded receiver during confirmation. Existing support forces are retained rather than arbitrarily reduced to a small generic target.

## Measured results

| Development test | Result | Evidence |
|---|---|---|
| Nut hold, 3 s | Pass | At least two loaded fingers throughout 1,500 samples; peak penetration 0.00993 mm; peak per-finger normal force 0.846 N |
| Screw hold, 3 s | Pass | Same gates; peak penetration 0.00910 mm; peak force 1.018 N |
| Wrench hold, 3 s | Pass with fixed loaded servo targets | Same gates; peak penetration 0.02139 mm; peak force 3.406 N. The adaptive hold variant fails its force guard. |
| Deliberate nut release | Negative control verified | About 9 cm vertical fall; final finger force zero; workbench load about 0.706 N, matching object weight |
| GaP support transfer | Pass | Confirm middle support, release thumb, hold one second on index/middle. Final position drift about 0.08 mm, rotation drift about 0.51 degrees. |
| Friction checks of that transfer | Three passes | Both contacting surfaces varied; solver coefficients confirmed as 0.5, 0.8, 1.2. Same initial geometry; not a generalization study. |
| 30-degree nut rotation | Fail | Best tested integral variant still ends about 6.19 degrees from the full target orientation and has residual motion |
| Screw alignment / wrench rotation | Fail | Orientation error, contact loss or force guard; positive grasp checks do not imply these tasks succeed |
| 8 mm in-hand translation | Fail | Outward-motion variant retains the object but has about 4.70 mm final position error and 10.44 degrees unwanted rotation |
| Full finger-gait cycle | Fail | Recontact/transfer attempts trigger support constraints. The successful release primitive does not establish a complete gait. |

Hold success requires final pose error at most 2 mm / 3 degrees, angular speed below 0.2 rad/s, linear speed below 0.02 m/s, a 0.2-second final hold, at least two fingers above 0.05 N at every physics sample, no external support, no force over 5 N, and no penetration over 1 mm. The thresholds were not relaxed to make failed motion trials pass. Hold peaks above use every physics step; several early motion diagnostics stored only decimated contact traces and should not be treated as equally instrumented trials.

The initially attempted object-only friction variation did not reduce pair friction because the solver selects the larger coefficient. Those runs are retained as `transfer_mu*`; the corrected two-surface runs are `transfer_both_mu*`. Do not count the former as distinct low-friction evidence.

All completed result records, including failures, are in `review_site/dist/data/development_ledger.json`. Raw states, controller reports, GaP graphs/traces, initialization records, and experiment logs are under `runs/physics_revision_20261012/`. `source_manifest.json` describes the source at the end of this development iteration; it is not an exact historical source snapshot for every early probe. Published clips include state/scene hashes. The earlier fixture-initialization bootstrap was also retried and failed; its fixture-dependent loaded pose is not used as a valid free-object result.

## What the paper should test

Contact-mode representations, finger gaits, and motion/contact optimization are established ideas: see [Sundaralingam and Hermans, 2018](https://arxiv.org/abs/1804.04292), [ContactGen, 2023](https://arxiv.org/abs/2310.03740), and [Jiang et al., 2025](https://arxiv.org/abs/2505.04978). The proposed contribution is the **agent-facing executable contact program**: semantic roles, verified transitions, object-centric effects, and structured failures that let GaP repair graph structure and preserve successful compositions.

A meaningful paper needs equal agent models, observations, low-level controllers, action horizons, and improvement budgets across A–F. Independent ablations must isolate roles, transition contracts, feasibility checks, and structured feedback. Report valid-action rate, task success after repair, calls/iterations to success, failure types and recovery, plus transfer to unseen geometry/friction/grasps. The authored GaP demonstration here establishes integration and one physical transition; it does not establish an LLM reasoning advantage.

The industrial ladder remains: nut turn-stop-regrasp; screw axis alignment and insertion-ready handoff; wrench head/handle regrasp, engagement and ratchet cycles; cable feeding/wrapping with calibrated tension. This revision adds real rigid-object scenes and contact evidence. Threaded fastening and deformable cable dynamics remain missing.

## Reproduction and resources

Use `.venv-newton/bin/python -m scripts.physics.manipulate --feedback --adaptive --task hold --steps 300 --name NEW_OUTPUT_NAME` for a nut hold from the saved dynamic initialization. Wrench's passing hold uses `--object wrench --feedback --passive`; screw's uses `--object screw --feedback --adaptive`. Use `scripts.physics.gap_transfer --friction .8 --case NEW_OUTPUT_NAME` for the GaP transfer. The scripts are development probes and reference this iteration's saved initialization directory; use fresh names to preserve evidence.

These single-world probes use Newton collision detection and the MuJoCo Warp solver on CPU, not the native-MuJoCo diagnostic used early for calibration. Shell wall-time limits were 300–600 seconds per probe, 900 seconds for rendering, with BLAS restricted to one thread and rendering to at most eight software-rendering threads. No new GPU training was started on the occupied GPUs. Available host memory and disk headroom remain ample; kernel-log access remains unavailable, so absence of logged experiment OOM errors is not a complete host crash audit.
