# Newton training on the three A100 GPUs

October 12: the original three spin jobs finished. Their 7,680-rollout evaluation demonstrates contact-driven spinning but 0% precise angle-and-hold capture; two zero-action control rollouts were numerically unstable. See [the aggregate](../runs/motor_eval_20261012T015042Z/aggregate.json). A new three-seed precision curriculum is running on GPUs 1, 6, and 7; `status` defaults to that batch. This is still a passive-fixture prerequisite, and does not complete the industrial A–F study.

Both launchers now wrap jobs in the resource supervisor: 8 GiB process-group RAM, 12 GiB GPU memory, 2 GiB job output; stop below 64 GiB available host memory or 20 GiB filesystem headroom. Core dumps are disabled and individual output files are capped at 256 MiB. Memory/aggregate disk thresholds are polled every five seconds, not enforced reservations. The supervisor records preflight refusals and stops surviving child processes if their parent exits.

This package trains a prerequisite motor skill for the action-abstraction study: the UR5e holds its wrist near a fixed pose while the 20-joint DG5F-S rotates a passive nut-shaped object with its fingers. The goal direction changes across episodes. The object has a revolute fixture constraint and no actuator. This stage does not test thread capture, axial advancement, wrench use, cable wrapping, or an LLM agent.

The robot is composed from the existing robosuite UR5e and Tesollo mounted right-hand assets. The flange transform is provisional. Initial finger poses are optimized against mesh-surface distances and self-contact. The convex palm collider fills the thumb-pivot cutout; that specific mechanical body pair is filtered, while other nonadjacent self-collisions remain active. Collision meshes and dynamics need further calibration before the industrial benchmark.

The Newton solver runs with MuJoCo Warp on CUDA, using Newton's convex collision backend. Collision detection uses a 0.5 mm gap per shape, and shapes on the same rigid body are excluded. Native MuJoCo is used only for asset initialization and independent CPU sanity checks. The environment provides state observations; no renderer participates in training. Any inspection PNG is rendered with MuJoCo OpenGL, not RTX. NVIDIA does not officially support A100s for Isaac Sim RTX rendering; RTX observations remain a separate validation requirement.

The local `.venv-newton` inherits packages from the existing Conda `newton` environment, with stable `mujoco==3.10.0` and `warp-lang==1.16.0` installed locally. The base environment and upstream repositories are not modified. Per-run metadata records exact runtime versions, source hashes, scene hash, GPU UUID, and limits.

```bash
cd /home/uyoo/Projects/dexagent

# Regenerate the composed scene only when no jobs are using it.
.venv-newton/bin/python -m dextrain.build_scene

# Physics smoke check on an idle GPU, with a fresh output path.
.venv-newton/bin/python -m dextrain.resource_guard --gpu 1 \
  --output runs/new_smoke --seconds 600 -- \
  .venv-newton/bin/python -m dextrain.env --num-envs 16 --steps 100

# Three independent PPO seeds. Refuses GPUs with existing compute jobs.
.venv-newton/bin/python -m dextrain.launch --gpus 2 3 7 --num-envs 2048 --hours 8

# Precision curriculum, then automatic held-out evaluation. Select idle GPUs.
.venv-newton/bin/python -m dextrain.launch_precision --gpus 1 6 7 \
  --warm-batch runs/nut_motor_20261010T020245Z --num-envs 1024 --hours 7.5

# Inspect jobs, latest metrics, checkpoints, and current GPU utilization.
.venv-newton/bin/python -m dextrain.status
```

The supervisor pins each job by GPU UUID and holds a cooperative per-GPU lock. It does not terminate other jobs or change GPU operating modes. Training saves a checkpoint after the first update and every 25 updates, and exits on nonfinite metrics, excessive numerical instability, or near-full physics buffers. SIGTERM/SIGINT requests a checkpoint and graceful stop after the current update. Current precision PIDs and full commands are in the batch identified by `runs/latest_precision_campaign.json`; each seed contains `process.log`, `resources.jsonl`, and `train/metrics.jsonl`. The previous spin batch remains at `runs/current`. Inspect the command line before signaling a PID, since operating systems can reuse PIDs after processes exit.

Training metrics measure reward, signed nut velocity, instability, PPO divergence, contact/constraint buffer occupancy, and throughput. Precision training adds goal error and measured episode successes; these are training outcomes, not held-out task success rates. Occupancies are sampled at each control step; inspect solver logs for intervening physics-substep overflow. Precision campaigns evaluate mean/sampled policies, zero action, and disabled-contact controls on fresh reset seeds after training. GPU utilization is measured rather than padded with synthetic compute. Each new batch executes the code and scene snapshot under `source/`; external libraries remain shared, and precision campaigns verify mesh hashes before training and evaluation.
