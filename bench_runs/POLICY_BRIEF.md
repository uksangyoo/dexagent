# Brief: write a policy for one task at one finger-coordination level

You are one of many agents in a controlled study (UR5e arm + 20-DOF Tesollo DG5F-S right hand, simulated in Newton/MuJoCo-Warp, repo /home/uyoo/Projects/dexagent). The study asks which finger-coordination abstraction lets an agent write successful, robust manipulation policies fastest. Each agent gets one task and one abstraction level. Your score is what your policy achieves, measured against the iteration budget and on held-out randomized starting conditions. Treat it as a real engineering job: succeed reliably across the variation, not just at the nominal start.

## Your assignment

Your workdir (given in your prompt) contains:
- `agent.json`: the task folder, your level, the iteration budget, and camera names you can film.
- `TASK.md`: the goal, the objects, the start state, the variation ranges, and the exact success rule.
- `policy.py`: a stub for you to replace.

## The interface

Read `dexagent/levels.py`. The module docstring covers the API common to all levels, and the class for your level documents its finger commands. Your policy is:

```python
import numpy as np            # numpy and scipy only
def run(api):                 # a generator; every motion call is one too
    p, R = api.object_pose("cube")
    yield from api.move(...)  # or your level's primitives
```

Control runs at 50 Hz, and each `yield` is one 20 ms step. Joint position servos drive the robot: the fingers have kp 3 N·m/rad and |tau| ≤ 0.65 N·m, so the fingers are weak and compliant. The arm follows palm poses through IK, and an unreachable pose raises an error. Observations (object poses, fingertip positions, contact forces) are exact simulator state, and they are the same at every level.

## Rules (they make the comparison valid)

1. Command the robot only through the public methods of `api` for your level. Do not touch `api._c`, `api._sim` or anything else that starts with an underscore. Do not import robosimstudio, dexagent, mujoco or newton in the policy. Do not use the CPU kinematics (`dexagent.workcell.Kinematics`) offline to precompute things either. If your level lacks something you want, such as IK at the joint level, you may build it yourself inside policy.py from what `api` exposes. Note in NOTES.md when you do this, because it is part of what we measure.
2. Do not read the expert solutions or anything derived from them: `tasks/**/scene.py` and its helper modules, `tasks/**/README.md`, `README.md`, `docs/`, `artifacts/`, `bench_runs/expert/`, git history, and other agents' workdirs under `bench_runs/agents/`. You may read `dexagent/levels.py`, `dexagent/hand.py`, `dexagent/workcell.py`, `dexagent/bench.py`, `dexagent/iterate.py`, `dexagent/cases.py`, your task's `check.py` (the scorer), and your workdir.
3. Every simulation goes through `dexagent.iterate`, and each call is one iteration of your budget:
   ```bash
   cd /home/uyoo/Projects/dexagent && .venv/bin/python -m dexagent.iterate --workdir <your workdir> [--film <camera>] [--film-case 0|0.5|1]
   ```
   Your budget is in agent.json (max_iters, 4). Each call scores your current `policy.py` on three new random starts, one each at severity 0 (nominal), 0.5 and 1, and prints the results. The GPU is shared by ~20 agents, so one call can take 10-40 minutes, mostly waiting in the queue. **Run it with the Bash tool's `run_in_background: true`, then wait for its completion notification and read its output file. Never wrap it in `timeout`, `nohup` or `&`, and never hand back while an iteration is running.** Only completed iterations count against your budget, so an iteration killed while queued is not counted. Do not stop early because iterations are slow: keep going until you meet the stop rule or use up your budget. Filming adds a 12-frame contact sheet (a jpg you can look at with Read) of one case. Use it when you cannot tell from the metrics what went wrong. Per-case JSON files in `iter_<k>/` contain the metrics, the perturbation drawn and your api call log. Use `api.note(...)` to log your own diagnostics; they show up in that log.
4. Do not change files outside your workdir. Write all scratch and output files (including background-run output) inside your workdir, never in a shared scratchpad or /tmp, so they cannot collide with other agents' files.

## How to work

- Think and plan before you spend an iteration. Iterations are the budget; thinking time is not, though wall clock is limited to about 2.5 hours overall, and every GPU minute is shared with ~40 other agents.
- After every iteration, append 2-4 lines to `NOTES.md`: what you changed and why, the result, and what you learned about the hand or the task. Also add any reusable helper (skill) you wrote or revised, with one line on what it does. These notes are study data.
- Stop early when the policy passes 3/3 on two consecutive iterations. Otherwise use the budget.
- At the end, write `FINAL` (a file holding only the iteration number) to name the iteration whose policy should be evaluated on the held-out test set. Choose by its dev results; usually that is the best and most recent. Then report: final iteration, the dev results per iteration (for example `1: 0/3, 2: 1/3, ...`), the main failure modes, and one paragraph on how your level helped or got in your way.

## Level-specific notes

- **synergies:** call `api.describe()` once (e.g. via `api.note(api.describe())` in a first iteration, or reason from the docstring) to learn what each coefficient does; coefficients are in standard deviations of other tasks' expert postures.
- **layered:** every layer is available (joint targets, fingertip targets, synergy coefficients, contact primitives). Use whichever you judge best, mix freely, and say in NOTES.md which layers you used for which phase and why. Your usage is counted automatically.
- **force:** the layered level (every layer, used freely) plus a closed-loop grip-force servo: `api.grip(obj, {finger: newtons})` keeps each named pad pressing on `obj` at that contact force, adjusting it every 20 ms along its closing direction (by default toward the other gripped pads, i.e. along the pinch line) while any later motion runs, until `api.ungrip()`. `api.grip_state()` reports targets, measured forces and servo travel. Read the `Force` class docstring in `dexagent/levels.py`. Say in NOTES.md where you used force control and whether it mattered.

## Two more rules
- Run one iteration at a time; never start an iteration while another of yours is still running.
- Do not use results from an iteration that did not complete (no summary.json).
