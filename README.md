# dexagent

Agent-authored dexterous manipulation for a UR5e arm with a 20-DOF Tesollo DG5F-S hand. A language/video goal becomes an agent-authored policy graph. A selected hand abstraction lowers each graph command to 26 bounded motor commands. Drake SAP hydroelastic contact integrates the dynamics on CPU, and measured feedback returns to the agent for revision.

## Layout

```
dexagent/            shared stack
  sim/               Drake workcell: model preparation, hydroelastic backend, controllers
                     (the three hand abstractions), contact graph, seating, thread geometry
  planning/          hand/arm geometry, initial states, kinematics, arm/grasp/workspace plans,
                     graph-policy compiler and request/feedback workflow
  agentic/           goal -> graph -> abstraction -> motors runner and bounded revision loop
  render/            recorded-state export and Blender Cycles/OptiX renderer
  provenance.py      recorded-artifact hashes across the path relocation
tasks/
  screwdriver/       prepared-grasp HOM study (study/) and continuous pickup-to-seat runner (sequence/)
  cube/              cube pickup, in-hand rotation and hold
  crate/             bimanual crate flipping on RoboSimStudio/Newton (work in progress)
tests/               unit tests for contracts, compilers, kinematics and seating
docs/                historical design notes from the October 12 Newton/GaP prototype
reference/           human reference videos and the HOM paper (arXiv:2609.25619)
third-party/         tesollo_model and robosimstudio (submodules); UR5e and crate-washing assets (vendored)
```

Generated outputs go to `artifacts/`, which is not tracked. The published site in `screwdriver_site/` is its own repository.

## Setup

```bash
git clone --recursive <repo-url>        # or: git submodule update --init --recursive
python -m venv --system-site-packages .venv-drake
.venv-drake/bin/pip install -e .
```

The recorded environment is Python 3.12, Drake 1.58.0, MuJoCo 3.10.1, NumPy 2.5.1, SciPy 1.18.0, trimesh 5.0.0rc1, tetgen 0.8.4 and pymeshlab 2025.7. RTX rendering needs a separate `bpy==4.5.10` interpreter: set `DEXAGENT_RENDER_PYTHON` to it. All GPU work on this workstation is restricted to physical GPU 1.

```bash
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1
.venv-drake/bin/python -m unittest discover -s tests -t .
```

## Tasks

| Task | Entry points | Status |
| --- | --- | --- |
| [Screwdriver](tasks/screwdriver/README.md) | `tasks.screwdriver.study.experiment`, `dexagent.agentic.run`, `tasks.screwdriver.sequence.pickup` | Frozen prepared-grasp comparison complete; continuous pickup-to-seat incomplete |
| [Cube](tasks/cube/README.md) | `tasks.cube.workflow`, `tasks.cube.run` | One continuous pickup -> 30° rotation -> hold success (relative-contact control) |
| [Crate](tasks/crate/README.md) | `tasks/crate/ctl up --scene tasks/crate/scene.py` | Work in progress |

The workflow for agent-authored policies (request, compile, run, feedback, revision loop) and the three hand abstractions are documented in [dexagent/agentic/README.md](dexagent/agentic/README.md).

## Current results

**Screwdriver, prepared grasp.** A frozen six-second, three-interface comparison gives 39.47° / 0.116 mm for fingertip paths, 26.10° / 0.079 mm for rigid object twist, and 154.10° / 0.434 mm for HOM-inspired relative-contact control. Three starting finger clearances preserve this ordering. These are prepared-grasp trials, not full pickup-to-seating successes. Material parameters are uncalibrated. Methods, sensitivity checks and limitations are in [tasks/screwdriver/HOM_REVISION.txt](tasks/screwdriver/HOM_REVISION.txt).

**Cube.** One continuous pickup -> 30° in-hand goal -> stable hold: 3.94° final error, 77.52 mm lift, 0.312 s accepted hold, 0.014° palm rotation. A matched rerun reproduced the trajectory. The fingertip-path and rigid-object-twist adapters collided from the same initial state. Perturbed-start and quarter-turn goals do not yet pass.

The current examples were authored by a coding agent in conversation. A separate live model provider has not been configured or evaluated.

## History

The October 12 Newton/GaP contact-grammar prototype (`dexagent/` grammar and controllers, `dextrain/` PPO curriculum, review site tooling) and the superseded screwdriver pipelines (controlled slip, full motion, physical task, screw-in) were removed on October 20. The screwdriver records used an overlapping wrist collision capsule and are not evidence of realistic manipulation. Both are recoverable from git history: `e460c20` holds the October 12 stack, and `af43140` is the last tree with `screwdriver_lab/` and `crate_lab/` before the restructure. Two inputs recorded by the old physical-task pipeline are kept as frozen data in `dexagent/sim/data/`.
