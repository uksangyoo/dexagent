# dexagent

Dexterous manipulation for a UR5e arm with a 20-DOF Tesollo DG5F-S right hand, simulated on the GPU with RoboSimStudio: Newton collision and the MuJoCo-Warp solver at 1 kHz, with 50 Hz control, rendered by RoboSimStudio's OVRTX (RTX) path tracer. Each task is a RoboSimStudio scene file with scripted routines. The tasks either reproduce a human clip in `reference/hand_demos` or are in-hand manipulation tasks of their own.

## Layout

```
dexagent/            the table workcell on RoboSimStudio: robot MJCF and kinematics, scene builder,
                     hand controller, objects, headless runner (see dexagent/README.md)
tasks/
  live               launch a live RoboSimStudio page for one scene folder
  hand_demos/        scenes that reproduce the clips in reference/hand_demos
  inhand/            in-hand manipulation scenes without a clip
reference/           human reference videos and the HOM paper (arXiv:2609.25619)
docs/                historical design notes from the October 12 Newton/GaP prototype
third-party/         tesollo_model and robosimstudio (submodules); UR5e assets (vendored)
```

Each scene folder holds `scene.py` (the scene and its `routines`), any helper modules beside it, and `ref/` stills from its clip. Live pages write to `run/`, and routines write to `results/`. Films go to `artifacts/`. None of these is tracked.

## Setup

```bash
git clone --recursive <repo-url>        # or: git submodule update --init --recursive
# The environment is RoboSimStudio's own lockfile (Newton 1.7, MuJoCo 3.15, MuJoCo-Warp, OVRTX 0.4.1), in .venv here.
(cd third-party/robosimstudio && UV_PROJECT_ENVIRONMENT=$PWD/../../.venv uv sync --frozen --python 3.12)
uv pip install --python .venv/bin/python -e .
# The locked warp-lang is a CUDA 13 build. Drivers older than CUDA 13 (this workstation: 570, CUDA 12.8) need the cu12 wheel:
uv pip install --python .venv/bin/python "warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
```

All GPU work on this workstation runs on physical GPU 1. `CUDA_VISIBLE_DEVICES=1` keeps both the physics and the OVRTX renderer on it. Without the mask, OVRTX opens a context on GPU 0 as well, even with `ROBOSIMSTUDIO_DEVICE=cuda:1`.

```bash
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. .venv/bin/python -m dexagent.run tasks/hand_demos/cube_pinch --camera reference   # RTX film in artifacts/rss/
PYTHONPATH=. .venv/bin/python -m dexagent.run tasks/hand_demos/cube_pinch --renderer tiled      # collider ray tracer, faster
tasks/live tasks/hand_demos/cube_pinch up     # live page with RTX snapshots; rebuilds when scene.py is saved
```

## Tasks

| Scene | Clip | What the routine does |
| --- | --- | --- |
| `hand_demos/cube_pinch` | IMG_4772 | Pinch a 40 mm cube, lift it, turn it 35° and put it back |
| `hand_demos/cube_palm` | IMG_4782 | Pick the cube up, roll it onto the open palm and place it |
| `hand_demos/pen_gravity_pivot` | IMG_4774, 4775 | Hold a pen level, loosen the grip so it swings down under gravity, catch it |
| `hand_demos/pen_pickup_pivot` | IMG_4773, 4780 | Pick a pen off the table and pivot it upright |
| `hand_demos/pen_finger_roll` | IMG_4776, 4777 | Lift a pen from a cup, roll it between the fingers and swing it |
| `hand_demos/pen_writing_grasp` | IMG_4783 | Take a tripod writing grasp and draw a stroke |
| `hand_demos/pen_table_slide` | IMG_4781 | Pinch a pen and let it slide through the fingers to the table |
| `hand_demos/scissors` | IMG_4778, 4779 | Pick up hinged scissors, open and close them |
| `inhand/cap_twist` | none | Unscrew a jar lid with finger strokes; the palm holds still |
| `inhand/card_pickup` | none | Slide a flat tile to the table edge and pinch it off |
| `inhand/baoding` | none | Palm up: circle two balls around each other |
| `inhand/cube_reorient` | none | Palm up: turn a cube 90° on the palm with finger pushes |

Per-scene results and known failures are in [tasks/hand_demos/README.md](tasks/hand_demos/README.md) and [tasks/inhand/README.md](tasks/inhand/README.md).

**Physical contract.** A routine writes only the 26 robot joint targets. Object poses, velocities and forces are read for planning and scoring and never written. The routines read object poses from the simulator, so they are privileged-state scripts, not perception policies.

