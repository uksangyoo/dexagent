# dexagent

Dexterous manipulation for a UR5e arm with a 20-DOF Tesollo DG5F-S right hand, simulated on the GPU with RoboSimStudio: Newton collision and the MuJoCo-Warp solver at 1 kHz, with 50 Hz control. Each task is a RoboSimStudio scene file with scripted routines. The tasks either reproduce a human clip in `reference/hand_demos` or are in-hand manipulation tasks of their own.

## Layout

```
dexagent/            the table workcell: robot MJCF and kinematics, scene builder, hand controller,
                     objects, headless runner, Cycles/OptiX snapshot renderer (see dexagent/README.md)
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

Python 3.12 for the main env. Use `uv venv` + `uv pip` (not `uv sync`): the optional `render` extra pins `bpy`, which only has 3.11 wheels, and `uv sync` resolves that extra into the lockfile.

```bash
git clone --recursive <repo-url>        # or: git submodule update --init --recursive
uv venv --python 3.12
uv pip install -e third-party/robosimstudio -e .
```

All GPU work on this workstation runs on physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`). Snapshots on the live page use Blender Cycles/OptiX in a separate 3.11 interpreter; point `DEXAGENT_RENDER_PYTHON` at it:

```bash
uv venv .venv-render --python 3.11
uv pip install --python .venv-render/bin/python bpy==4.5.10
export DEXAGENT_RENDER_PYTHON="$PWD/.venv-render/bin/python"
```

```bash
export PYTHON="$PWD/.venv/bin/python" CUDA_VISIBLE_DEVICES=1
PYTHONPATH=.:third-party/robosimstudio $PYTHON -m dexagent.run tasks/hand_demos/cube_pinch --camera reference
tasks/live tasks/hand_demos/cube_pinch up     # live page; rebuilds when scene.py is saved
```

### Naive retarget baseline (cube_pinch)

Existing scene twin + MediaPipe/HSV tracks from the matching `.mov`, then open-loop IK replay (objects teleported). See [dexagent/retarget/schema.md](dexagent/retarget/schema.md).

```bash
# Optional: synthetic tracks from the scripted routine (sanity)
$PYTHON -m dexagent.retarget.record_from_routine tasks/hand_demos/cube_pinch

# .mov → estimate → retarget → film (+ human|sim compare with hand overlay)
$PYTHON -m dexagent.retarget.from_video \
  --scene tasks/hand_demos/cube_pinch \
  --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov
# → artifacts/rss/cube_pinch/retarget_from_video.mp4
# → artifacts/rss/cube_pinch/retarget_compare.mp4

# Or play saved tracks headless / on the live page
$PYTHON -m dexagent.run tasks/hand_demos/cube_pinch --routine "retarget replay" --camera reference
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

