#!/usr/bin/env bash
set -euo pipefail
export PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
# Force GPU 0 — a leftover CUDA_VISIBLE_DEVICES=1 in the shell hides the only card.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:third-party/robosimstudio

# Full .mov → estimate → retarget → film
# --object is inferred from the scene folder (cube_pinch→cube, scissors→scissors).
"$PYTHON" -m dexagent.retarget.from_video \
  --scene tasks/hand_demos/cube_pinch \
  --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov \
  --speed 0.25

