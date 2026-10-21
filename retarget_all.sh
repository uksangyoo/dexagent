#!/usr/bin/env bash
# Run naive video→retarget→compare for every hand_demos scene / reference clip.
# Films land in artifacts/rss/<scene>/retarget_{from_video,compare}_<clipstem>.mp4
set -euo pipefail
cd "$(dirname "$0")"

export PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
# This box has one GPU. Do not inherit CUDA_VISIBLE_DEVICES=1 from the shell —
# that hides the 5090 and Warp falls back to CPU.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:third-party/robosimstudio

# scene_dir | reference clip  (all pairs from README / scene.py CLIP[S])
EXAMPLES=(
  "tasks/hand_demos/cube_pinch|reference/hand_demos/IMG_4772_cube_pinch_lift.mov"
  "tasks/hand_demos/cube_palm|reference/hand_demos/IMG_4782_cube_palm_roll.mov"
  "tasks/hand_demos/scissors|reference/hand_demos/IMG_4778_scissors_open_close_a.mov"
  "tasks/hand_demos/scissors|reference/hand_demos/IMG_4779_scissors_open_close_b.mov"
  "tasks/hand_demos/pen_pickup_pivot|reference/hand_demos/IMG_4773_pen_pickup_pivot.mov"
  "tasks/hand_demos/pen_pickup_pivot|reference/hand_demos/IMG_4780_pen_pickup_stand.mov"
  "tasks/hand_demos/pen_gravity_pivot|reference/hand_demos/IMG_4774_pen_gravity_pivot_a.mov"
  "tasks/hand_demos/pen_gravity_pivot|reference/hand_demos/IMG_4775_pen_gravity_pivot_b.mov"
  "tasks/hand_demos/pen_finger_roll|reference/hand_demos/IMG_4776_pen_finger_roll.mov"
  "tasks/hand_demos/pen_finger_roll|reference/hand_demos/IMG_4777_pen_swing.mov"
  "tasks/hand_demos/pen_table_slide|reference/hand_demos/IMG_4781_pen_table_slide.mov"
  "tasks/hand_demos/pen_writing_grasp|reference/hand_demos/IMG_4783_pen_writing_grasp.mov"
)

n=${#EXAMPLES[@]}
i=0
for entry in "${EXAMPLES[@]}"; do
  i=$((i + 1))
  scene="${entry%%|*}"
  clip="${entry#*|}"
  stem="$(basename "$clip" .mov)"
  scene_name="$(basename "$scene")"
  tracks_out="$scene/tracks/from_video_${stem}.json"
  out_dir="artifacts/rss/${scene_name}"

  echo
  echo "=== [$i/$n] $scene_name ← $stem ==="
  "$PYTHON" -m dexagent.retarget.from_video \
    --scene "$scene" \
    --clip "$clip" \
    --tracks-out "$tracks_out" \
    --speed "${SPEED:-0.25}"

  # Keep per-clip films (from_video always writes the un-suffixed names).
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/retarget_from_video.mp4" ]]; then
    mv -f "$out_dir/retarget_from_video.mp4" "$out_dir/retarget_from_video_${stem}.mp4"
  fi
  if [[ -f "$out_dir/retarget_compare.mp4" ]]; then
    mv -f "$out_dir/retarget_compare.mp4" "$out_dir/retarget_compare_${stem}.mp4"
  fi
done

echo
echo "done: $n retarget examples → artifacts/rss/*/retarget_*_IMG_*.mp4"
