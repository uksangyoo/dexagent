# Track schema (`dexagent.retarget.tracks/v1`)

Per-frame hand and object poses in the **work-surface frame** (same as RoboSimStudio scenes), meters.

```json
{
  "schema": "dexagent.retarget.tracks/v1",
  "fps": 50.0,
  "frame_count": 100,
  "clip": "optional/path/to/source.mov",
  "meta": {},
  "frames": [
    {
      "t": 0.0,
      "palm": {
        "xyz": [0.5, 0.0, 0.9],
        "R": [[1,0,0],[0,1,0],[0,0,1]]
      },
      "tips": [[...], [...], [...], [...], [...]],
      "objects": {
        "cube": {
          "xyz": [0.55, 0.0, 0.77],
          "quat_xyzw": [0, 0, 0, 1]
        }
      }
    }
  ]
}
```

- `tips`: five fingertip pad centres in world, thumb → little.
- `palm.R`: 3×3 row-major rotation (palm frame: +z along fingers, +x out of palm).
- Quaternions are `(qx, qy, qz, qw)`.
- Optional `landmarks_uv`: MediaPipe 21 hand landmarks in normalized image coords `[0,1]` for overlays.
- Tracks **start at the first fully detected hand** (`meta.trim_start_s` is that instant in the source clip; track `t` is re-based to 0).

## Commands

```bash
# Synthetic tracks from the scripted routine (sanity)
python -m dexagent.retarget.record_from_routine tasks/hand_demos/cube_pinch

# .mov → estimate → retarget → film (+ side-by-side compare)
python -m dexagent.retarget.from_video \
  --scene tasks/hand_demos/cube_pinch \
  --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov
# → artifacts/rss/cube_pinch/retarget_from_video.mp4
# → artifacts/rss/cube_pinch/retarget_compare.mp4   (human|sim, time-aligned)

# Rebuild compare only (no sim)
python -m dexagent.retarget.compare \
  --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov \
  --tracks tasks/hand_demos/cube_pinch/tracks/from_video.json \
  --sim artifacts/rss/cube_pinch/retarget_from_video.mp4 \
  --out artifacts/rss/cube_pinch/retarget_compare.mp4

# Replay saved tracks on the live page / headless runner
python -m dexagent.run tasks/hand_demos/cube_pinch --routine "retarget replay" --camera reference
```

Limits: monocular table-plane lift, HSV cube, MediaPipe hands — naive baseline only.
