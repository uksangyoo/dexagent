# In-hand tasks

In-hand manipulation scenes for the UR5e + DG5F-S right hand that have no human clip. They are built on the same table workcell as `tasks/hand_demos`, follow the same physical contract, and are run the same way:

```bash
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. .venv/bin/python -m dexagent.run tasks/inhand/<scene> --camera <camera>   # RTX film in artifacts/rss/<scene>/
tasks/live tasks/inhand/<scene> up
```

| Scene | Camera | Task | Result (Newton 1.7, October 21) |
| --- | --- | --- | --- |
| `cap_twist` | `jar_view` | Unscrew a jar lid (`jar.xml`, a lid on a friction hinge) with thumb-index-middle strokes while the palm stays still | **success**: 414° in 7 strokes, palm yaw change 0.36° |
| `card_pickup` | `edge` | Drag a 3 mm tile over a book's edge with two fingertips (extrinsic dexterity), then pinch the overhang and lift | **success**, 7 of 7: overhang 38 mm, lift 101 mm |
| `cube_reorient` | `palm` | Palm up: turn a 40 mm cube 90° about the palm normal by finger gaiting | **success**, 2 of 2 final (4 of 4 overall): 87–88° in 6 gaits, cube stays flat on the palm |
| `baoding` | `palm` | Palm up: circle two balls around each other | **fails**: a ball drops after 5–87° of exchange over three runs (best ever 211°, on Newton 1.6) |

## Notes

- **`cap_twist`.** A released finger must clear the rim, or the return stroke turns the lid back. Released pads sit 18 mm outside the rim and 8 mm higher, and the hand lifts 60 mm before it opens.
- **`card_pickup`.** The drag stops at a 12 mm overhang target. Stick-slip and the release carry the tile about 26 mm further. A larger target ends past the tile's half-length, and the tile tips off the book before the pinch.
- **`cube_reorient`.** The fingers have to be chosen with care:
  - With the cube mid-palm, the thumb's middle and distal links pass through it, and only a 45° thumb-index angle gives a collision-free pinch.
  - So each gait pushes with the thumb and little finger as a couple. The thumb takes the thumb-side face at its fingertip end, and the little finger takes the opposite face at its wrist end. Each push swings 20°, then the fingers back off and lift.
  - The cube drifts about 20 mm toward the fingers over the run, and later gaits vary between runs: sometimes the ring finger touches instead of the little finger.
- **`baoding`.** The cupped DG5F-S palm is not a bowl:
  - The heel is open: a ball spills over it 5 mm above the palm floor.
  - The thumb side is open under the thumb.
  - The flat floor is about 50 mm wide, too narrow for two balls to pass side by side.

  A static cup with the little-finger metacarpal raised reached 27° before a ball rolled over the heel. A plausible next step is a third, passive fixture (a shallow dish held in the palm) or a different hand posture search.
- **Friction is the larger of the two values.** Newton combines friction by taking the larger of the two geoms' coefficients (`newton/_src/solvers/mujoco/kernels.py`). An object authored with friction below the fingertips' 0.9 still gets 0.9 against the hand. Lower object-hand friction needs geom priority, as in `hand_demos/pen_gravity_pivot/grip.py`'s `smooth_pen`, or a lower hand value.
