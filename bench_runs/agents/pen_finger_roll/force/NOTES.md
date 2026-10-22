# pen_finger_roll at level force


## iter 1: 0/3
- Plan: thumb-index pinch (line along palm z) 15 mm below pen top, force-servo grip 1.2 N (`api.grip`), lift 150 mm, then swing/roll by opposite pad shear (move_tips) closed-loop on the measured pen angles. Helpers: `W(gen)` wraps any motion to run a checker-replica monitor each step (rest axis, tilt, accumulated twist); `Shear` = closed-loop opposite-pad shear servo with online sign/gain calibration.
- Result: grasp + lift work in all 3 (grip settles at ~1.1 N, 4-5 mm servo travel, pen out with 88 mm clearance). Swing reached +16 deg with ~4.2 mm shear (gain ~4 deg/mm, i.e. pads behave like rigid rotation about the pinch, not rolling at 15 deg/mm), then on reversal the pen snapped to +70..86 deg and was ejected (force servo ran to its 15 mm travel limit, force 0). Roll never tested (pen gone). Incidental roll ~13 deg during swing.
- Learned: force servo + large shear = pen squeezed out once it slips; need slower swing, rigid rotation of pads about the pinch point, travel limit, and time-series logging.

## iter 2: 0/3
- Changed: roll first via opposite pad shear along palm y (closed loop on accumulated twist), swing via rigid rotation of both pad targets about palm y through the pinch centre; grip 1.5 N, max_travel 10 mm; compact time-series notes.
- Result: roll works: calib gain -13..-18 deg/mm of shear (pure rolling, ~1/r), roll range 31-36 deg reached in all 3 cases. But at |shear| ~3.4 mm the pen was squeezed out every time: grip-servo travel climbed 6 -> 10 mm (cap), force -> 0, pen dropped. Swing never tested.
- Learned: the default grip closing direction (toward the other pad) points diagonally once pads are sheared, so the pads slide past each other and eject the pen. Keep shear <= ~2.5 mm and pin the closing direction along the unsheared pinch line (re-issue api.grip with `directions` from the current pen pose every update).

## iter 3: 0/3
- Changed: re-issued `api.grip(..., directions=...)` before every pad update to pin the closing direction on the unsheared pinch line; roll shear capped at 2.4 mm; travel cap 15 mm.
- Result: worse: roll gain halved (6-8 deg/mm) and the pen was lost in all 3 cases at only 0.4-1.9 mm shear (roll range 29-33 deg reached first). In every loss the swing tilt creeps +3..7 deg first, then the pen snaps to ~85 deg tilt within 0.2 s while servo travel runs to the cap.
- Learned: the loss mode is the squeeze instability of a 2-pad pinch on a straight pen: once the pen is tilted relative to the pinch line, the squeeze couple (~2hF sin(rel)) grows with the tilt and beats the friction couple (~2r mu F, both scale with F, so grip force does not help). Pinned closing directions did not help. A 10 Hz loop is too slow to catch it.

## iter 4: 0/3
- Changed: 50 Hz pad controller (one 20 ms move_tips per step). Swing first: pad rotation phi about palm y through the pinch centre = -(measured tilt change) + a bounded PI lead, so the pinch stays near-aligned with the pen while friction carries it. Roll after: one-way opposite y-shear at 1 mm/s with the same tilt-following stabiliser. Back to default grip directions.
- Result: the pen was lost during the first swing in all 3 cases, at only +7..+10 deg of tilt (phi -10 deg, PI lead saturated at -5 deg); servo travel ran past 10 mm, force 0. Incidental roll ranges were 42-55 deg, so roll is the easy part.
- Learned: a thumb-index pad pinch cannot carry the pen more than ~10-16 deg about the axis normal to the pen and the pinch line. A pad lead of a few degrees already starts the squeeze-out. The swing needs a third contact (e.g. a finger or the palm as a fulcrum along the pen) or a pinch line along palm y, so that the swing is a free hinge. I ran out of budget before I could try that.

## Helpers (skills) written
- `W(gen)`: wraps any api motion generator so that a checker-replica monitor (rest axis, swing tilt, accumulated twist, pen bottom) runs every 20 ms step.
- `Act` (iter 2-3): 1-D closed-loop pad servo on a measured pen angle, with online sign/gain calibration. Command is either opposite shear or rigid pad rotation about a palm axis.
- `pads()/tick()` (iter 4): 50 Hz composition of pad rotation (phi) and opposite shear (u) around the pinch centre, sent with move_tips.
- Own IK use: none beyond move_tips. Pad targets were computed in the palm frame from api.object_in_palm.

## Force control: where and whether it mattered
- `api.grip(pen, {thumb, index: 1.2-1.5 N})` after closing the pads to 2 mm standoff, kept on through lift and all in-hand motion. It made grasp + lift trivially robust (3/3 every iteration, no tuning of pad offsets: the servo absorbed a ~4-6 mm error in my assumed pad-surface offset). During the in-hand motions it did not help and probably hurt: when the pen began to squeeze out, the servo kept closing (travel to the cap) and so ejected it faster. Pinning the closing direction (iter 3) did not help either.
