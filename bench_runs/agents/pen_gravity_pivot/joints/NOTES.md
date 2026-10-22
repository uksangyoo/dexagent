# pen_gravity_pivot at level joints


## iter 1: 0/3
- Policy: in-air calibration sweep (single-joint sweeps + 6 random postures, measured q -> api.tips()), fit a product-of-exponentials FK model per finger (fit residual ~0, chain order 0..3 for all fingers), own least-squares IK on that model; thumb-index pinch near one end at hand.py's PINCH_CENTRE, middle pressing on top of the short end as a lever, then "table trick" pivot.
- Result: hand knocked the pen off the rest during descent/closing (pen flung off the table). Offline (with my fitted model, no CPU kinematics) I found the thumb's links lie across the pen's long part in that pinch, and the thumb pad can only face +z at x~0.10, z<0.09.
- Skill: `Finger` POE model + `fit_finger` (calibrate FK from tips) + `ik` (position + soft pad-direction objective via last-joint flex direction). This is "IK built inside policy.py from api data" (rule 1 note): models fitted from my own iteration-1 calibration log and hard-coded from iter 2 on.

## iter 2: 0/3
- Redesigned offline with my fitted model: tripod grasp (thumb on one side, index opposite it, middle 24 mm further along on the index side), pen at 20 deg in the palm y-z plane (palm yaw = pen yaw + 20 deg), 100 mm below the palm, gripped 40 mm from the end away from the rest. Then carry clear of the rest, lower to 40 mm above the table, release the middle (thumb-index = free hinge, far end drops onto the table), raise the hand straight up so the pen turns in the pinch.
- Result: grasp + level hold work in all 3 (2.1-2.3 s level, ~99 mm lift); the table-assisted pivot reached 90 deg palm-frame pivot in the nominal case, but the pen then slipped out of the pinch: thumb/index forces fell from ~0.9 N to 0.1-0.16 N once the middle let go.
- Learned: middle release unloads the thumb-index pinch (thumb was balancing index+middle); the friction tripod holds the pen level easily; the table trick gives a swing-free pivot.

## iter 3: 0/3
- Change: thumb-index squeeze 8 mm (from 4) at the moment the middle lets go, so the pinch alone holds the pen.
- Result: pinch force after release 1.1-1.4 N (was 0.1-0.16), but in all 3 cases the pen squirted out of the pinch at 37-62 deg of pivot while the hand rose (film): a hard pinch resists the turn (wide pads / short end meeting the finger links) and levers the pen out. Level hold still fine (2.2 s).
- Learned: grip force trades off hinge freedom vs. holding; weak (0.15 N) pinch let the pen reach vertical in iter 2 nominal but dropped it later.

## iter 4 plan
- Medium squeeze (5.5 mm) while the pen turns, lift in 5 mm steps with feedback on the pen's tilt from vertical, squeeze to 8 mm once it is within 12 deg of vertical (or near the top), then finish lifting.

## iter 4: 0/3 (budget exhausted)
- Change: 5.5 mm thumb-index squeeze during the turn, stepwise 5 mm lift with feedback on tilt from vertical, squeeze to 8 mm once within 12 deg of vertical.
- Result: best so far. In all 3 cases the pen turned in the pinch to within 4-5 deg of vertical (max palm-frame pivot 80-89 deg, level hold 2.3-2.5 s, lift ~99 mm). Squeeze reached 0.8-2 N, but the pen still left the fingers during the last ~10 cm of lift, so the final 1 s window failed (pen on the table).
- Hypothesis (untested): squeezing hard on a vertical capsule with rounded pads pushes it out along its axis, or the short end above the pinch catches on the finger links. Next things to try: keep the medium squeeze and bring the middle back as a third contact on the hanging pen instead of squeezing harder, and lift more slowly.
- FINAL = 4 (all iterations 0/3; iter 4 got furthest).
