# scissors at level layered


## iter 1 — 0/3
Plan: wide thumb/index pinch across loop A (thumb behind back wall, index in front of front wall) + ring finger in hole B swung about the hinge with contacts `rotate`; placement chosen by an in-policy grid search over the palm-frame anchor using `api.reach` (fingertips IK). Layers: joints (seed pre-shape), fingertips (placement/squeeze), contacts (rotate, tuck), arm (carry).
Result: nothing lifted. (1) my scale estimate from pivot height was wrong: start pivot height is 6.8 mm * size exactly (before settling; settles to ~4.6 mm), so s was clipped to 1.1 and the index landed on top of the front wall (30 N). (2) the search picked the lowest palm (a_x=0.06): tucked little touched the table and the thumb stayed 24 mm above its target, never touching. (3) sev-1 case: ARM IK FAILED on the way to a palm yaw of 1.05 (scissors yaw +20 deg) near the robot base; a failed move is unrecoverable.
Skill: `_search(api, ...)` grid search of palm-frame anchor + hole-B finger choice via api.reach (IK only, no sim time).

## iter 2 — 0/3
Changes: scale s = start pivot height / 6.8 mm (exact), palm yaw only 35 % toward the scissors yaw (no arm IK failure this time), search prefers the highest palm (a_x=0.10).
Result: still no lift. Thumb now reaches 9 mm and touches the back wall (4 N), middle finger sits in hole B (9 mm high, free). But the index sat 15 mm high with 7-46 N on the scissors: the loop shank runs from the loop's OUTER front corner diagonally to the pivot, so the index at loop-local y=18 mm landed on the shank. Pinch lost on lift (forces 0 at +30 mm). Thumb/middle pressed the table with 10-25 N during squeeze (pads at 8 mm too low).

## iter 3 — 0/3 (notes written after the restart, from the iter_3 logs)
Changes: pinch line moved to loop-local y = 12 mm (to dodge the shank), squeeze retried 3x with growing squeeze, palm height +.
Result: no lift in all 3 cases. Index pad again stuck ~17 mm above the table on top of something with 30-60 N, the thumb pushed the back wall (14-43 N) and slid the scissors 2-5 mm per try; on lift all forces went to 0. Free pads sit ~2.5-3.5 mm above their targets (thumb 12.5 mm for a 10 mm target), i.e. at/above the 10 mm wall top.
Learned (from a frame grabbed from the iter-1 film, scratch/crop0.png): each shank is a ~9 mm red capsule from the loop's FRONT wall at about y = +-18 mm (not the outer corner) to the pivot. So y = 12 and 18 both put the index pad on the shank; the free part of the front face is the OUTER part of loop A (y > ~29 mm).

## iter 4 — 0/3 (last iteration; budget used)
Changes: pinch line moved OUT to loop-local y = 35 mm (clear of the shank capsule), anchor search widened in palm y (0-50 mm), pad height 9.5 mm*s, palm lowered by the free thumb's measured height error after descent (bias ~1-3 mm), lift check on loop A rise, carry height +150 mm.
Result: first time the pinch engaged in all 3 cases (thumb behind back wall + index on front wall, ~3 N each; the ring finger sat free in hole B). At +30 mm loop A rose 22-23 mm (pivot 9-14 mm, blade tips still on the table), but the scissors ROLLED 16-20 deg about the pinch line toward loop B (R[2,1] ~ 0.28-0.33) because the CoM (~pivot, y = 0) sits ~35 mm off that line, and they dropped during the carry to +150 mm. The cycles and set-down then ran with an empty hand (hinge never moved), so no cycles counted.
Learned: placement is now right; what's missing is holding force and roll support. Squeeze was only ~3 N (kp 3 / 0.65 N m fingers, 5 mm squeeze). Next would be: bigger squeeze (8-10 mm), check thumb/index forces > ~2 N and roll < 10 deg before the high lift, lift slowly in steps, and give loop B a second support (hole-B finger pressing against B's outer wall, or a middle-finger pinch on loop B) so the pair cannot roll about the A pinch line.
Layers used (all iterations): joints (seed pre-shape, little finger tucked), fingertips (placement/squeeze/release via move_tips, plus api.reach as IK for the in-policy anchor search), contacts (tuck, rotate of the hole-B finger about the hinge axis), arm (carry).
Skills: `_search` (in-policy grid search of palm anchor + hole-B finger by api.reach), height-bias correction from a free fingertip's measured z.
