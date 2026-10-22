# pen_writing_grasp at level force


## iter 1 — 0/3 (crash: numpy shape bug in stroke-start computation, my mistake)
- Plan: palm-down hand_down(0.7), middle/ring/little tucked (contacts `tuck`), thumb-index pinch (fingertips layer `move_tips`) centred on the pen axis point 22 mm below its top, computed in the *measured* palm frame; then force layer `grip(pen, thumb 2.5 N, index 2.5 N)`, lift 85 mm, carry to a spot ~85 mm from the cup, closed-loop touchdown + drag stroke (per-step `carry`), lift and hold.
- Result: grasp + lift worked on all 3 cases (tip 84-86 mm, grip settled at 2.2-2.5 N with 4-6 mm servo travel, pen lean in hand <=9 deg), then crashed on a (2,)+(3,) array bug. Held-at-end was true even so.
- Learned: PINCH_C (80,40,118) mm is accurate to ~1 mm in the measured palm frame; force grip closes robustly regardless of pen diameter. Added an offline mock api (mock/mockrun.py, no sim/kinematics, just fake poses) to smoke-test the policy's code paths before spending iterations.

## iter 2 — 0/3 (pen lost at touchdown in every case)
- Grasp/lift again fine (tip 84-86 mm). Film: during the 1 s descent to "12 mm above the table" the arm overshot, the tip hit the table, the pen pivoted about the thumb-index pinch axis and squirted out (lean 90, grip servo ran to max travel with 0 N).
- Learned: a two-pad pinch has ~no stiffness about the pinch axis (only torsional friction ~0.005 m x grip force), so any axial push on the tip buckles the pen out of the grip. Fast arm moves overshoot by >10 mm vertically (integral action). Force control keeps the pads loaded but cannot stop that pivot.
- iter 3 change: grip 4 N (more torsional friction), grasp 10 mm lower (shorter lever), staged slow descent with settling holds (25 mm, 6 mm), 3 mm/s closed-loop touchdown on the measured tip height with the command parked at the arm's measured height on contact; stroke lowers only when contact is lost and backs off when the pen-in-palm angle grows (pivot detector), 55 mm drag over 4 s. Skill: `pivot(ax_ref, pen_in_palm_axis(api))` = in-grip pen rotation [deg], used as a cheap overload sensor.

## iter 3 — 0/3 (pen dropped during the lateral carry, before any table contact)
- With grip 4 N and the pinch 32 mm below the pen top, the pen already pivoted 8-15 deg in the grip during the lift and fell out early in the sideways carry (sev 1 then hit an arm-IK error chasing the fallen pen). Iter 1/2 settings (2.5 N, 22 mm below top) had carried the pen fine in 6/6 cases.
- Learned: a stronger force-servo grip on a thin capsule between two round pads is *less* stable here (more squirt / pivot), not more; more force is not a substitute for a better-placed contact. The staged-descent/touchdown/stroke code never got exercised.
- iter 4 change: revert grasp to 2.5 N at 22 mm below the top; keep the iter-3 slow staged descent (25 mm, 7 mm, longer settles), gentle touchdown and pivot-guarded stroke.

## iter 4 — 0/3 (pen lost during the staged descent, before the touchdown loop)
- Grasp/lift fine again (tip 82-86 mm, lean 2-5 deg, grip 2.5 N). Film: while descending toward "25 mm above the table" the hanging pen's tip reaches the table and the pen pivots out of the pinch, so the touchdown and stroke logic never ran. Either the arm overshoots badly when descending (iter 2 showed the same), or the pen slips down in the grip during transport. Unresolved, and the budget is used up.
- Overall: the force layer (`grip`) made grasp formation trivial and robust (6/6 lifts across sizes and leans with no tuning), but the failure was in the grasp *topology*. A thumb-index pinch on a 7.5 mm capsule has no rotational stiffness about the pinch axis, and force regulation does not add any. A tripod with contacts spread along the pen (e.g. a sideways palm with index and middle on one side) would probably have been needed.
- FINAL = 4 (all iterations 0/3; 4 is the most complete and most recent policy).
