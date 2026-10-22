# pen_writing_grasp at level synergies


## Skills written (policy.py)
- `settle(api, s)`: command synergy coeffs, wait, return measured palm-frame pads.
- Synergy-space pad IK (built myself, level has no fingertip IK): central-difference pad Jacobian measured in free air at start (2k probes), damped Gauss-Newton `ik_step` toward thumb/index pad targets with a weak "keep other pads high" term, Broyden rank-1 updates `broyden` while refining by measurement (`solve_pads`).
- `pen_state(api, hseg)`: pen lowest point / tip / top / lean from pose; hseg calibrated once the pen has settled in the cup.

## Iter 1: 0/3
- Online synergy IK (measured pad Jacobian, k=6) found a clean thumb-index pinch (pads 45 mm open -> squeeze); grasp 1.6 N/pad, pen lifted 100 mm in all 3 cases. Pen hangs ~9 deg leaned in the pinch.
- Sev 0: dragged 120 mm with tip estimate on the table, lean ~10 deg, held at end, but scorer contact fraction only 0.44 (6 strokes): too-light touch (2.5 mm press) -> chatter. Sev 0.5/1: pen fell out of the 2-finger pinch during the drag (0.5 s in / after 110 mm); grip force had decayed 1.4 -> 0.4 N in sev 0.
- Learned: s-space pinch works (2 pads only, other fingers unused); the 2-pad pinch is a weak gimbal, so drag torque must be small and the tip should trail.

## Iter 2: 0/3
- Reused iter-1 synergy postures (no online calibration, -14 s sim), squeeze 9 mm (3-4 N/pad), press 4 mm, dragged toward the pen's lean direction (turned out to be along the pinch axis).
- Sev 0.5/1: pen buckled out of the pinch within 0.3 s of the drag (lean 10 deg + 4 mm press -> large axial load on a tilted pen in a 2-pad gimbal). Sev 0: 62 mm stroke but contact fraction 0.63, then pen dropped on the final lift (grip decayed 3.3 -> 0.85 N during the drag).
- Learned: scorer contact = any impulse >= 1e-6 N s, so the low fraction is real chatter (pen ratchets up in the grasp until the tip barely touches). Dragging along the pinch axis is worse than perpendicular to it. Lean x press force = buckling torque on the pinch.

## Iter 3: 0/3
- Grasped lower (42 mm below top), squeeze 12 mm, then rotated the hand about the pinch point to stand the pen upright, closed-loop press. The lower grasp left the pen 14-18 deg tilted with weaker grip (1.1 N in sev 1) and the pen dropped during the uprighting rotation / carry in all 3 cases. Regression: reverted.
- Insight: a pen dragged with its tip on the table, held in a near-frictionless gimbal (2-pad pinch), only slides steadily when its lean exceeds atan(mu=1) = 45 deg (two-force strut); below that the friction torque rotates it in the pinch or unloads the tip (=> chatter of iters 1-2).

## Iter 4: 0/3 (last of budget; run after the pause restart)
- Fixed the untested iter-4 plan's sign: tip LEADING the drag (pushed strut at lean ~50-52 deg > atan(mu)=45 deg, hand height closes the loop on lean: raise -> steeper, lower -> flatter), drag direction taken from the measured tip-top vector, pivot = measured thumb/index pad midpoint, 65 mm stroke, conditional re-squeeze before the final lift.
- Result: rolling the hand 52 deg about the pinch axis only leaned the pen 20-28 deg (it swung back toward vertical under its own 5 g weight: the 2-pad pinch is a near-free hinge about the pinch axis, torsional friction 0.005 is not enough). The one-shot lean correction (+25-30 deg more roll) then dropped the pen in sev 0 and 0.5; sev 1 hit arm IK (5.7 mm residual) during the carry with the rolled hand.
- Learned: in this hand a thumb-index pinch cannot set the pen's angle at all about the pinch axis, so any strategy needing a controlled lean (strut drag, uprighting) needs a third contact (middle finger / palm side) to make the grasp rigid. With synergies that third contact would need its own online pad-IK search; I ran out of iterations before trying it.
- Skill revised: tilt-about-measured-pinch-point (rotate palm about Rd[:,2] around tips_world midpoint) + one-step closed-loop lean correction; harmful with a hinge-like 2-pad grasp.

## FINAL = 1
All iterations 0/3. Iter 1 came closest on dev: sev 0 lifted, drew a 30 mm stroke and held the pen at the end; it missed only the 80 % contact-fraction rule (0.44).
