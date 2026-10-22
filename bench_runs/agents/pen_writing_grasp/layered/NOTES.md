# pen_writing_grasp at level layered


## iter 1: 0/3
Plan: palm-down thumb-index pinch (fingertips layer `move_tips`, pinch centre from hand.py), other fingers `tuck` (contacts layer), arm via `carry`; lift along the pen axis, pitch hand 25 deg so the tip leads, closed-loop touchdown/stroke on the pen's lowest point (per-step `carry`).
Result: pen lifted out of the cup in all cases (max tip 111-117 mm, grip forces 3-5 N), but it rotated in the 2-point pinch (lean 8->20 deg after lift) and fell out during the pitch/placement; one case hit arm IK failure chasing the swinging pen.
Learned: a 2-pad pinch is a hinge about the pinch axis (only torsional friction) and weak about the other in-plane axis (couple arm = pen diameter); squeeze 7 mm past the surface already gives 4-5 N. The api log keeps only the last 60 lines, so per-step carry calls flush notes -> buffer notes and emit at the end.

## iter 2: 0/3
Changed grasp to a 3-finger chuck (fingertips layer `move_tips`): hand rolled thumb-down (palm y = world down, aligned to the pen axis), pen along palm y, thumb pad (y=.029) on the wrist side opposing index (y=.040) and middle (y=.018) pads; ring/little `tuck`ed. Notes buffered and flushed at the end.
Result: nominal and sev-0.5 grasped (thumb/index/middle ~4.3/2/2.4 N) and lifted 112 mm; nominal even drew a passing stroke (path 27.7, extent 25.5, contact 0.98) but in both the pen rotated out of the grip during the stroke (tip friction torque, lever ~90 mm vs 22 mm finger spacing) so nothing was held at the end. Sev 1 (long pen, 8 deg lean): pen knocked over during the approach (pen 33 mm off in palm x, 33 deg tilted) -> no grasp.
Learned: tip friction torque is the enemy of any finger grip; the chuck is stiff in lift/transport but not against a vertical-pen drag.

## (resume) iter_3 was killed while queued (no summary, not counted). Next run is iter_4 = budget 3 of 4.

## iter 4 (budget 3): 0/3
Changed: approach turns the hand 0.16 m above the pen before descending along its axis (fixes sev-1 knock-over); after lift the pen is tilted closed-loop on its measured axis (minimal rotation about the grip, arm layer) to 50 deg lean with the tip LEADING the stroke (with mu=1 the sliding reaction is then ~along the pen axis -> no twisting torque, no jamming); stroke height regulated on the pen's axial slip in the grip (grip compliance as a force sensor); IK-failure fallback to 43/36 deg (`safe_carry`: catch ValueError, snap command to the measured palm).
Result: grasp+lift now 3/3 (all cases ~4/2/2.4 N, tip 111-113 mm). Sev 0.5 reached 50 deg and drew a valid stroke (path 65.7, extent 33.7, contact 0.96) but the grip had loosened during the tilt (index 0.47 N) and was gone by touchdown -> pen not held at the end. Nominal and sev 1: arm IK failed (residual 4-6 mm) at the tilt and at the hover: the stroke area was ~0.8 m from the base (0.30,-0.58), near full UR5e reach; sev 1 dropped the pen.
Learned: workspace, not physics, limited the tilt; reorienting the hand loosens a 5 mm-squeeze chuck -> re-squeeze around the measured pen after every reorientation. The axial-slip signal is confounded once the pen rotates in the grip.
Skill: `safe_carry(api,P,R,s)` arm move that survives IK failure; `align(a,b)` + about-grip rotation = closed-loop pen-axis alignment.

## iter 5 (budget 4): 0/3 (regression)
Changed: stroke start moved to the -py side of the cup, 3 cm toward the base (to stay in arm reach), pen lowered to 70 mm before tilting, tilt target 45 deg (fallbacks 38/30) over 2 s; squeeze 7 mm (forces now 5.5-6/2.7-3.4/3.1-3.8 N); `regrip()` re-centres the chuck pads on the measured pen (fingertips layer) after the tilt, after the hover and before the final lift; stroke height held in a band on the pen's lowest point (axial slip only as a safety stop).
Result: the grip was lost DURING THE TILT in all three cases: the arm swung violently there (film: arm whips up, pen flung; nominal also had IK failures at x~0.65). Sev 0.5 kept a loose pen and still drew a valid stroke (path 44, extent 32.5, contact 0.98) but nothing was held at the end. Regrip was skipped because the pen axis had already left palm y.
Learned: the hand reorientation (arm layer) is the weak link of this task; near the edge of the UR5e workspace the arm IK jumps branches or fails, and a finger grip cannot survive that. Strokes themselves are easy (even a loose pen passes the stroke rule); holding through reorientation is the problem. A better design would tilt only a little (or not at all, e.g. thumb-down chuck kept with a small lean) and work closer to the base.
FINAL: iter 4 (grasp+lift 3/3, one valid stroke; iter 5 regressed in all cases).
