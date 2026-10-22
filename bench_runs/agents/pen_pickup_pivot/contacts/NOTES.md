# pen_pickup_pivot at level contacts


## iter 1: 0/3
Plan: thumb-index pinch across the pen 45 mm from centre (pinch = free hinge, pen has no torsional friction), contacts 35 deg above the equator so the shells clear the table, then swing the hinge on a closed-loop arc about the far end (stays on table) to vertical, then lift. Reusable helper: palm-frame pinch-location search with api.reach (picked PINCH_C+(5,0,0) mm, residual 0.1 mm).
Result: grasp forces thumb 2.6 N / index 1.2 N, but the pen dropped out during the first ~15 deg of the arc (forces 0 by 22 deg); my loop then chased a pen it no longer held and the arm IK failed. Lesson: pads above the equator are a wedge opening downward; the weak fingers let the pen squirt out. Need lower contacts and a loss check/regrasp.

## iter 2: 0/3
Changed: contacts 25 deg above equator, squeeze 3 mm, loss check per arc segment, regrasp onto the equator at 14 deg, retries with other elevations.
Result: the arc lifts the pen end nicely (th tracks target: 2.9/9.6/13.9 deg) but pinch force decays 0.9 -> 0.57 N as it rises (pen sliding down out of the downward-opening wedge), and the regrasp at 14 deg drops it. Retries knocked the pen away during descent (open thumb sweeps low) and thumb IK then failed (75 mm). Lesson: regrasp below the equator much earlier (as soon as the end clears ~5 deg); recompute contact frame from current world geometry, not the stored object frame; replan after descending.

## iter 3: 0/3
Changed: initial pinch 20 deg / 4 mm squeeze, regrasp to -10 deg at 5 deg pen elevation, retry re-aims after descent.
Result: all three identical: grasp reads thumb ~3 N, index ~0 N (index shell hits the table when its pad is placed below ~7 mm height); lift to 5 deg OK with balanced 1.1-1.5 N; regrasp to -10 deg -> index force 0, pen lost. Retries push the pen away (thumb overpowers), never recover.
Lesson: the index comes from above on the far side and cannot get onto or below the equator (its tip/links hit table or pen top); keep the index high (>=28 deg) and put the thumb low instead -> asymmetric, near-antipodal contact pair (thumb 15/index 45 at grasp, thumb -12/index 28 after the end is up).

## iter 4: 0/3 (budget cut to 4: last iteration)
Changed: asymmetric contact pair (thumb 15 deg / index 45 deg at grasp, planned regrasp to thumb -12 / index 28 at 6 deg).
Result: worse. Grasp 5.4/2.1 N, but the pinch already fell to ~0.86 N at 2.5 deg and the pen was lost by 5 deg, before the regrasp. Raising the index makes the downward wedge worse.
Overall lesson: with weak (kp 3) fingers and 16-20 mm fingertip shells, a top-down thumb-index pinch on a 7.5 mm pen lying on the table cannot get its contacts onto or below the equator. Pinch force always decays as the end rises, and the pen slips out between 5 and 15 deg. The arc-about-the-far-end pivot controller (closed loop on pen elevation) tracked well while the grip held. What is missing is a first lift that does not depend on that pinch, e.g. a palm-sideways (rolled-wrist) grasp so the pads come in horizontally, or tilting the pen with another finger first.
FINAL = 2: the best progress (lifted ~14 deg in all 3 cases, no arm-IK crash). Every iteration scored 0/3.
