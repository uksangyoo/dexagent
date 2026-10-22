# pen_pickup_pivot at level joints


## Skills (policy.py)
- `fd_jacobian(api, q, fingers)`: measured finite-difference Jacobian (pad centre vs 4 joints) by physically perturbing joints in free air and reading api.tips(). Built because the joints level has no fingertip IK.
- `servo_tips(api, q, J, targets)`: closed-loop pad placement in free air (damped least squares + Broyden updates on measured tips). My own fingertip IK.
- `level_hinge(api)`: rotate the palm about the pinch point so the hanging pen points straight down (component about the hinge line removed).

## Iter 1: 0/3
Plan: thumb-index pinch across the pen near one end (hinge along finger dir), slow straight lift so far end drags and pen swings vertical, level, touch down, lift. Servo IK converged (fingers moved), but the pen never moved (pivot 0.3 deg, 0 contact force): the pinch closed on air. api_log kept only the last 60 lines (flooded by per-step move calls), so my notes were lost; level_hinge then wildly rotated the palm because tilt was 90.
Learned: print() from policy lands in iter_k/log_*.txt -> use for diagnostics. Guard level_hinge (tilt<40).

## Iter 2: 0/3
Added print() diagnostics. Servo IK converges to <0.5 mm in free air. Descent stopped on table contact with pad centres still 11-15 mm above the table (fingertip ends protrude ~11 mm below the pad centre). Pinch made contact (0.4 N each) but pad centres were 8.5 mm above the pen axis, 9.5 mm to each side -> contact normal ~42 deg from horizontal = friction-cone limit (mu 0.9): the pen squeezed out downward as soon as the hand lifted. Thick tips vs 7.5 mm pen is the crux.

## Iter 3: 0/3
Pinch now presses the tips onto the table and aims pads 4 mm lower than reachable, tighter squeeze (1-2.4 N); retries up to 3 times. At sev 0/0.5 the pen was held after a 15 mm test lift (forces 0.6 N, grip end up ~4 deg) but my acceptance test (tilt<85) wrongly rejected it and released; at sev 1 it slipped out. Squeeze decays during the lift (2.4 -> 0.65 N): the pen wedges down between the round pads (pad centres ~6 mm above pen axis, normal ~32 deg). Retries were spoilt because the fingers opened while moving away and kicked the pen.
Learned: geometry, not force, limits this pinch; contact is held only while the squeeze is kept up.

## Iter 4 (final budget) change
Lift is now the grasp test, with adaptive squeeze: every 5 steps, if a pad force drops below 1.5 N the closing target is extrapolated further (pads closer, aimed lower). Grasp lost -> open in place, back off, re-read pen, retry (time-guarded).

## Iter 4: 0/3 (budget exhausted)
Adaptive squeeze + aim 5 mm low: worse. Initial pinch force only 0.4-0.85 N (index often 0) because aiming lower moved the index pad away along the pinch axis, and the pen was lost within ~2 s of lifting in every case; retries never got both pads on. Iteration 3's posture (aim 4 mm low, sep 6 mm) gave the best grip (2.4 N, held the pen through a 15 mm test lift at sev 0/0.5), so FINAL = 3.
Takeaway: at the joints level all of my effort went into building fingertip IK (FD Jacobian + Broyden servo) and probing the tip geometry blind. The real problem, a thin pen between thick round pads near the table, needs pad orientation control (tip ends curled toward each other) that I could not express without a tip-orientation model.
