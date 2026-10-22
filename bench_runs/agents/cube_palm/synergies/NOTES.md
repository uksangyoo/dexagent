# cube_palm at level synergies


## iter 1 — 1/3 (sev1 PASS, sev0 fail, sev0.5 fail)
Changed: first policy. Palm-down wrap grasp with the cube just under the palm, lift, roll 180 deg about palm z keeping the cube centre fixed, open to a "cage" posture so the palm carries it, re-grasp, roll back, set down, release. Synergy postures found per-run by an in-policy LM fit of the 6 coefficients to fingertip targets, using measured api.tips() as forward model (FD Jacobian + Broyden) — a self-built synergy->fingertip IK (allowed: built from api only).
Result: fit was poor (rms ~17 mm): the thumb cannot be brought under the palm (y~0) in this synergy space (open thumb points along +y, pad at y=127 mm); the fitted postures differ run to run. sev1 worked: thumb+ring wrapped the cube against the palm (fist-like cage), 2.24 s on palm, set down 18 mm off. sev0: descending with the pre-shape, thumb/ring landed on the cube top (189 N), cube pushed into the table, never lifted. sev0.5: cube held further out (z=100 mm), cage still touching, opened fully -> cube fell off.
Learned: describe(): s0 = power curl (ring/middle curl strongly), s1 = thumb across (-y), s2 = thumb +y/-z. Natural thumb-index pinch sits at ~(75,45,100) mm palm frame, off the palm. Fist-like wrap with thumb+ring holds the cube on the palm well.
Skill: fit(s0, targets) — in-policy LM over synergy coefficients with measured tips.

## iter 2 — 0/3
Changed: dropped the per-run fit; fixed synergy postures S_GRASP / S_CAGE taken from iter-1's passing case; descend with an OPEN hand (iter 1 sev0 failed by landing the pre-shape on the cube), then close cage -> grasp.
Result: all 3 lifted the cube (~178 mm) but contact forces were 0 at grasp: the fist postures press the table around the cube and, as the hand rises, ring+middle scoop it from below, so it sits in a finger pocket ~25 mm above the palm (palm frame x~73 mm, z~103 mm), pinched by thumb+middle. Palm-up, opening the cage let the cube roll off sideways (0 s on palm).
Learned: with synergies the fist "pocket" is a scoop, not a palm-press; iter-1's success came from the cube being pushed against the palm by contact during the descent (luck). Need a controlled transfer from the pocket to the palm.

## iter 3 — killed mid-run (not counted; 1 case logged, sev1 fail)
Changed: palm sloped 38 deg toward the wrist before opening fully to the open hand in 0.8 s. The logged case lost the cube off the little-finger side (palm y=-120 mm). In iter 2 it went off the thumb side, so the cause is the fingers flicking the cube as they open quickly, not the slope direction. The grasp/lift/roll stage reproduces well: palm-up, the cube sits pinched by thumb+middle at palm (73, 10, 103) mm in all 4 runs (sev 0 to 1, edge 38.7 to 42.7 mm).

## iter 4 — 3/3 (sev0, sev0.5, sev1 all PASS)
Changed: kept grasp/lift/roll. New transfer: tilt the palm-up hand 0.6 rad with the fingers raised (api.hand_up(yaw, 0.6), rotated about the cube centre) so gravity points down the palm toward the wrist. Then relax SLOWLY, in steps of alpha += 0.02 every 0.14 s along S_GRASP -> open_coeffs, and run an in-policy copy of the scorer's on-palm test after each step. The posture freezes as soon as the test passes and is held for 80 passing steps, then the hand re-closes to S_GRASP, untilts, rolls back, sets down and releases.
Result: the pinch let go at alpha 0.20-0.28. The cube dropped onto the palm at (44, 7..16, 79..89) mm and the palm carried it for 1.84-2.1 s with ~0 N finger force. The regrasp wrapped thumb (+ring) against the palm (10-20 N), and the set-down landed 20-48 mm off.
Learned: the transfer only needs the pinch to open a little. The fingers stay as a loose cage, and since the cube weighs just 0.12 N, resting contacts are far below the 0.5 N finger limit. Feedback freezing plus slow relaxation removed the flick, and tilting toward the wrist makes the cube land on the palm instead of the knuckles.
Skill: on_palm(api, z0) is an in-policy copy of the scorer's carry test. The relax loop is a feedback "open until the condition holds, then freeze" release.

## iter 5 — 3/3 (policy unchanged from iter 4; confirmation run, stop rule met)
Result: all three passed. Time on the open palm was 2.08-2.36 s and the set-down landed 15-27 mm off. In the nominal case the pinch let go only at alpha 0.58, against 0.24 in the others, so the feedback loop (open until the carry test passes) is doing real work: a fixed opening amount would not have suited every case.
Learned: the policy is robust to the full variation range because the grasp converges to the same pocket pose and the release is closed-loop. FINAL = 5.
