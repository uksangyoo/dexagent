# pen_writing_grasp at level joints


## iter 1: 0/3
Built measurement-based finger IK in policy.py (no IK at joints level): survey 10 random thumb/index postures, then Gauss-Newton with finite-difference + Broyden Jacobian on api.tips(), targeting a thumb-index pinch along palm z (centre (80,40,118) mm, from hand.py's description). Approach pen top palm-down, close until forces, lift, carry, feedback-lowered stroke, lift+hold.
Result: pen never left the cup in any case (film: pads closed around/above pen top without holding). My api.note diagnostics were lost because the api_log keeps only the last 60 entries and per-step moves flooded it.
Learned: buffer notes and flush at the end. Skill: `measure/jacobian/solve` = measurement-based pad IK (~10 s sim).

## iter 2: 0/3
Buffered notes (flushed at the end) + closed-loop alignment of measured pads to the pen axis + grasp retry. Notes showed: index IK converges to 0.0 mm, but the thumb cannot reach (80,40,105) (stalls ~8 mm off at z~98, near its workspace edge); extrapolating the joint-space closing made the thumb slide to y=30 so the pen escaped sideways (forces 0 all retries). Bug: the "above" alignment also corrected z, so the hand descended twice and knocked the pen.
Learned: thumb reach is the binding constraint for a palm-down thumb-index pinch; use the thumb as a fixed post at a reachable point and close only the index (index IK is reliable). Seeded IK with iter-2 solutions instead of the random survey.

## iter 3 (folder iter_4; an earlier iter_3 folder was killed mid-run and does not count): 0/3
Changed: stroke along the pinch axis (palm z), slow tip-height-feedback touchdown, force-limited stroke (pen-in-palm deflection + tip height + lean guard), grasp +0.2 closing.
Result: grasp and lift now work in all 3 cases (tip ~100 mm, pad forces ~3 N), but the pen toppled to 90 deg every time: in 2 cases while hanging in the air (during carry / settle, before any table contact), in 1 case within 0.8 s of a very light touchdown (h 0.15 mm, defl 0.05 mm).
Learned: a thumb-index tip pinch 25 mm below the pen top is a near-frictionless hinge about the pinch line (torsional friction only); the hanging pen is marginally stable and any tip drag topples it. The grasp, not the stroke controller, is the bottleneck. Skill: `state()` = (tip height, pinch deflection along palm x, lean) for force-limited contact.

## iter 4 (folder iter_5): 0/3
Changed: slower lift (2.0 s) and carry (2.6 s), grasp closing back to +0.2 (cap 1.05), much lighter stroke controller (back off at 0.2 mm pinch deflection or +4 deg lean), carry diagnostics.
Result: grasp + lift fine in all 3 (tip ~100 mm, lean 1-8 deg, pad forces ~2.2 N), but the pen fell out of the pinch during the carry in all 3 cases (film: upright after lift, lying on the table by the next frame), so the stroke never started.
Learned: slowing the arm did not help; the pen leaves the two-pad tip pinch during a lateral+downward palm motion. The fix would be a different grasp (a third brace finger, e.g. middle pad on the pen 20 mm below the pinch, or a lower grasp so the pen butts against the proximal phalanges), which needs more joint-space exploration than the budget left.
FINAL = 4 (folder iter_4): same 0/3 but one case carried the pen upright to the table, the furthest any run got.
