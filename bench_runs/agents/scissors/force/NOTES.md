# scissors at level force


## Plan (before iter 1)
- Grasp: external thumb/index pinch of loop A along its length (thumb on back face, index on front face) with the force servo (`grip`, 5 N, object-frame directions +-x). Long contact baseline (45 mm) resists the blades drooping; a cross-loop pinch would be a free hinge.
- Actuation: a third finger (middle or ring, chosen at runtime by `api.reach` IK residuals over a 27-point grasp-centre grid) inside loop B, position-controlled with hinge-angle feedback (target = loop-B wall point at measured angle +-12 deg lead), so holding and opening are decoupled.
- Skill `plan()/B_pt()`: map scissors-frame points to palm-frame pad targets; `drive(open)`: closed-loop hinge servo through the B finger.
- Self-built: grasp-centre search from api.reach (no offline kinematics); size estimated from pivot height (4.7 mm * scale).

## iter 1: 0/3 (nothing lifted in any case)
- Pinch servo reached ~5 N on loop A, but on lift both forces fell to 0 and servo travel saturated (20 mm): pads slid up over the wall tops. Pad targets were ~12 mm above the table (wall is 11 mm tall) and the thumb was blocked ~5 mm higher still (tip on table, pad centre far above the tip end) -> contact on the top edge with a downward-tilted normal.
- Size estimate wrong: start pivot height is 6.8 mm*scale (the scissors drop ~2 mm to 4.7 mm in the first moments), so I assumed 4.7 and got scale 1.12 always -> grasp off-centre by ~7 mm. IK search picked C=(80,40,130) mm, B finger = middle (residual 0.2 mm).
- Learned: the fingertip pad centre sits well above the tip end; the low 11 mm wall needs vertical distal links and the tips touching the table.

## iter 2: 0/3, but the scissors came up in 2 of 3 cases
- Changes: fixed the size estimate (6.8 mm basis, now within 1%), pads commanded low with vertical distal links (`directions`), touch-down calibration (read measured pads vs targets to find where the table blocks them), and a retry (2 mm lower, +1.5 N).
- Try 0 (5 N, pads ~9-13 mm up) never lifted. Try 1 (6.5 N, 2 mm lower) lifted 35 mm in 2/3 cases (100 mm clearance in the nominal case). It then dropped: nominal case on the first B-finger push (hinge got only to 2 deg); sev-1 case during the full lift (18 deg droop).
- Learned: the pinch only holds when the pads are as low as possible with >=6 N. The B finger already pressed on B with 1-3 N before the grip: it pressed loop B down or sideways, which twists A about the pinch line, and torsion is the pinch's weak axis. My position-lead drive (12 deg lead) pushes much too hard.

## iter 3: 0/3. The pinch now holds on the first try in all 3 cases (7 N, pads preloaded into the table): 51-54 mm clearance, ~22 deg droop
- Every case dropped the scissors at the start of the first B-finger drive, before any opening (B servo force stayed 0 with travel saturated at 45 mm). Cause, from the timing on film: the plant move re-aimed the B finger's distal link along the drooped scissors normal (22 deg tilt). That swept the fingertip into the loop-B walls and twisted A out of the compliant pinch.
- Learned: the pinch is compliant vertically (kp 3), so the scissors droop about 20 deg. Anything that loads loop B other than in-plane pushes releases the grip.

## iter 4: 0/3 (budget used)
- Change: the B finger keeps its original vertical distal direction when it moves to the loop-B wall, and the plant move is slower (0.4 s). It did not help. The nominal case lifted (56 mm clearance) and again dropped at the start of the first drive. In the sev-1 case the scissors slid out during the full lift. The sev-0.5 case missed on the first try, took the retry path and hit the 60 s sim limit. Its finally-block notes were lost because the bench never closes the generator.
- Conclusion: with 7 N on two compliant pads, the loop-A pinch is just strong enough to lift, but it sits at the edge of slipping (droop ~22 deg, servo travel 20-27 mm of its 30 mm). Any extra load from the loop-B finger, or the slow creep, ends the grasp. A better grasp would add a third pinch contact (e.g. the middle finger also on loop A's front face, with the ring finger in loop B), or hook a finger through each loop. I ran out of iterations before trying either.
- Force control: used for the loop-A pinch in every iteration and for the loop-B drive in iters 3-4. It mattered: the pinch only held at >=6.5 N with the pads at the bottom of the wall. The servo found the contacts despite size errors and the IK offsets (10-27 mm of travel), which position control would not have done. It could not fix the grasp geometry (two-point pinch, weak in torsion, compliant vertically).
- FINAL = 3: the same 0/3 as the others, but it lifted the scissors on the first try in all 3 cases (51-58 mm clearance, 2-5 s in the air). Iter 4 was no better and lost one case to the time limit.
