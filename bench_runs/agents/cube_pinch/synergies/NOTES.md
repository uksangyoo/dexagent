# cube_pinch at level synergies


## Skills (helpers in policy.py)
- `SynIK` (built by me, rule-1 note): online synergy->pad-position model. Measures api.tips() after
  +1 sd finite-difference probes of every coefficient in the air (k moves), then Gauss-Newton with
  Broyden updates to find coefficients giving a thumb-index pad vector (0,0,gap) along the palm finger axis,
  midpoint softly near (80,40,118) mm palm frame. This is my own "synergy IK" from public api only.
- note buffer: re-emits all api.note lines at the end (bench keeps only the last 60 log lines).

## Iterations
- iter 1: first full policy (SynIK calibration ~6 s in air, k=6; yaw-aligned thumb-index pinch along palm z,
  descend open, close with palm shift for midpoint drift, lift 110 mm, palm turn 42 deg about vertical through cube
  in 4 steps, measured-pose place, open, retreat). Result 3/3. Turn achieved 33-36 deg (~8 deg slip in grasp),
  final offset 1-8 mm: the full re-open to s_open sweeps fingers (other fingers move a lot under synergies) and
  drags the cube ~7 mm. SynIK converged to <1 mm on the closed pinch, ~6 mm on the wide open one.
- iter 2: turn 42->50 deg, place target -1 mm, release = 40% open + lift + full open in the air. Result 3/3;
  turn now 44-45 deg (big margin), but final offset got worse: 4.5/12.3/13.2 mm (limit 15). Placed offset is ~1 mm;
  the cube sits ~6 deg tilted about the pinch axis in the grasp, and on release it falls flat and slides ~10 mm
  (partial-open + lift drags it more than iter 1's full open). Stop rule met (3/3 twice).
- FINAL = 1: same dev score, and its worst margins (offset <=7.7 mm vs 15, turn >=33 deg vs 25) are safer than
  iter 2's 13.2 mm offset. Next fix had there been budget: iter-2 turn angle with iter-1 release, and
  press the cube flat before opening.
