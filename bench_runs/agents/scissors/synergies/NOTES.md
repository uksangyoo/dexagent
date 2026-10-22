# scissors at level synergies


Skills written (policy.py):
- `settle(api, s)`: command synergy coefficients, wait 0.4 s, return measured pads (palm frame).
- `Model` + FD Jacobian: measure d(pads)/d(s) at run time in the air (k probes, 1 sd each); damped LS
  solve of s for palm-frame pad targets, with one-sided "keep other fingers above the pads" rows.
  This is a home-made fingertip IK on top of the synergy level (built only from api.move/api.tips).
- `refine(...)`: measured-feedback correction of the LS solution (3-4 rounds).
- `clearance(api)`: same lowest-point estimate as check.py, used to gate the in-air cycles.

## iter 1: 0/3 (no lift in any case)
Plan was a thumb/index pinch with the palm yawed 90 deg (pinch axis across the loops), synergy pad IK via a
run-time FD Jacobian. Learned: k=6; the thumb pad barely moves along the fingers (palm z) under any synergy,
it moves sideways (palm y; s3 ~ -26 mm/sd, almost pure thumb); the index mostly moves along z. So the
thumb-index pinch along palm z was unreachable (pad errors 20-40 mm), the thumb hit the scissors (166 N), nothing lifted.
Also: pivot frame rests 6.8 mm (x size) above the table, not 4.7 -> usable size estimate.

## iter 2: killed while queued (study paused); policy never scored. Not counted.
On resume I dropped the iter-2 design (thumb in B + ring in A, squeeze inner walls to hold, spread outer
walls to open): switching from squeeze to spread releases the only grip, so the scissors would drop mid-cycle.

## iter 3 plan (rewrite)
Default palm-down yaw = scissors yaw (thumb over loop A). Ring pad into hole B by B's outer wall, little pad
outside that wall; little moves toward ring -> pinch on B's outer wall = grip independent of the hinge.
Thumb pad in hole A; cycles are closed-loop steps of s3 (thumb sideways, ~-26 mm/sd) on the measured hinge.
Size estimated from the settled pivot height (4.7 mm nominal; the t=0 height 6.8 mm is pre-settling).
Skills (revised): `solve()` generic damped-LS synergy pad IK with equality rows (pad depth equal, sideways
separations) + active-set inequality rows (middle/index shallower); `refine()` measured-feedback loop;
placement correction of the arm using measured tips_world before the guarded descent.

## iter 3: 0/3 (no lift in any case)
Refine failed: the 12 mm "middle above ring" guard drove s to +-3.2 where the s=0 Jacobian is badly wrong;
measured thumb pad ended 43 mm shallower than ring (err ~40 mm, never converged). Ring pad, aimed 2.5 mm off
B's outer wall / 1 mm off the hole end, landed on the wall top (25-35 N, pad 17-22 mm up) -> no pinch, no lift.
Arm placement error 4-7 mm at severity>0; one measured-tip correction fixed it on the retry (0.1 mm).
Learned: effective pad radius ~10 mm (not 8.5); linear synergy model only trustworthy within ~1 sd.

## iter 4 change
Pads aimed at hole centres (y=+-sep/2, x=-65 sc), PAD_R 10; guards softened (middle only <= ring depth,
ignored by the descent guard since it can rest compliantly on the shank); LS damping 6 + trust region 0.8 sd,
Broyden update of J from each measured refine step (skill `refine` revised). Solutions now |s| <= ~1.4.

## iter 4: 0/3 (no lift)
Pad IK now converged (err 1.5-2.4 mm, |s| <= 0.8) and arm placement was exact (0.0-0.2 mm) in all cases.
But the descent was blocked ~10 mm short in every case: thumb contact 57-65 N (stiff, i.e. not the pad
against a compliant joint) plus middle 12-17 N on the shank/pivot; ring/little pads stopped at 17-19 mm,
above the 11 mm walls, so the ring-little pinch closed on air. Middle is at ring depth but ~70 mm further
along the fingers -> it lands on the pivot region.

## iter 5 change (last budgeted)
Palm tilted 9 deg fingers-up about palm y (middle/index pads ride ~10 mm higher than thumb/ring/little).
Adaptive descent (skill): if blocked short of depth, back up, then shift the blocking pad (thumb via s3 +-4/7 mm,
otherwise the palm along scissors y) and retry, up to 3 tries; second grasp attempt only if t < 24 s.

## iter 5: 0/3 (budget used up)
The 9 deg tilt made things worse: the thumb now blocked earlier (15-40 N at f=0.75, pad 31 mm up), the tilted
yaw ~0.9 rad poses gave arm placement errors of 10-50 mm that the measured-tip correction amplified instead of
fixing, and one case hit arm IK failure (residual 4 mm) on the lift. Severity 0.5 shows the only lift so far
(pivot +5.9 mm, clearance 2.3 mm) from a pinch that landed on the wrong walls. Thumb shifts of +-4 mm by s3 never
cleared the thumb block, so it is not the pad's y position: some other part of the adducted thumb meets loop A.
FINAL = 4 (no errors, accurate pad IK and placement, scissors left flat and closed; iter 5 crashed one case).

Main lesson for this level: the synergy span couples the fingers I needed independently. Ring abduction is not
in it at all, and lifting middle/index above the gripping pads needs |s| ~ 3, where the s=0 linear model breaks.
I had to build a measured-Jacobian pad IK (damped LS + active-set guards + trust region + Broyden) before any
grasp could be posed. Even then, the thumb posture the synergies give at the required adduction does not fit
into a 27 mm loop.
