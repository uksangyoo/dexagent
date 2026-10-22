# pen_gravity_pivot at level synergies


## Plan (before iter 1)
Strategy: thumb-index pinch near one end (pinch line horizontal, perpendicular to the pen) + middle finger as a 3rd contact
to keep the pen level (A: middle backs the index on the long part, friction; E: middle presses on top of the short part).
After the level hold: roll the hand about the pinch line with the pen clamped until the pen is ~vertical, release the middle
(pen pen-contacts have no torsional friction -> the pinch is a free hinge), roll the hand back ~85 deg: gravity keeps the
pen vertical while it turns ~90 deg in the fingers. Avoids an undamped free pendulum swing (nothing would damp it).

Skills written (policy.py):
- synergy calibration in-policy: tips at mean and at +1 sd per synergy (measured, in the air) -> linear tips(s) model.
- `solve(spec, ...)`: weighted least-squares (s + free pinch centre c) for palm-frame pad targets, refined by 3 measured
  Gauss-Newton steps, with an active-set keep-out for ring/little. This is my own "fingertip IK" built on the synergy level.
- palm orientation from measured pads: maps (pinch line, pen direction) in the palm frame onto the world grasp frame.
- `rotm` roll about the measured pinch line through the measured pinch midpoint, in small segments (chord error small).

## iter 1: 0/3
Policy calibrated the synergy->tip map in the air (mean + 1 sd per synergy), then Gauss-Newton for a thumb-index z-pinch +
middle. Bug: a ring/little keep-out (little pad x <= 55 mm) was infeasible and drove all 6 coefficients to +-2.5 sd, so the
"pinch" had thumb->index along palm x/y and the derived palm rotation pointed the fingers DOWN; the hand knocked the pen off.
Learned: at s=0 the pads sit ~85 mm out of the palm; s0 curls index/middle/ring (index -42 mm z), s3 swings the thumb
-30 mm in y toward the index, s2 the opposite, s5 moves the middle alone (-22 mm z). The linear model offline says
s~[1,.1,-.1,.9,0,-.7] gives an almost exact thumb-index z-pinch (gap 17 mm) with the middle 10 mm above the pen 20 mm
along it. Also: notes must be emitted in try/finally (an exception lost them in one case). Nominal rest is +12 mm toward +y.

## iter 2: 0/3
Used the iter-1 tip model (no in-air calibration). Thumb-index pinch WORKED: pen lifted 118-122 mm, pinch forces ~3 N.
But the middle never touched (it sat ~2 mm above the pen: the GN-refined pinch line came out tilted 38 deg in the palm
x-z plane so "above the pen" was off), so the pen hung on the free hinge and swung +-20-40 deg (no damping, as expected),
then dropped during the roll. Fix: after closing on the pen, read the pen pose in the palm frame and place the middle on
the short part's top (20 mm from the pinch, 1.5 mm press) with a local linear solve around the measured posture.
Skill: `local(targets, s_ref, T_ref)` = linear synergy solve around a measured posture (keeps thumb/index, moves middle).
(Process note: iter 2 was killed by my own mistaken pkill before its severity-0.5 case finished, so it was never recorded
in progress.jsonl; its two finished cases (sev 0, sev 1, both fail) are what the notes above are based on. A launch for
iter 3 was also killed while queued. The next recorded iteration directory is iter_4.)

## iter 4 (2nd recorded): fails (closed posture diverged)
Added the measured-pen middle press, but also raised the in-air Gauss-Newton rounds to 4: with the Jacobian measured at
the mean, the refinement DIVERGED (s -> [-0.4,-2.2,1.9,2.5,1.2,-2.5]), the "closed" thumb sat 16 mm off, the palm
orientation came out wrong and the grasp never happened. Lesson: the 6-D synergy map is too nonlinear for fixed-J GN
far from the mean; only small damped local steps are safe.
Change for iter 5: fixed closed posture s_c = iter-2's refined one (it pinched the pen with ~3 N), pads measured in the air;
open = damped local step; middle press weighted only along the pen-normal (n_up) and pinch (u) directions; grasp 32 mm from
centre (keeps the middle on the short part for -10 % pens).

## iter 5 (3rd recorded): 0/3, but level hold reached in 2/3 (0.48 s, 0.92 s)
Fixed closed posture + measured-pen middle press works: pen lifted 163 mm with middle force ~0.12 N, held_level true in
sev 0.5 and 1, max palm-frame pivot 88 deg (but only because the pen then fell out). Failures: (a) the long part rose to
~24 deg above horizontal during the hold (ap in palm gained -x), (b) the pinch force decayed steadily 2.3 N -> 0.5 N
(sev 1: -> 0.06 N) while the posture was constant (pen working its way out of the convex pads), and the pen dropped as
soon as the clamped roll started.
Changes for iter 6: pinch 2 mm tighter per pad (local step, gap ~13 mm); closed-loop levelling after the lift (middle moved
along n_up by elevation error x 24 mm lever, up to 5 steps); slower roll (0.35 s per 8 deg); thumb/index targets anchored to
the measured closed pads in every local solve (no drift accumulation).

## Resume (2026-10-22): iter_6 was killed before any case finished (not counted); its policy was never scored.
Re-read iter 5: pinch force decays roughly exponentially (2.3 -> 0.5 N in ~3 s) while the pen is horizontal, and the pen
drops early in the roll; in sev 1 the pen centre fell ~85 mm during the roll while its palm-frame axis stayed constant ->
most likely the pen creeping/sliding out of a pinch whose force had already decayed (MuJoCo soft friction creeps under the
gravity + middle-press load, which in the horizontal phase acts along the squirt direction of the convex pads).
Changes for iter 7 (on top of the untested iter-6 ones: 2 mm tighter pinch, closed-loop levelling, anchored local solves):
- all control after the press is per 20 ms step with a persistent finger command st["s"] (no fixed-target holds);
- `regrip()` skill: pinch-force servo; if mean thumb/index force < 1.6 N, close both pads along the measured pinch line by
  up to 1.2 mm per 0.2 s (cap 4 mm total), through the local linear synergy solve (middle kept);
- `arc()` skill: roll the palm about the measured pinch line exactly on the circle at every step (smoothstep profile,
  88 deg in 2.0 s, back 78 deg in 2.2 s), logs commanded-vs-measured palm rotation error (IK only checks position);
- less time horizontal: lift 1.0 s, levelling steps 0.26 s, then roll as soon as my own stricter level check
  (lift > 75 mm, < 12 deg) has run 0.56 s (cap 1.6 s);
- roll-back 78 deg (was 74) for margin on the 60 deg palm-frame pivot rule.
Offline: only a numpy mock api (no physics) to check the generator runs end to end; no kinematics precompute.

## iter 7 (4th recorded, last): 0/3, but the level hold passed in 3/3 (1.26, 1.32, 1.18 s; lift 146-169 mm)
- The middle press, not soft-contact creep, is what kills the pinch. In sev 0 and 0.5 the press tilted the long part up
  (tilt from horizontal 24 -> 37 deg) and the thumb/index force fell from 3.0 to 0.2-0.7 N during the 1 s lift. Levelling
  (the middle backed off 4 x 5 mm) brought the pen level, but the force never recovered: regrip added 4 mm and the force
  stayed at ~0.15 N, so the pen had been levered sideways out of the pads (squirt) rather than the pads loosening. In sev 1
  the press missed (middle force 0) and the pinch kept 2.6 N right through the lift with the pen already level.
- The pen drops early in the clamped roll in every case. In sev 1 it fell at ~45 deg of roll with the force at 1.5 N and
  the palm tracking well (1.5 deg error). In sev 0 and 0.5 the logged palm orientation error reached 77-110 deg: the arm
  IK checks only the position residual, so it gives up orientation on this roll about the pinch line (axis 85 mm off the
  palm), and the "rotation about the pinch" turns into a translation that drags the pen.
- Skills: `arc()` (per-step exact rotation about a measured line, logs the IK orientation error) and `regrip()`
  (force-triggered pinch tightening via local synergy solves). regrip cannot recover a pen that has squirted out.
Next steps had there been budget: no middle press (pinch only; sev 1 shows it lifts level), shorter G_OFF so gravity
torque is small, a roll direction or palm pose with an IK-feasible orientation (check rot_err and abort or replan), or
the plain release-and-catch: open the hinge under gravity and put a stop finger at vertical.
FINAL = 7 (the only iteration with a level hold in 3/3; all iterations scored 0/3).
