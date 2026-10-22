# pen_pickup_pivot at level synergies


Skills (policy.py):
- calibrate(): online synergy->pad map: measure tips at s=0 and +-1 sd per synergy in the air (central-difference Jacobian), then solve()+refine (Broyden-updated, re-anchored) for an open (42 mm) and closed (20 mm) thumb-index pinch with middle/ring/little tucked >=20 mm above the pinch. This is my own IK over synergy space built from api.tips() (rule-1 note).
- move_until(): iterate an api.move generator and stop early on a condition (touch search, tilt monitor).
- level_pen(): rotate the hand about the pinch point so a hanging pen's axis goes vertical (levels the free pinch hinge).

## iter 1: 0/3 (all cases identical failure: pen never touched)
- Online calibration around s=0 (+-1 sd) worked as data, but the linear tip model is poor beyond ~1 sd (pad depth palm-x saturates: predicted 115 mm, measured 86 mm), so refinement jumped around and ended with a bad open posture (little finger extended).
- Touch search fired with thumb/index pad centres 17/12 mm above the table (fingertips point down; the tip end is ~12 mm below the pad centre). Closing raised the pads (~5 mm), so the pinch closed above the 7.5 mm pen: zero contact force, nothing lifted.
- Sim is deterministic: calibration numbers identical in all 3 cases -> offline design from logged data is valid.

## iter 2: 0/3
- Local calibration around a hand-picked pinch guess (h=0.5 sd) + Broyden refine gave a clean 18.6 mm thumb-index pinch, but its hinge runs mostly along palm y and the little finger sat at pinch depth almost on the hinge line, so the little finger hit the table first: touch search fired with thumb/index pads 20-40 mm up; grip closed in air (sev 0.5, 1). Sev 0: little finger knocked the pen and the xy-correction (re-reading the pen pose) chased it; pen fell off the table.
- Linear tip model from +-0.5 sd differences is good to ~10 mm over ~1 sd.
- Fix for iter 3: cost requires every other pad >=16 mm shallower than the pinch; anchor at offline fit of iter-2 data; palm roll about the hinge chosen to maximise world clearance of other fingers; E1 side away from other fingers; touch detection only on thumb/index; grasp target fixed from the initial pen pose.
- (iter_3 dir is a launch I killed while still queued for the GPU -- not counted, no results; next run is iter_4.) Before relaunching I also made the hinge sign/palm roll choice prefer fingers pointing away from the robot base (in iter 2 sev 0 the chosen orientation brought the forearm down near the table and the pinch ended 100 mm off target), with an IK-failure fallback to the other sign.

## iter 4 (3rd budgeted): 0/3
- Measured pads at the anchor S_C0 differed ~10 mm from the iter-2 linear prediction and the local Jacobian there is small (joints at limits), so the online refine diverged (closed-posture cost 49 -> 116) and returned its LAST, not its best, measurement: a 32 mm "closed" pinch with the thumb 28 mm deeper than the index (palm tilted 48 deg). Little finger and thumb jammed the pen (8 N each in sev 0); the pen was knocked off the table in all cases (stood_on_end=true only because it tumbled).
- Lesson: on this hand the synergy->pad map is strongly nonlinear (clipping); online Gauss-Newton in synergy space is not reliable with 6 coefficients. Re-using measured postures is safer.
- Final iter: no online optimisation. Closed = S_C0 as measured (16.6 mm, hinge mostly along palm z, other fingers clear after roll by 44 mm); open = S_C0 + 1.2*(S_M2 - S_C0) (index swings out from an almost fixed thumb); pen placed 14 mm from the thumb pad; touch search also stops if a non-pinch finger deflects >4 mm.

## iter_5 (killed mid-run after the study pause; not counted). Partial results of the iter-4-revised policy: sev 0 and 0.5 both 0 pivot.
- Open posture S_C0+1.2(S_M2-S_C0) had a pinch line ~60 deg off the closed one (open v=[-.12,.19,.97] vs closed [-.66,.73,.15]): closing swept the index around instead of closing on the pen; thumb rose 14 mm; zero pen force.

## iter 6 (4th and last budgeted): rewrite of grasp
- Skill `Model`/`calibrate`: a fixed 1-D synergy path s(lam)=S_O+lam(S_C-S_O) between two postures logged earlier that share the same thumb-index line (S_O=[2.37,-.38,-.98,.8,1.42,.01] 38 mm, S_C=S_C0 19 mm; index nearly fixed, thumb sweeps toward it), pads measured in the air at lam=-.35,0,.5,1,1.25. No online optimisation (iter 1-4 lesson). Wider lam=-.35 opening used only if measured line/clearance is OK.
- Skill `candidates`: palm rotation that maps the measured hinge to the horizontal perpendicular of the pen and the other fingers' side straight up, rolled phi about the hinge; scored to keep fingers >=25 deg pointing down (iter 4 sev .5/1: with horizontal fingers the arm could not descend - touch search never hit), fingers away from the base, other fingers leaning to the long-end side so the short end can swing up. Falls back on IK failure, arm-tracking error, or a touch search that never hits.
- Grasp: touch down with the open pinch around the pen (pen axis 2 mm past the open midpoint toward the index), close along the path while raising the palm by the measured path's pad-height change (tips keep sliding on the table), test lift 10 mm, retry with more press and a different pen offset. Then level hinge, slow lift (pen should swing upright on its far end), lift clear, level_pen.
- Result: 0/3, and all three cases failed the same way (no pen contact, pivot 0). The calibration matched the plan: the open lam=-.35 pinch was 44.6 mm, its line within 20 deg of the closed one, and the other fingers sat 37 mm above the pinch. The arm reached the fingers-down pose (track err 4 mm). But the touch search fired at once with the thumb/index pads still 31/35 mm above the table. It was probably a false trigger from the fingers still settling after the xy correction (a deflection threshold of 1.5 mm with only a 0.25 s hold). The close then ran in the air. The palm compensation dropped the thumb to 16-20 mm while the index stayed at 32 mm, which tilted the hinge 25-40 deg. All 5 retries repeated this exactly, because the sim is deterministic. lam=1.25 gave the same aperture as lam=1 (21.5 mm), so the path saturates past S_C.
- Lesson: the touch search needs a longer settle and a re-zeroed reference, or a geometric stop (pad z near the table) as a sanity floor. A trigger that fires before any descent should be ignored. A retry has to change the cause, not just the press.
- Budget used (4/4). FINAL = 6: all iterations scored 0/3. Iter 2/4's 'stood_on_end' was the pen tumbling off the table, not real progress. Iter 6 is the cleanest design and needs only a touch-trigger fix.
