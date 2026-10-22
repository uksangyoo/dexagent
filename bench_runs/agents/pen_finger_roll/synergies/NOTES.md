# pen_finger_roll at level synergies


## Iter 1: 0/3 (all: grasp + lift OK, pen dropped during in-hand swing)
- Plan: thumb-index pinch along palm z (pen along palm x), swing = differential pad shear along palm x, roll = shear along y; closed loop on the scorer's own tilt/twist formulas.
- Skill written: in-policy "synergy IK": measure api.tips() at s=0 and s=0.6e_j (free air, k=6 -> ~2 s), Gauss-Newton + Broyden on measured pads for a target rel = index-thumb pad vector and pinch centre, then a local finite-difference Jacobian; V = s-directions for unit mm changes of rel x/y/z.
- Learned: k=6. GN reaches rel within 0.5 mm in 5 measured steps; pinch grasp + 130 mm lift worked 3/3 (forces ~0.5-1 N). But x-shear gave only ~0.6 deg/mm of swing, loosened the grip (model error of the free-air Jacobian in contact) and the pen dropped within 1 s. Twist reached 28-40 deg as a side effect.
- Insight: for a two-pad pinch, the easy (hinge) rotation is about the pinch axis; swing is about palm y, so the pinch axis should be along palm y.

## (killed run, not counted) lateral thumb-index pinch (pinch axis along palm y) + table-pivot swing
- The one finished case (sev 1) never touched the pen: the lateral pinch formed from the free-air synergy model put the pads past each other (contact at rel_y=-9 mm with 0 N). Lateral pinch is hard to form with k=6 synergies; abandoned.

## Iter 3 folder = 2nd completed iteration: back to iter-1 z-pinch grasp (3/3 lifts), roll first, swing by table pivot
- Roll: rel-y differential shear (index abduction vs thumb y) closed loop on twist, in the air, with grip keeper (tighten rel z when a pad force < 0.3 N).
- Swing: move to free table 12 cm from the cup, lower until the pen tip touches, move palm along palm z (palm +z -> tilt goes negative, derived from geometry) closed loop on in-palm tilt while palm height follows d*cos(tilt); lift to rim+22 mm to read tilt in air (only above-rim tilt is scored); spring-back estimate compensates the table target. Goals +18, -max(18,33-hi), 0, up to 3 tries each.
- Skill: `pivot()` table-pivot tilt servo; `cycle(goal)` = lower/pivot/lift with spring-back learning.
- Result (folder iter_3): 0/3. Grasp+lift 3/3, ROLL WORKS 3/3 (range 36-37 deg; rel-y shear gain ~10-12 deg/mm, so in-air pad shear rolls the pen easily). Swing failed: on the table the pen buckled flat (tilt -> 90 deg) and dropped. Cause: palm height followed the *measured* tilt (positive feedback: vertical tip load x lever d*sin(w) adds to the tilt torque). Grip after roll was also low (0.23 N).
- Learned: pinch torque capacity is tiny (~2*mu*N*r ~ 0.002-0.004 N m), so any vertical push on a tilted pen buckles it.

## Iter 4 folder (3rd completed): arc-constrained pivot
- Change: pinch centre driven along the circle of radius d about the pen's bottom (rigid-geometry arc, commanded angle phi ramps at ~13 deg/s, slow integral on measured tilt), so over-rotation in the pinch lifts the tip instead of buckling; after roll, shear returned to 0 and squeeze re-tightened to >= 0.5 N per pad.
- Skill revised: `pivot(goal)` = arc pivot about the tip contact; `geom()` = pen bottom / grip point / pinch offset in world.
- Result (folder iter_4): 0/3. Roll again 3/3 (36-43 deg). Arc pivot did not help: the pen (pre-tilted ~8-10 deg by the roll/lift) popped out of the pinch right at table contact and fell flat.
- Learned (key): in this z-pinch the curled index and thumb distal links run parallel to the pen, i.e. line contacts along the pen. Rolling across the finger's round cross-section is nearly free (roll works, ~10 deg/mm), but tilting about palm y fights the line contact, so the pen either follows the pads or pops out; it never slides into a new tilt. The swing has to come from pitching the pads themselves.

## Iter 5 folder (4th = last completed): in-air swing by an in-contact synergy Jacobian
- Change: after roll, probe each synergy by +-0.3 sd while holding (measure d tilt/d s_j), take the least-norm s-direction with d tilt = 1 deg and commanded free-air rel z (squeeze) unchanged, servo along it to +17, then -max(16, 33-hi), then 0 and hold; squeeze bumped if a pad force < 0.25 N. Table pivot removed.
- Skill: `swing_to(goal)` = 1-D closed-loop servo along a measured object-space synergy direction (in-contact Jacobian built from the scored observable itself).
- Result (folder iter_5): 0/3. The very first in-contact probes (+0.3 sd on s[0] / s[1]) dropped the pen in all 3 cases (s[0]: 8-13 deg/sd of tilt, yet the pen did not come back and fell on the next probe). At 0.35-0.5 N grip a 0.3 sd synergy step moves too many joints at once to keep a 7 mm pen pinched. Budget used.

## Final
- FINAL = 4 (dev 0/3; grasp, lift and roll pass 3/3 and the pen stays held until the swing). No iteration passed.
- Per-iteration dev: 1: 0/3, (2 killed), 3: 0/3, 4: 0/3, 5: 0/3.
