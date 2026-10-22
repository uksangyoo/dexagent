# pen_finger_roll at level contacts


## iter 1 (0/3)
Plan: palm-down approach, thumb+index pinch along palm z at 30 mm below pen top (grasp/approach with explicit radial `normals`, since the default normal points from the pen centre and would be axial), lift 140 mm, then closed-loop swing via `rotate(pen, palm y, ...)` of thumb+index about the grasp point (abandoning the generator once my replica of the scorer's tilt crosses the target), then roll via `rotate` about the pen axis.
Result: grasp + lift fine (forces ~1.7 N each, bottom 0.889). +13 deg tilt reached quickly (tracks command ~1:1). Going back the pen followed only to about -2.5 deg, then dropped back into the cup (nominal, sev .5), or hinged over to -85 deg and fell (sev 1). Roll range 34-40 deg accumulated incidentally (lift + swing twist).
Learned: thumb-index shear swings the pen readily in +tilt (thumb toward fingertips/down, index up), the opposite direction past grasp-zero loses the grip. Closeup camera stays on the cup, so the lifted hand is out of view.
Skills: `Mon` (replica of check.py state from api observations), `drive(mon, gen, stop)` (steps any primitive, updates the monitor each step, abandons the primitive when stop() holds).

## iter 2 (no result: lost)
Same policy as iter 3 below. My shell wrapper `timeout 890` killed the run while it was still queued for GPU slots (cases started only after ~12 min of waiting), so iter_2 has no case files. Counted as a used iteration.

## iter 3 (pending at hand-back)
Changes vs iter 1: in-hand pre-tilt of +14 deg (positive side, which swings freely) while the pen bottom is only 3 mm above the rim, so the scorer's rest pose is captured mid-range. Then swing to rest-11, rest+20, back to rest. `regrip()` (re-run `grasp` at the pen's current pose) after every swing segment re-centres the pads on the pen, lift is 100 mm so the closeup camera keeps the hand in view, and swing logs samples of (tilt, thumb force, index force, pad separation).
Skills: `regrip(sq)` re-places thumb+index pads perpendicular to the pen's measured pose; `swing(target, rel)` closed-loop rigid rotate about palm y with force/drop guard.
Result iter 3: 0/3. Pre-tilt to +14 worked in all cases (0.7 s), but the grip weakened: thumb/index forces 1.4 N during the pre-tilt, 0.5 N after regrip + lift, and the pen relaxed from +16 to +6 grasp-relative during the lift, so rest was captured mid-relaxation. During the swing to rest-11 the forces decayed 0.5 -> 0 with pad separation shrinking 21 -> 17 mm: the pen squeezed out of the two-fingertip pinch (film: held only by the extreme white tips, then fell). One case flopped to +73 deg and dropped.
Learned: measured pad separation at contact is ~21 mm for the 7.5 mm pen (effective pad offset ~6.8 mm, not 8). A two-point fingertip pinch on a 7.5 mm capsule is too weak/unstable at squeeze 3 mm.

## iter 4
Changes: squeeze 6 mm for thumb/index (grasp and every regrip); the middle finger added as a light stop (squeeze 1 mm) on the pen's -y side, 10-30 mm lower on the pen, choosing the height/direction by probing `api.reach` at runtime (residual < 4 mm, else skipped); middle moves rigidly with thumb/index in every `rotate`; swing aborts and regrips when the thumb+index force drops below 45 % of its value at the start of the swing.
Result iter 4: 0/3. Grip force rose to 4 N (5.4 N with the middle stop) at grasp, but decayed steadily to ~0.7 N by the end of the lift and to 0 during the swing, with the measured thumb-index pad separation shrinking 21 -> 15 mm: the pen is pushed sideways (palm +-y) out of the two convex fingertip pads. A cylinder between two opposed convex pads is laterally unstable; squeezing harder only makes it leave faster. The middle stop on one side did not help.

## iter 5 (last; budget cut to 4 completed iterations)
Changes: tripod grasp. Thumb on the wrist side (-z). Index on the fingertip side tilted toward +y at the grasp height. Middle on the fingertip side tilted toward -y, ~22 mm lower on the pen. The two fingertip-side contacts cradle the pen laterally. Directions and heights are chosen by probing `api.reach` at runtime (falls back to the pinch if unreachable). Squeeze is 4 mm on all three, and all three move rigidly in every rotate/regrip.
Skill: `grip(lift)` returns object-frame points and explicit radial normals for the tripod from palm-frame directions, so `approach`/`grasp`/`regrip` stay consistent as the pen tilts and rolls.
Result iter 5: 0/3. The tripod never engaged. `api.reach` residuals for the middle pad on the fingertip side of the pen were 5.7-7 mm for every height and angle I tried, above my 2 mm accept threshold and 4 mm gate, so the policy fell back to the two-point pinch at 4 mm squeeze. It failed the same way: forces 2.4 N at grasp, 0.3-0.6 N after the lift, then 0 within 0.4 s of starting the negative swing, with pad separation shrinking 21 -> 16 mm.
Takeaway: with contact primitives only, I could not find a stable way to hold a 7.5 mm pen. The fix I would try next is a tripod with a looser reach gate (about 7 mm), or a palm-side support, so the pen is cradled rather than pinched between two convex tips. Budget exhausted (4 completed iterations).
