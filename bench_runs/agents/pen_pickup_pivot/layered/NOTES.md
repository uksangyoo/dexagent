# pen_pickup_pivot at level layered


## iter 1: 0/3
Plan: thumb-index pinch (fingertip layer, move_tips) ~45 mm from the pen centre, pinch line horizontal and perpendicular to the pen (free hinge, no torsional friction); lift the hinge so the far end drags on the table until the pen stands, then lift clear. Middle/ring/little tucked (contacts layer `tuck`), arm via `carry`.
Table-touch probing by finger deflection (|tips()-tip_targets()| change > 1.5 mm) works: pad centres sit 9.5-12 mm above the table when the tip touches; the pen centre is at 3.75 mm, so the pinch contacts are well above the pen's equator (~40 deg wedge).
Result: squeeze of 3 mm/finger gave ~2 N; on lifting the pen squirted out of the wedge within 1 s (watermelon-seed) and was flung ~0.5 m; retry then failed arm IK (pen out of reach). Learned: squeeze must be light and the contacts as low as possible; guard retries against unreachable pens.
Skills: `_until(gen, cond)` runs any motion generator and stops early on a condition (sensor-guarded moves); `dev(api)` = measured minus commanded pad positions (contact/deflection sensing without force sensors).

## (restart) iter_2 folder: killed mid-run, not counted
Partial logs of iter_2 (same policy as iter 1 but extra squeeze) showed the close step reached 4.8 N per finger (targets were set relative to the overshooting commanded targets, ~7 mm into the pen); pinch lost within 1 s of lifting, then the pen was flicked ~0.8 m (probably by `open()`).
Next iteration (iter_3) change: pinch force set from MEASURED pads (targets = measured + 0.6 mm inward, trimmed to 0.08-0.9 N by reading contact_forces), pads 0.4 mm above the table while closing, brisk lift through the low-angle wedge zone (70 mm in 1.6 s) then on to +175 mm, gentle release instead of open().
Skills: `set_pinch(api, gap_th, gap_ix)` = force-by-offset pinch from measured pad positions (fingertip layer); `release(api)` = pads apart along the pinch line then palm up (no flick).

## iter 3: 0/3
Changed: force-by-offset pinch from measured pads (trimmed to 0.08-0.9 N), pads 0.4 mm above the table while closing, brisk lift, gentle release.
Result: all 3 cases the pinch slipped off within ~0.5 s of lifting (pen never rose; pz stayed 4 mm). The closing detection (both forces > 0.02 N) never fired, so pads went too deep (contact angle on the pen steep); trimmed forces ended unbalanced (e.g. thumb 0.82 N vs index 0.25 N); table friction hides the imbalance until lift-off, then the pen squirts. The `closeup` camera is a far view; filming did not resolve the pinch.
Learned: position-offset squeeze cannot set small balanced forces on this hand (finger stiffness differs, ~1 N per 0.5 mm). Retries pushed the pen out of reach.

## iter 4 plan
Index placed as a stop just outside the predicted contact (contact distance from measured pad 'radius' = table-touch height + pen radius), thumb closes until touch (pen cannot roll away); then a per-finger force servo (fingertip layer, every 0.1 s: target z += 2 mm/N x (0.25 N - f_i), clipped 0.3 mm/step, max 2.5 mm commanded penetration) runs during the squeeze AND during the whole lift (palm moves in 0.1 s segments via move_tips(xyz=...)).
Skills: `servo(palm, seconds, ft)` = one 0.1 s step of per-finger contact-force servo with optional palm motion; `lift(dz, T)` = palm lift in servo segments with loss detection; `note()` buffer re-emitted at the end (bench keeps only the last 60 log lines).

## iter 4: (cases 0 and 0.5 failed identically; see below)
Result: the thumb never moved during its straight-line close (measured thumb stayed ~16 mm from the pen axis while commanded to ~10 mm, no IK residual flagged, zero pen force), so no pinch was ever formed; the index tracked its commands within ~1 mm. The thumb stall (table drag or the straight-line per-step IK with a direction constraint) probably also explains iter 3's "hit=False" and unbalanced forces.
## iter 5 plan (launched while iter 4's severity-1 case was still running, to fit the wall clock)
Thumb never moves in z: the ARM parks the thumb pad 14.5 mm from the pen axis (palm point CP over the hinge), the index alone closes (straight line, stop when both forces > 0.015 N) and the index alone runs the force servo (target mean pinch force 0.25 N), thumb is a passive stop. Lift in servo segments as in iter 4.
iter 4 final: 0/3, all three cases identical (thumb stall, zero pen force, "lost" at the first servo segments; pen never moved).

## iter 5: (severity-1 case) 0/3 pattern again
Arm parking of the thumb worked (thumb 15 mm from the pen axis as commanded), but now the INDEX stalled during its straight-line close, ~17.7 mm from the pen axis (commanded ~9.4 mm), with contact_forces reading 0 for both fingers. So whichever finger does the straight close stops 15-18 mm from the pen axis. Geometry check: the tip mesh is 17.9 x 16.1 x 19.6 mm and the distal link leans ~17 deg toward the pen, so the leading tip surface sits ~12 mm ahead of the "pad centre" at pen height: the pads probably DO touch the pen at ~16 mm (pen pinched, pressed into the table, so it cannot roll), and my contact-distance model (pad radius = table-touch height, ~12 mm) was ~4 mm short. Why contact_forces read 0 in that state I could not resolve (earlier iterations did read 0.3-5 N). The force servo therefore saw "no contact", its 2.5 mm max-penetration clamp kept the index from pushing, and lost-detection aborted every lift within 0.4 s.
Learned (for whoever continues): (1) measure the actual pad-to-pen contact distance by deflection (|tips()-tip_targets()| growth), not by force and not from the pad-centre model; (2) don't gate the lift on contact_forces alone - use the pen pose (does it rise/tilt with the palm?); (3) the remaining open question is still whether a light, balanced wedge pinch on a 7.5 mm pen survives lift-off at all in this sim.
iter 5 final: 0/3 (all three cases identical: index stalls ~17.8 mm from the pen axis, forces read 0, every lift aborted, pen never moved).

## Summary / FINAL = 5
Dev results: 1: 0/3, (2: killed, not counted), 3: 0/3, 4: 0/3, 5: 0/3. All tied at 0/3; iter 5 named FINAL as the most recent and most developed policy (iter 1's high pivot/"stood" metrics came from the pen being flung, not from a controlled pivot).
Layers used: arm `carry` (+ palm motion inside `move_tips(xyz=...)`) for approach, table probing, thumb parking and lifting; fingertip layer `move_tips` for preshape, table-touch probing by deflection, closing and the force servo (most calls); contacts layer only for `tuck` (middle/ring/little out of the way) and `open` in iter 1. Synergies and joints were never used: the task needs mm-level pad placement against a 7.5 mm pen, and neither layer gives that directly.
