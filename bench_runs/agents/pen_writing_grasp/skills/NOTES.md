# pen_writing_grasp, phase 3 (force level + cross-agent memory, 6 iterations)


## Plan (before iter 1), built from earlier agents' work
- Reused: layered agent's 3-finger chuck (thumb-down hand, pen along palm y, thumb vs index+middle 22 mm apart),
  its approach (turn high above the pen, descend along the pen axis), `align`, `chuck_targets`, `safe_carry` idea;
  force agent's buffered notes; pen_gravity_pivot/force idea of grip directions converted from the palm frame to the
  object frame (`thumb_dir`); synergies/layered insight that the tip should LEAD at a lean (table reaction ~along pen).
- New: force servo on the THUMB ONLY (index/middle position anchors; a servo on all three pads has no equilibrium under
  an external torque and would drift), per-step `move_grip` (carry a palm point along a line while slerping, IK-failure
  safe), tilt 30 deg toward the arm base (keeps the palm < 0.7 m from the shoulder), compression-regulated stroke.
- Scratch: scratch/mock.py (offline mock api, rigid attachment + table clamp + reach limit) for smoke tests.

## iter 1 — 0/3, but the first strokes ever on this task: sev 0 stroke path 43.1/extent 26.7/contact 0.98 (passes), sev 0.5 45.4/25.1/0.996 (passes), sev 1 26.3/18.8
- Chuck grasp + lift 3/3 (tip 80 mm, thumb 4 N servo, index ~1.7, middle ~2.2 N). 30 deg tip-leading tilt reached without any arm IK failure (palm <= 0.69 m from the shoulder). Strokes valid in 2/3.
- Failure in all 3: the pen was not held at the end. Thumb servo travel crept 3 -> 12 mm (cap) during tilt/hover/stroke while the
  index force fell and middle rose: the pen was pushed out of the chuck. Likely cause: the thumb's closing direction was given once in
  the pen's object frame; the pen turning/spinning in the grip turns that direction, so the servo pushes the pen sideways/axially (positive feedback).
- iter 2: `tick()` re-issues the thumb grip with a palm-frame-derived direction every 3rd step during every motion (all motion now per-step),
  thumb 3.5 N, max travel 10 mm; touchdown trigger needs low < 1.5 mm; stroke press: descend if tip off, back off if compression > 0.9 mm;
  per-phase in-palm diagnostics (`state()`: pen axis point at thumb height, tilt in palm, forces, travel).

## iter 2 — 0/3; strokes valid in sev 0 (40.5/31.1/0.998) and sev 0.5 (57.0/28.7/0.998), sev 1 extent 19.3 (<20); never held at end
- Direction refresh did not fix it. New in-palm diagnostics show the mechanism: grasp/lift are rock steady (pen axis at palm x 82, z 119,
  tilt 2 deg, forces 3.5/1.5/2.0 N); during the 30 deg tilt and hover the pen pivots in the palm y-z plane (-2 -> -10 deg), the index
  force falls 1.5 -> 0.8 N, the thumb servo creeps 2 -> 10 mm and the pen slides toward the palm (x 82 -> 74 -> 40 mm) and drops out.
  Interpretation: the index/middle position anchors sag when the gravity direction on the (kp 3) fingers changes with the hand tilt; a
  constant-force thumb then pushes the pen through the yielding anchors. The stroke itself works even with a loose pen (contact 0.998).
- iter 3: anchor servo inside `tick()`: integral loop (move_tips each step) that holds the index/middle pads' MEASURED palm-frame
  positions (api.tips()) at their grip-time values, offset capped at 8 mm; thumb keeps the 3.5 N force servo. Rationale: 2 stiff
  contacts + 1 force-controlled preload is a determinate chuck; 3 force servos would have a free rotation mode.

## iter 3 — 0/3; strokes valid in all 3 (sev 1 42.8/25.4/1.0, sev 0 31.3/22.9/1.0, sev 0.5 52.3/26.1/0.996); not held at the end in any
- Anchor servo works (pad position error ~0, pen y-z tilt now constant through the tilt), but a second escape mode remains: the pen
  slides/rolls along palm -x (toward the palm) over the domed index/middle pads, 81.5 -> 79 mm during the tilt and -> 75 mm during the
  (static) hover, index force 1.4 -> 0.8 N, thumb servo creeping after it; lost between hover and touchdown (x 38 mm).
- Reading: a cylinder squeezed between convex pads is laterally unstable (watermelon seed) and is only held by friction; with a laterally
  compliant thumb the pen can roll across the fixed pads, and the constant-force thumb keeps feeding it out.
- iter 4: thumb pad also anchored, but only in palm x (integral loop on its nominal target; z stays force-servoed); thumb max travel 7 mm;
  diagnostics now log each pad's x offset from the pen axis and the thumb's drift.

## iter 4 — 0/3; strokes valid in all 3 again (sev 1 31.0/23.6/0.997, sev 0 33.7/25.4/0.995, sev 0.5 40.1/21.6/0.998); not held
- Thumb x now held (drift < 0.5 mm) and still the pen slides toward the palm (-x) relative to ALL three fixed pads, starting with the
  tilt (pad offsets from the pen axis +1.6/-2.2 mm at grip -> +11/+8 mm at hover end), then drops out before touchdown.
  Same direction (-x, toward the palm) in all 9 cases of iters 2-4. Conclusion: an in-line chuck of three convex pads on a 7.5 mm
  cylinder has no lateral restoring force; anchoring positions does not help, the cage needs a contact on the escape side.
- iter 5: ring finger as a backstop on the palm (-x) side of the pen: parked at palm x 50 mm (reach scan for y/z), after the grip probed
  +x in 0.4 mm steps until it touches the pen (> 0.3 N, at most 8 mm), then position-anchored like index/middle. Grasp 15 mm lower
  on the pen (top 35 mm above the middle contact) so the pen reaches the ring's y.

## iter 5 — 0/3; strokes valid in sev 0 (45.8/38.4/0.952) and sev 0.5 (73.5/37.9/0.95), sev 1 11.2/9.0; not held
- Bug of mine: the ring backstop was skipped in all 3 cases. The ring was parked at palm y -12 mm (the reach scan's pick) and the pen top
  reached only y -16.8..-17.0, i.e. 4.8-5 mm past it, just under my 5 mm threshold. So iter 5 = iter 4 with a 13 mm lower grasp:
  the same -x escape during tilt/hover (pad offsets +1.6 -> +15 mm).
- iter 6 (last): ring contact y chosen relative to the measured pen top (10/7/14 mm inside the top end, y <= 0), reach-checked, then probed.

## iter 6 (last) — 0/3; sev 0 best stroke 16.8/7.4, sev 0.5 no stroke, sev 1 22.5/10.6; not held
- The ring backstop engaged in all 3 cases (probe 14-17 steps, 0.6-0.7 N) and did its job: the -x escape during the tilt nearly vanished
  (pen axis x 81.9 -> 81.0-81.2 at tilt end, vs 81.9 -> 78 in iters 3-5).
- New failure, already visible as a smaller event in iter 5 and never in iters 1-4: at the start of the hover move the pen suddenly pivots
  in the palm y-z plane (-1 -> -35 deg), pad forces spike to 7-9 N and the anchors saturate, then the pen is gone. Both iters 5 and 6 use
  the 13 mm lower grasp, so my best guess is that the lower hand (thumb side down) now strikes the cup rim on the way to the hover
  point (the hand trails the tip toward the cup at 30 deg lean). This was not verified: no budget left.
- FINAL = 4: all iterations scored 0/3. Iters 3 and 4 both drew a valid stroke in 3/3 cases and lost only the final hold; 4 is the more recent
  of the two and adds the thumb-x anchor.

## Summary
- Dev results: 1: 0/3, 2: 0/3, 3: 0/3, 4: 0/3, 5: 0/3, 6: 0/3. Valid strokes per iteration: 2, 2, 3, 3, 2, 0 of 3. Held at end: 0 in every iteration.
- Layers used: contacts (`tuck` little/ring), fingertips (`move_tips` for the chuck preshape/close, the per-step anchor servo and the ring probe),
  arm (`carry` per step through `move_grip`), force (`grip` on the thumb only, re-issued every 3rd step with a palm-derived direction).
  No synergies or joint targets. Built myself on top of the api: an integral position servo for pads (measured `tips()` -> `move_tips`),
  a per-step palm-point mover with IK-failure recovery, and a compression-regulated touchdown/stroke.
- Force control: the thumb servo made the chuck preload independent of pen size (3.5 N in every case) and kept it through lift and tilt.
  It was also what fed the pen out of the grip once the geometry went unstable: a constant-force pad chases an escaping object.
  The servo acts only along each pad's closing line, and the failures in this task were all lateral (rolling/squirting across convex
  pads) or rotational, so force control was necessary but not sufficient.
- Skills worth reusing: `tick()` (per-step wrapper: grip-direction refresh + pad anchor servo + diagnostics), `state()` (pen pose in the palm
  frame at a pad height, pad-vs-axis offsets, forces, servo travel), `move_grip()`, the 30 deg tip-leading tilt toward the base (no arm
  IK failures in 18 cases), the reach-scanned + force-probed backstop finger, scratch/mock.py.
