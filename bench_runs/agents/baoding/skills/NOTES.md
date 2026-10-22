# baoding, phase 3 (force level + cross-agent memory, 6 iterations)


## Before iter 1: what I took from the earlier agents
- layered (best phase-1 run, 118/112/94 deg): thumb pin of the upper ball from above + PI tilt orbit of the free ball
  under it works 3/3 to about -80..-120 deg; the free ball then stalls at about (y 20, z 70) against the thumb's links.
  Reused verbatim-ish: `tilt_R`, `Ctl` (tilt servo + scorer replica + edge guard), `press_targets`, `pin`, `orbit`, `open_arc`.
- force: little-finger servo pin at the palm centre holds 1.5 N for tens of s, but a single pad on a sphere is levered
  out when the other ball pushes it sideways (thumb wall squeezed b into a, 12-16 N). Velocity-predicted pin descent noted, not used.
- joints: the finger-base groove holds both balls; the right ball ejects first at ~8 deg wrist tilt; once loose, balls
  run too fast. Pure-tilt swirls cannot rotate the pair (both balls get the same acceleration).
- Geometry argument: with one ball fixed as a post in a pocket, 180 deg needs the other ball at the post's antipode,
  which is outside the palm. So both balls must move: P1 orbit (layered), then P2 the THUMB DRAGS the pinned ball
  left/down (pad behind the ball, grip servo presses) while the free ball rises into the vacated right pocket,
  then P3 thumb lifts and gentle +y tilt rolls the dragged ball under the other (to the same thumb-link corner).
  Expected end phi about -170..-190, i.e. total about -(190..230) from phi_t1 15..47.

## iter 1: 0/3 (swap 107 / 48 / 40 deg; NO drops in any case, max tilt 7.6)
- P0+P1 (layered's pin+orbit) reproduced: a stalls at (y 20, z 70) under/against the thumb links, -77..-81 deg
  (sev1 only -39 because phi at t=1 was 10: a was low at t=1).
- P2 thumb drag failed: b went left along the groove only to y -15 (pressed into the ring root, 3 N) and would not
  leave the groove downward; worse, the thumb's own links pressed a with 13 N (thumb + palm forces on a) and
  dragged a left with b. P3 (thumb lifted, +y tilt 5 deg) moved nothing useful.
- Film (palm camera): the thumb arcs over the far (+y) half of the palm; its links lie right over the a-stall spot.
- Reach probe (x=60 mm, mm residual): index reaches y15..30 for z 50-110 (y15: z50/65 0); middle reaches the whole
  centreline y0 z50-110 (0); ring y-15 z50-110 (0-2); little z65 for y-30..15 (0). So the fingers can press balls
  anywhere on the palm centreline from above; the thumb is not the only possible pin.

## iter 2: 0/3 (swap 101 / 89 / 39; no drops again)
- Change: P2 = index finger pins a (at the stall spot) with the grip servo, thumb lifts off b, b orbits a (layered orbit()).
- The index could not reach a: the thumb's links lie over a; the index pad stopped 13-17 mm high on the thumb, servo
  hit max_travel, and the thumb/palm forces on a read 18-25 N (the index pressed the thumb onto a). Skill lesson: check
  contact forces of ALL fingers on the target, not the gripped finger's travel only (brief warned; confirmed).
- After the thumb pad lifted, b left only ALONG the groove (to y -16) and stalled: the orbit's tangential term has a -y
  component and the groove is a channel along y, so any y-tilt slides b along it instead of out of it.
- sev1: during pin() the hold tilt (+z 1.5 deg) let a roll up into the LEFT pocket before P1 started -> P1 orbit failed (+42).
- Next: keep a low during pin(); P2 = thumb re-pins a from above with a normal that has no +y component (so the links
  do not sweep a), grip servo 3 N on it; then EJECT b from the groove with a pure wrist-ward (-z) tilt ramp (+small +y
  toward a), and only once it is out hand over to the PI orbit around a.

## iter 3: 0/3 (swap 104 / 93 / 19; ALL dropped) - worse
- Thumb re-pin of a by layered's pin() (lift 20 mm, hover, press): the instant the pad lifted off b, the thumb's links
  rose and a rolled out over the +y (thumb-web) side (same failure the layered agent had). The thumb links ARE a's wall.
- sev1 (r +10 %, mass +33 %): keeping a wrist-ward tilt (-2 deg) during the P1 pin let the big ball roll off the
  wrist/-y corner before P1 even started.
- Next: (a) PD tilt `keep_low(ctl, name, yt, zt)` holds the lower ball near (y -20, z 68) during settle and pin,
  instead of fixed tilts; (b) `slide_pin`: move the thumb pad in a straight line, no lift, from b's top onto a's top
  (normal without +y), with tilt (+y,+z) holding a in the link/b corner during the slide; then servo 3 N.

## iter 4: 0/3 (swap 71 / 60 / 1; no drops) - P2 never ran
- My first keep_low PD had no baseline: around zero tilt the palm's own slope (~12 deg toward the fingers) plus a -y
  term rolled a straight into the LEFT pocket (y -35, z 89) before/while b was pinned in all 3 cases, so P1's orbit
  had nothing to orbit (it turned +60..+73 the wrong way). Lesson: "level" is not neutral on this palm; the -3 deg
  wrist-ward tilt the layered agent used is the near-neutral baseline.
- Next: keep_low = -3 deg baseline + small PD around (y -22, z 74), applied from t 0.3 s; P2 slide-pin still untested.

## iter 5: 0/3 (swap 53 / 40 / 107; sev1 dropped)
- keep_low with the -3 deg baseline still misplaced a at t=1 (nominal a at z 49 -> phi_t1 28; sev0.5 a again rolled
  into the left pocket during the pin -> P1 failed). Fixed tilts from the layered agent remain the better settle.
- slide_pin (straight thumb slide b-top -> a-top, normal (0.86,0,0.51)): held a in the nominal case (thumb 3.7 N, then
  19.9 N under the servo with travel at -6 mm, i.e. the links also wedge a), but lost a over the thumb side in sev1.
- Eject loop bug: its "b moving" exit fired at once (the slide had jostled b), so no wrist tilt was applied and the
  orbit just stalled with b still in the groove.
- Next (last): P0 constant -3, pin hold tilt 0 (between +1.5 -> left pocket and -2 -> wrist drop), slide depth 2 mm,
  eject exits only on a real 8 mm drop of b after 0.3 s, abort P2 (and end on a safe +z tilt) if a moves >12 mm.

## iter 6: 0/3 (swap 113 / 101 / 86; sev0 and sev0.5 dropped) - budget used up
- P0/P1 back to the proven settings: P1 again -63..-83 in all 3 cases (a at the thumb-link corner, phi_t1 28..55).
- slide_pin lost a over the +y (thumb-web) side in 2/3 cases while the thumb pad slid from b's top to a's top; in the
  third it held (3.6 N) but as soon as the wrist-ward eject tilt reached -6.5 deg the thumb pressed a with 26 N
  (servo retracted to its -6 mm limit, links wedging) and a squirted 12 mm to +y -> my abort guard ended P2 safely.
- Conclusion: once a sits in the thumb-link corner, ANY change of the thumb configuration (lift, slide, re-pin) frees a
  toward the open thumb-web side; and no other finger can reach a there because the thumb lies on top of it. The
  P1 stall spot is a dead end for a two-phase scheme that relies on the thumb for both phases.

## FINAL = 2
All six iterations scored 0/3. Iteration 2 is chosen: it kept both balls in the hand in every case and had the
largest total rotation of the no-drop iterations (101 / 89 / 39 deg; iter 1: 107 / 48 / 40). policy.py restored to it
(later versions kept in scratch/policy_iter*.py).

## Skills (helpers) and reuse
- Reused from bench_runs/agents/baoding/layered: `tilt_R`, `Ctl` (tilt servo with rate limit, edge guard, scorer
  replica), `press_targets`, `pin`, `orbit` (PI tangential-speed orbit), `open_arc`, and its P0 settle (-3 deg wrist
  tilt, thumb park). These gave a reliable -63..-83 deg phase in 14 of 18 cases. Idea-level reuse from force
  (servo pin direction re-issued in palm terms, grip_state checks), joints (groove holds both balls; right ball ejects
  first), and the brief's warning to check all fingers' contact forces.
- New: `drag` (thumb pad behind the ball + grip servo, waypoints; iter 1), `finger_pin` (long-finger pin chosen by
  reach over finger x normal; iter 2), `slide_pin` (straight thumb transfer between balls without lifting; iters 5-6),
  `keep_low` (PD tilt around the -3 deg baseline; iters 4-5, did not work), reach probe for index/middle/ring/little
  (iter 1 log), `hold_until`, per-chunk extra-finger targets with rate limit in `Ctl.step`.

## Force control: where and whether it mattered
- Grip servo used for the thumb drag (2 N, direction re-issued every chunk), the index pin (2 N) and the thumb re-pin of
  a (3 N). It never decided an outcome: in the drag the press held 2 N but the ball would not leave the groove; the
  index servo ran to max_travel because the pad was resting on the thumb (force on the ball 0); the re-pin servo
  retracted to -6 mm because the thumb links wedged a at 20-26 N. The force reading itself (grip_state + contact_forces
  per finger) was the useful part: it exposed the thumb-on-ball contacts that the pad position hid.
