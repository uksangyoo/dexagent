# baoding at level joints


## iter 1: 0/3 (swap 48-49 deg, b dropped at the end in all cases)
- Policy: gravity "swirl" (arm only, fingers at home): palm tilted 15-19 deg, downhill direction steered by feedback
  on the measured pair angle (bowl model: low point = psi - 90 + lead). Rate-limited tilt vector; pivot about the palm centre.
- Learned: the home hand is not a bowl. Palm slopes ~12 deg down toward the finger bases; there is a channel ("trough")
  along palm y at z ~ 90-103 (MCP region). By t=1 s ball_a already rolled into it at (-31, 90) next to b (8, 103), psi0 ~ 72 deg
  (deterministic across severities). Tilting toward +z/+y just jams both in the trough (a -18, b +19): no rotation for 45 s.
  Home: thumb splayed out (tip y=+101), fingers only mildly flexed (MCP 0.6). The 'closeup' camera does not show the hand.
- Skill: `palm_R(R0, tau)` - palm rotation whose in-plane downhill direction is the 2-vector tau (|tau| = tilt), plus a
  pivot `W - R @ PC` so the palm turns about a palm point.

## iter 2: ABORTED after its nominal case (killed while the other 2 cases were still queued; not counted)
- Policy: curl index+middle to eject b from the trough, strong -y tilt to pass b under a, then re-pack. Little ray folded 0.35.
- Nominal case: folding the little ray (j5_1=0.35) changed the settle (a stayed at (-14,73) instead of the trough), and the
  curl [MCP 1.5, PIP 1.2, DIP 0.8] drove index/middle tips deep into the palm (tip z~55), shoving b to -y and a toward the
  wrist; a rolled off over the -y/wrist corner (wrist end is open there) with only a 3 deg wrist tilt. Killed to save budget.
- Seen on film ('palm' camera works): palm is small, two balls nearly fill it; thumb arcs out on the +y side (a ramp/wall),
  wrist flange is a low wall at z~0-10; the -y/wrist corner is open.

## iter 3: killed before running (not counted). Policy was a psi-scheduled gravity circuit (fingers at home); never scored.

## (resume, 2026-10-22) reasoning before iter 4
- A pure tilt swirl cannot work in this box-shaped palm: both balls get the same acceleration, so they translate together;
  the pair only turns when one ball is HELD (pocket/finger) while the other moves. iter 1's 45 s stall is consistent.
- Pocket-pendulum: under a +y/+z tilt the right ball sits in the top-right pocket (index base + finger-base wall). Rotating
  the downhill direction past ~80 deg toward +y makes the other ball lose its top-wall support and swing round the
  pocketed ball's lower side (right column). Shift the column left (-y tilt, little ray folded = -y wall), then rotate
  the downhill back through +z: the top ball stays in the top-left pocket and the lower ball swings up on its right.
  One cycle LOAD -> SWING_DOWN -> SHIFT_LEFT -> SWING_UP turns the pair ~180 deg. State machine on measured positions.
- Skill: `guard(a, b)` adds downhill components pushing balls away from open edges (z<48, |y| edges), added to every command.

## iter 4: 0/3 (swap 26-56 deg, all dropped ~23-26 s)
- Policy: pocket-pendulum state machine (LOAD lam=40 -> SWING_DOWN lam ramp to 108 deg, tilt 10-13 deg).
- Result: the pre-load works in all cases (a=(-18,95), b=(19,103) at t=1), but NOTHING moves afterwards: the
  finger-base groove holds BOTH balls even with downhill (+y 12, -z 4 deg). The left ball never dropped out of it.
  At tilt >= 12.4 deg with lam 108 the right ball escaped through the gap between index base and thumb
  (y 22 -> 29 -> 49 -> 63 at z ~ 90-100): that +y/finger-base gap is the main exit. Tilt <= 9.5 deg at lam 108 was safe.
- Learned: the "pockets" are a deep groove (1-D channel along y at z 95-103). To get a ball out you need a real -z
  (wrist-ward) tilt or a finger push; the flat palm (z < 85) is the only passing lane.

## iter 5 (nominal case seen first): eject-pass-rise, fingers at home
- EJECT (downhill toward the wrist, ramped): the RIGHT ball (b, deeper pocket) pops out first, at only ~7.6 deg.
  Below the +y end of the groove the palm is a steep ramp toward +y (thumb web): b went y 16 -> 21 -> 39 -> 61 in
  0.4 s while z 93 -> 61, i.e. straight out between index and thumb. The -y guard (gain too high, up to 18 deg)
  then threw ball a out over the -y side. Lesson: never let the right ball leave the groove while the
  index-thumb gap is open; guards must be capped.
- iter 5 result 0/3 (swap 31 / 33 / 157 deg, all dropped). In the 0.5 and 1 cases b sat at y ~ 6-10 (under the middle
  finger) instead of 19 and NOTHING left the groove even at 18 deg toward the wrist: the groove's only weak point is
  its +y end (y ~ 19), which opens onto the thumb-web ramp. The fast RECOVER (60 deg/s back to +z) then flung ball a
  out (-y side in one case; in the severity-1 case a ran the length of the groove past b and out at +y, scoring
  -157 deg on the way). Fast tilt reversals are themselves dangerous for a 50 g ball.

## iter 6: 0/3 (swap 89 / 4 / 154 deg, all dropped). Run concurrently with iter 5's last two cases (time budget).
- Changes: thumb swung toward the index before t=1 (online calibration from api.tips(): q[1] +0.30 moved the pad
  (60,101,74)->(34,107,73), i.e. down onto the palm plane, not inward; -0.30 gave (82,88,74), 13 mm closer to the
  index, so -0.30 was used); ejection steered slightly to -y (lam -165); only the right ball may be ejected; guards capped at 6 deg.
- Result: the right ball now ejects (at ~8 deg, as before) and runs to -y UNDER the left ball instead of out the
  thumb gap - the intended pass happened (counted 142-154 deg in the severity-1 case) - but it arrives far too fast,
  overshoots to the wrist end (z 27-43) and leaves over the open -y/wrist corner (twice); in the 0.5 case the left
  ball popped out at 17 deg and was thrown out the -y side.
- Learned: the mechanism (groove = holder, flat palm = passing lane, steer the ejected ball to -y) is right; what is
  missing is energy control. Next step would be: eject at the threshold and immediately tilt back toward +z (slowly),
  fold the little ray (joint_5_1) as a -y/wrist wall during the pass, and lower PASS tilt to ~4 deg.
- Skill: thumb calibration from api.tips() (try a joint delta, keep it if the pad moves toward the target, else flip).
