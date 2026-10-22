# scissors, phase 3 (force level + cross-agent memory, 6 iterations)


## Plan (before iter 1), from reading all five earlier scissors agents
- Reused: the closing-squeeze lift (index in hole A + ring in hole B pushing the inner walls; contacts iters 3-5 and joints iter 4 lifted 3/3 with it),
  the in-air hand-over sequence of joints iter 4 (thumb on A's outer face, index crosses hole A to A's outer wall, ring alone drives B; it opened 25 deg in the air),
  contacts' `reach` placement grid search and pad measure-and-correct loop, `pads_obj`/`status`/`clearance` helpers, joints' little-finger parking pose,
  size from the start pivot height (6.8 mm * s; joints/layered/force).
- New with the force servo: every pad push is `grip` with explicit object-frame directions (the default direction changes when a finger is ungripped).
  Grip targets are kept force-consistent (e.g. thumb 8 = index 4 + ring 4) because inconsistent targets make the whole set drift.
  The servo measures |force| whatever wall a pad touches, so before the ring is servoed the other way it is moved to the hole centre (zero force).
- Skills (policy.py): `L2P` object-frame point -> palm frame from the live pose; `ring_center()` B-frame hole centre; `drive(dir, until, F, tmax)`
  ring force servo with hinge-angle stop and a force boost if the hinge does not move.

## iter 1: 0/3 (nothing lifted; ~2 min wall, the GPU queue was empty)
- Ring pad could not reach hole B's centre (placement error stayed 8 mm at abduction 0.84 = its limit): joints' little-finger parking [0.8,-0.6,0,0] sits in the ring's way
  (this is also why joints' ring Jacobian came out wrong). Index pressed on something at 11 N during the descent at x=-60 (contacts saw the same 14 N).
- The lift grip pushed index -y / ring +y with 25 mm max travel: the index cannot abduct toward the middle finger that far, so the servo IK lifted the pad
  (z +12 mm) over the wall, force 0. Lesson: lateral pad force comes only from abduction (kp 3, 0.65 N m at ~0.1 m lever => <= ~6 N), and a servo
  target beyond the finger's reach makes the IK trade depth for lateral reach. Keep max_travel small and add a downward component to the direction.
- Thumb IK in the air was meaningless (scissors still on the table).

## iter 2: 0/3, but lifted 200 mm in 2/3 (sev 0, 0.5), dropped at the thumb step
- Little parked by `tuck` pose: ring placement now within 0.5 mm. Abduction limits (logged): thumb +-1.57, index [-0.84, 0.26], middle [-0.58, 0.73],
  ring [-0.21, 0.84], little [0, 1.05]. So index and ring can each move only ~10 mm toward the middle finger: the closing squeeze is near their limits.
- The servo lift was confused: the index already read 6-10 N at the hole centre (tip on the inner wall top), so the servo backed it off 6 mm and the
  first lift test failed; the 2nd grip (6 N, 14 mm) lifted (5.8 N each in the air, 26 deg pitch). Sev 1: the ring backed off onto B's outer wall, no lift.
- Drop: engaging the thumb I re-targeted index/ring to 1.5/4.5 N before the thumb had any contact (travel hit 30 mm, 0 N) -> squeeze gone.
- Change for iter 3: position squeeze (pads 4 mm into the inner walls, like joints iter 4), servo only on the thumb; index/ring are springs that
  balance the thumb force automatically. Lesson: with several servoed pads, a re-target must never weaken a hold before the new contact is confirmed.

## iter 3: 0/3. Position squeeze engages cleanly (11-15 N, lift test +10 mm in 3/3) but every case slipped out during the 21 cm lift
- Film: the scissors hang ~35 deg blade-down from the index/ring line and slide off while the arm accelerates. Position springs lose force
  as the loops rotate; in iter 2 the servo-held squeeze (5.8 N) survived the same lift in 2/2 cases.
- Change for iter 4: position squeeze to engage on the table, lift test, then hand the squeeze to the servo (5 N, small travel) for the big lift;
  thumb engaged with force-consistent targets (thumb 2.5 + index 2.5 = ring 5) that drift A toward the thumb; fall back to the squeeze if no thumb contact.

## iter 4: 0/3, lift 3/3 (190-200 mm, 3-4 s in the air), all dropped while the thumb moved in
- Servo hand-over after the lift test works: 5 N each, scissors hang 26-37 deg blade-down. But the servo travel cap (12 mm) saturated and
  forces sagged to 2-4 N as they pitched; the thumb's park->pre path (into a pitched frame, where the loop backs rise) knocked them out
  (sev 0: the thumb pad stayed 30 mm high, i.e. it was blocked by the scissors).
- Change for iter 5: hand-over at a LOW lift (+50 mm palm): the drooping blade tips rest on the table and limit the pitch while the thumb
  and index are re-arranged; the big lift (+19 cm) happens only after the thumb+index pinch on A is confirmed. Servo travel 20 mm.

## iter 5: 1/3 (sev 1 PASS: cycles 17.1/12.5 deg, set down flat and closed). sev 0: 2 cycles in the air but set-down failed; sev 0.5: 1 cycle
- The low hand-over works in 3/3: thumb reaches its pre-pose within 1 mm, thumb+index pinch A (5 N / 3 N), big lift to ~200 mm, ring opens to 15-18 deg.
- My own cycle count was stricter than the scorer (it only read the peak at my stop), so sev 0 kept cycling after 2 real cycles; meanwhile the pair
  pitched further in the A pinch (zR 0.7 -> 0.23) and the curled middle finger touched the loops (2-8 N), blocking closes at ~7 deg.
  Lowering a near-vertical pair straight down jammed it into the table (166 N on the ring), the blades were forced open to 55 deg.
- Closes often stalled at 1.2-1.6 deg (ring at its +y reach); the scorer only needs < 3 deg.
- Change for iter 6: scorer-rule tracker `watch()` sampled in the drive loops, stop after 2; close threshold 2 deg; middle finger kept straight (out of the loops);
  thumb 5.5 N / 40 mm travel; lay-down skill: lower to the table, then if pitched move the pinch back and down along the circle around the resting tip.

## iter 6: 0/3 (budget used). Regression: no lift in sev 0/0.5, sev 1 lifted 41 mm and dropped at the low hand-over
- The only change in the table phase was the middle finger straight instead of curled: the position squeeze then reached only 5/3 N (iter 5: 15/12 N)
  and the lift tests failed. The curled middle was evidently part of the grasp (a third contact/brace between the loops), not just "out of the way".
  The later fixes (scorer-rule counting, 2 deg close, lay-down) were therefore never exercised.
- FINAL = 5 (1/3 dev, the only pass; lift and A hand-over 3/3). Its known weaknesses: it counts cycles more strictly than the scorer and keeps cycling,
  and lowers a strongly pitched pair straight into the table. A next step would be iter 6's counting/close/lay-down changes with the middle curled again.

## Summary
- Dev results: 1: 0/3, 2: 0/3, 3: 0/3, 4: 0/3, 5: 1/3, 6: 0/3. FINAL = 5.
- Reused from earlier agents: contacts' `reach` placement grid search, pad measure-and-correct loop, pads_obj/status/clearance helpers and its
  tuck pose for the little finger; joints' closing-squeeze lift and in-air hand-over sequence (thumb on A's outer face, index across hole A,
  ring alone drives B) and its position squeeze; size from the start pivot height (joints/layered/force). Joints' little parking pose turned out
  to block the ring (iter 1). The brief's "three contacts" lesson matched: the curled middle mattered for the lift.
- Force servo: decisive in the air. The squeeze that survives the big lift is servo-held (position springs slipped in 3/3, iter 3); the thumb only finds
  A's outer face because the servo drives it in; the ring opens/closes B as a force servo stopped by hinge feedback. Pitfalls I hit: the servo backs a pad
  off when a spurious contact (tip on a wall top) already reads the target; a target beyond a finger's lateral reach makes the IK lift the pad;
  inconsistent targets on several pads make the whole grasp drift; |force| does not tell which wall a pad touches, so re-centre before reversing.
