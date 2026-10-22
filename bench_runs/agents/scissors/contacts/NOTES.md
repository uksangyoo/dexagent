# scissors at level contacts


## iter 1 (0/3): thumb in loop A + ring in loop B, closing squeeze on the inner walls, plan to lift and roll the hand
- The pads went into or onto the loops and reported 8-15 N against the scissors, but every grip let go within the first 50 mm of lift, so the scissors never left the table. The force was most likely pads pressing DOWN on the wall tops or edges rather than squeezing sideways. A two-finger squeeze on 11 mm walls through the holes is unreliable.
- The film shows the shank geometry: thin bars run from each loop's blade-side wall to the pivot, forming a triangle with open space inside. The pivot height at t=0 is not settled (6.8-7.3 mm). After settling, the rest height tracks the size (4.6 mm at size 1.0, 4.8 at 1.03, 5.0 at 1.08, roughly 4.65*s).
- Skill: placement search. At runtime, api.reach() is checked over a grid of palm offsets and the offset with the lowest worst-case IK residual is kept. I also wrote a helper that expresses palm-frame pad targets as object-frame points with normals, so approach() can pre-shape the pads. That is self-built fingertip placement on top of the contact API.

## iter 2 (killed mid-run when the study paused, not counted; one sev-1 case finished, 0 lift)
- x-pinch of loop A (thumb behind the rear face, index in front of the blade-side face, both at y=33 mm) with the middle finger in hole B. The pads rode up onto the wall top edges during the squeeze. Pad centres ended 10.8 mm (thumb) and 15 mm (index) above the table, but the walls are only ~11 mm tall. The thumb could not get its pad centre below ~14 mm (the thumb tip hits the table first). Index and middle pads can reach ~7.5-8 mm. The lift raised the pivot 10 mm, then it fell.
- Lesson: on 11 mm walls, push laterally only with index/middle/ring pads placed deep in the holes, with z held low. A pinch with its line along x leaves roll about x to torsional friction. A y-normal pair spanning both loops (closing squeeze) is balanced through the closed stop.

## resume (iter 3 = v3): closing squeeze with index deep in hole A and ring deep in hole B, thumb on A's outer face
- Pad placement uses a measure-and-correct loop: preshape above the holes, measure pads in the object frame, correct x/y, then lower straight down into the holes. A pad is retried if it stays high (that means it is on a wall). Lift test at 15 mm, and re-squeeze harder if the pivot did not follow.
- Pitch is resisted only by torsional friction (normals along y). If it loses, the scissors should hang from the two contacts like an axle, and the blade tips stay >20 mm up.
- Cycles: index relocates to A's outer wall (pinching A against the thumb). The ring goes to B's outer wall and api.rotate()s about the pivot to open, then goes to B's inner wall and rotates back to close.
- Skill (self-built on the contact API): goto() places pads at exact object-frame points with straight lines (grasp with +z normals, zero squeeze), plus a pad-error correction loop that uses pads_obj(), the measured pads in the object frame.

## iter 3 (0/3, but the lift worked 3/3): closing squeeze, index in hole A + ring in hole B
- All 3 cases (sizes 0.97-1.0, mass up to 1.22x) lifted 115-122 mm, nearly flat (zR 0.95-0.98, so the pitch held) and closed, with grip forces 4-10 N in the air. The first squeeze (12 mm) slipped. The re-squeeze with 20 mm penetration and the pad target 4-5 mm below where the pad stops (it presses down) held every time.
- Index/ring pad centres bottom out at ~12-13.5 mm above the table inside the holes, so my "pad on a wall" check misfired (that is the tip limit, not a wall). Pad placement after the measure-correct loop was within ~1 mm in x/y.
- Failure: to start the cycles the index moved off A's inner wall to its outer wall, expecting the thumb to oppose. The thumb was not in contact. My target had a sign error, and the thumb pad hovered ~20 mm up. The scissors dropped instantly.
- Lesson: never release a closing-squeeze finger until its replacement contact is confirmed by force. In the air the table no longer limits fingertip depth, so contacts can go to mid-wall height.

## iter 4 (v4): one squeeze straight at 20 mm; then in the air: thumb on A's outer face at mid-wall, optional middle "shelf" under the inner walls, index to A's outer wall, ring cycles
- The lift now holds on the first try: 25-30 N squeeze forces on the table, 3.6-11 N in the air, pivot up 115 mm, pitch up to 32 deg in the heavier sev-0.5 case (zR 0.85, clearance 64 mm).
- Failure: the thumb grasp went in a straight diagonal from above/outside to the mid-wall target, so its path crossed A's outer wall top. That pushed A down and stripped the scissors out of the index/ring squeeze, before any cycle.
- Fix for iter 5: the thumb first descends outside the wall, then presses inward. The index relocates only if the thumb force is >1 N (with a second, deeper thumb try); otherwise skip the cycles and keep the grip.
- iter 4 final: 0/3. All three lifted 74-121 mm clear, all dropped at the thumb step.

## iter 5 (0/3, final budget): thumb descends outside the wall first, index relocation gated on thumb force
- Lifted 77-104 mm in all 3 cases (2.5 s in the air at sev 0/0.5). The sev-1 case (yaw -15 deg, pivot friction 1.3x) slipped out during the 13 cm lift itself. In sev 0 the thumb's straight descent outside A's outer wall still dislodged the scissors: the thumb could not reach the mid-wall target (IK residual 6 mm), and its links swept the loop. The gate then skipped the cycles, so no open/close was ever attempted in the air.
- Bottom line: the index+ring closing squeeze through the holes (20 mm penetration, pads pressed down) is a reliable lift and carry. What I never solved was a second, independent hold on part A, which the ring needs before it can open B. The thumb cannot get low enough on 11 mm walls on the table, and in the air every thumb approach disturbed the grip.
- Next idea (untested): open with a wedge. Press the middle pad down into the 11 mm slot between the loops while the ring backs off, so both inner walls stay pinched (index/middle on A, middle/ring on B) and no finger ever has to leave its wall.
FINAL = 5 (all iterations 0/3; it is the most recent, with the safest logic and the best in-air time at sev 0/0.5).
