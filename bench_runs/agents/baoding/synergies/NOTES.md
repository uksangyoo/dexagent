# baoding at level synergies


## iter 1 — 0/3 (swap 33/25/27 deg, both balls lost every case)
- Kept the home finger posture; arm tilted the palm 15 deg and swept the downhill direction psi from +y (thumb) round the palm, PD-free weak safety bias. Logged api.describe(), home tips, ball traces.
- Home "cup" is nearly flat: tips x~81 mm, thumb splayed out at y=+101. Balls settle side by side at the finger bases (a y~-30, b y~+8, z~95-103; pair angle ~75 deg), sitting in pockets: at 15 deg tilt they creep only ~20 mm in 1.5 s, then b crests the index base and rolls off the open thumb side (y>30), a off the little side. No walls on +y, -y or wrist side.
- describe(): s0 = whole-hand curl (index/middle pads -45..-50 mm z, ring deep, little in +26 y); s1 = middle extends / little curls in; s2 = thumb abducts out (+37 y) & little spreads (-17 y), so -s2 brings thumb and little in; s3 = thumb in & up (-21 y, +14 z); s4/s5 smaller thumb/middle/index tweaks. home coeffs [-1.14,-0.45,0.77,-0.71,-0.74,0.89].
- Skill: Tracker (replica of the scorer's signed cumulative rotation), tilt_pose(R0, W, vec) = palm pose tilted so palm-frame direction vec points downhill, pivoting about a palm point.

## iter 2 — 0/3 (swap 4/3/8 deg; balls kept in hand in all cases)
- Cupped with synergies (home + [0.5,0,-1.2,0,0,0]); tilt 10 deg with psi fed back from the pair angle (psi = ang - 90 + 35 deg lead, rate limited) + PD virtual-wall tilt keeping balls in a safe box.
- Feedback deadlocked: psi converged to a fixed 22 deg, 10 deg tilt toward the finger pocket never dislodged the balls, pair never turned. Lesson: the drive must keep advancing open-loop; >=14 deg needed to move balls out of the finger-base pocket.
- Cup posture measured tips: thumb (51,82,80) (only came in to y=82, not the linear-predicted ~50), ring (94,-18,135) and little (79,-51,111) curl up into walls on the -y/finger side. Synergy linear predictions are rough far from the mean.
- Skill: contain(api, Rp, balls) = PD virtual-wall tilt from ball position + palm-frame velocity.

## iter 3 — 0/3 (swap 101/5/13 deg; balls lost in all cases)
- Thumb further in (cup = home + [0.5,0,-2,1.2,0,0]) + open-loop 14 deg swirl at 24 deg/s + PD virtual walls.
- Balls stayed jammed at ang ~70-75 for ~6 s of swirl in every direction (even 14 deg toward the wrist does not move them): they sit ~6 mm below the palm face (centre x~40 vs face 26+19) in a trough along the finger-base crease, so only y-tilts slide them (slowly). Tilt toward the thumb squeezed b against the inward thumb (thumb/middle forces 6-7 N, saturated), and the balls popped out violently (one case got 100 deg as they flew). Pushing the thumb into the ball space was a mistake.
- Learned: gravity alone cannot make the balls pass each other here; one ball must be lifted out of the trough by fingers, then steered by palm tilt (it rolls freely on the flat palm).

## iter 4 — 0/3 (swap 22/17/0 deg; balls lost in all cases)
- (Resumed after the pause; the earlier iter_4 folder was a killed run, so it was removed.) Same plan as before (P1 slide +y, P2 index+middle synergy curl EJECT=[0,-1,0,0,-1,2] to eject the +y ball wrist-ward, P3 ball-on-plate PD steering it behind the other, P4 roll back in), now with generic roles, retries/reseat, soft virtual-wall guard, and stopping on the tracked rotation.
- Failed before the finger primitive was ever tested: P1's 15 deg thumb-ward tilt, ramped in quickly (alpha 0.15), threw the +y ball over the open thumb side within ~0.7 s. The brake trigger (y > 16 mm) came too late because the trough does not slow the balls much once they are moving. In sev 0.5 the -y ball also left over the little side. In sev 1 (pair_rot 19 deg, r +7.5 %) ball_a rolled out of the hand while the policy was only holding, before t = 1.1 s, so the home posture does not even contain the drop at high severity.
- Learned: with the open home thumb, any thumb-ward tilt >= ~10 deg is unsafe. Next steps would be closing the thumb partway (s2 about -1.5) as a soft wall before any +y tilt, and catching the drop (curl the little/ring side slightly) during the first second.
- Skill: guard(ps, vs) = soft virtual-wall tilt for several balls. Tracker reused for in-policy termination.

## FINAL
- All four iterations scored 0/3. FINAL = 2: it is the only iteration that kept both balls in the hand in every case (swap 4-8 deg). Iterations 1, 3 and 4 lost balls in every case.
