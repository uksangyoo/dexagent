# baoding

Two balls lie in the upturned hand. Circle them around each other in the palm, Baoding style, until they have exchanged places (half a revolution of the pair or more), without letting either ball leave the hand, and finish with the palm still facing up and both balls in it.

**Objects.** `ball_a` (gold) and `ball_b` (blue): spheres, nominal radius 19 mm (38 mm diameter), 50 g each, friction 0.5. Their poses are readable at any time.

**Start.** Palm face up (palm +x = world +z), fingers pointing along world +x, thumb toward world -y; palm origin at about world (0.50, -0.10, 1.00), 250 mm above the table. The fingers start in a cupped posture (the robot's home pose for this scene). The balls are released 3 mm above the palm face at t = 0 and drop into the hand: in the palm frame (mm; +x out of the palm face, +y toward the thumb, +z toward the fingertips; the palm face is at x ~ 26) their nominal centres are `ball_a` (48, -6, 62) and `ball_b` (48, 10, 102), i.e. `ball_a` nearer the wrist, `ball_b` nearer the fingers, about 5 mm apart surface to surface.

**Variation (severity 1).** `ball_b` shifts +-4 mm in palm y and in palm z; the pair turns +-25 deg about the palm normal around `ball_b` (so `ball_a` swings around it); ball radius 19 mm +-10 % (both balls alike, the 5 mm gap between them kept, so `ball_a` moves with the spacing); ball mass 50 g +-50 % (both alike). Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- the ball-to-ball vector (`ball_a` -> `ball_b`), projected onto the palm plane, turns by at least 180 deg about the palm normal, counted as signed cumulative rotation (turning back cancels progress), and counted only from t = 1 s (after the balls have dropped in and settled) while both balls are in the hand and the palm normal is within 30 deg of world up;
- neither ball ever leaves the hand during the run. "In the hand" means its centre, in the measured palm frame, has -10 < x < 75 mm, |y| < 60 mm and -30 < z < 150 mm;
- at the end both balls are in the hand and the palm normal is within 30 deg of world up.

The palm frame is computed from the measured arm joints, so turning the whole hand about the vertical does not count as rotating the balls. Metrics: `swap_deg` (final cumulative rotation), `max_swap_deg`, `both_in_hand_end`, `ever_dropped`, `final_palm_tilt_deg`, `max_palm_tilt_deg`, `ball_centre_dist_end_mm`.

**Physical facts.** Newton combines friction by taking the larger of the two contacting geoms' coefficients, so ball-hand contacts use the hand's 0.9 (ball-ball contacts use 0.5). Hand contacts carry torsional friction (condim 4, 0.005 m patch). Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint. The balls fall freely onto the table if they leave the hand.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
