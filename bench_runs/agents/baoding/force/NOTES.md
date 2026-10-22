# baoding at level force


## Plan (before iter 1)
Pinned-pivot gravity orbit: a finger (little preferred, else ring, else thumb with reversed direction) presses ball_a
onto the palm with `api.grip` (force servo, 1.2 N); the palm tilts 18 deg (arm `carry`, pivot at ball_a) with the
in-palm downhill direction pointing from ball_b to ball_a plus a rate-controlled lead, so ball_b leans on ball_a and
orbits it. Own tracker mimics the scorer's signed rotation; stop at 205 deg and level the palm.
Helpers: `tilt_R(R0, theta, alpha)` palm rotation giving in-palm downhill direction theta; `tracked()` wraps any api
generator to update the rotation tracker each step.

## iter 1: 0/3 (swap 46/37/36 deg, both balls dropped every case)
Settle data: balls do NOT stay at nominal. In 0.6 s ball_a rolls ~35 mm to the palm's low spot beside ball_b:
a ~(40,-30,90), b ~(39,9,103) mm palm frame, touching, pair across the palm (phi ~70 deg). Ball centres at x = r + 21 mm.
Home tips: thumb (59,101,74) (far out on +y, no wall), index (80,58,167), middle (80,5,177), ring (80,-23,169), little (57,-48,149).
Pin failed: radius estimate from x was wrong (clipped to 15 mm), so the little pad was IK'd into the ball, pushed it
aside, ended on its upper shoulder (pad 18 mm off-centre); force servo then pressed sideways; at the first 18 deg
tilt ball_a rolled out from under the pin, both balls fell. Force servo itself did reach 1.15 N quickly.
Lesson: with ball_a at the little edge no 38 mm orbit fits in the palm anyway -> pin ball_a immediately after the drop.

## iter 2 changes
Chase-pin from t=0 (radius from initial gap: r = (dist-5mm)/2), pad descends onto the current ball top, grip with
explicit straight-down direction (`directions` in object frame); slower ramp to 15 deg; pin-drift guard (reduce tilt,
re-pin once if lost); safe-box guard on ball_b (lead -> 0); orbit direction from a geometric path check.

## iter 2: 0/3 (swap 5 / 1 / 68 deg; sev0 kept both balls, others dropped ball_b or ball_a)
Sev1 is the proof of concept: little pin landed on ball_a at (4,69) mm and the force servo held it at 1.5 N for 45 s
through 15-20 deg tilts (ball_a never moved >2 mm). Ball_b orbited +80 deg in 0.5 s but too fast (lead started at its
60 deg max because the measured rate was 0) and rolled off over the thumb side. Sev0: chase too slow, ball_a reached
the low spot (-30,87) before the pin; the pin was the ring (little reach 5 mm > 4 mm threshold), whose links then
jammed ball_b: it did not move at 21 deg tilt for 40 s. Sev0.5: ring pin missed, ball_a escaped.
Force control mattered: the straight-down servo is what makes the pin hold under tilt (iter1's position-only press slid off).

## iter 3 changes
Faster pin (0.2 s to above, 4 x 40 ms velocity-predicted descent, grip by ~0.6 s), prefer little (reach < 8 mm).
Orbit: position reference running ahead at 18 deg/s (max 20 deg ahead), lead = 28 deg + 1.2 e - 0.8 w, clipped
[-35, 50]; lead -> 0 when ball_b separates >3 mm or leaves the safe box; tilt 13 deg (max 20).

## iter 3: 0/3 (swap 57 / 12 / 7 deg; sev0 dropped ball_b, sev0.5 and sev1 kept both balls but ball_b never moved)
The fast velocity-predicted pin works: little pad lands on ball_a within ~0.6 s, ball_a pinned near the palm centre
(sev0 (0,66), sev1 (-5,73)) and held at 1.5 N for the whole run (drift <2 mm at 20 deg tilt). Sev0.5 chose the ring
(little reach 8.5 mm) and ball_a rolled to (-22,76) first.
New failure: ball_b sits in a pocket at the finger roots (always z ~ 102-103 mm) and does not leave it even at 20 deg tilt
with 50 deg lead (probably wedged against the little finger's links, which pass ~27 mm from it). In sev0 it broke free
after 4 s (stiction) and then ran off over the +y side in 0.5 s, faster than the tilt loop could brake.

## iter 4 changes (last)
Little threshold 12 mm; thumb moved in as an outer wall on the +y side of the orbit (pad at a_y + 3r + 10 mm) if
reachable; when stalled raise tilt 0.5 deg/tick up to 24 deg, and as soon as ball_b moves (>8 deg/s) drop tilt back
to 13 deg so the breakaway does not become a runaway; log contact forces on ball_b to check the wedge hypothesis.

## iter 4: 0/3 (swap 28 / 13 / 79 deg, max 106 / 13 / 84; sev0 and sev1 dropped, sev0.5 kept both but no rotation)
The stall -> more tilt rule (to ~21 deg) broke ball_b out of the finger-root pocket, and the thumb wall caught it:
ball_b rode the thumb around ball_a to phi ~100-107 deg (b at y ~40, z ~56), the furthest any iteration got. Then the
pin was lost: little-finger contact force on ball_a jumped to 12-16 N and ball_a popped out sideways (ball_b, now
on the wrist side, levered ball_a out from under the single pad). The re-pin did not recover, and the level-out after it
briefly went past 30 deg of tilt.
Sev0.5: little pad (servo backed off 6 mm) also pressed on ball_b (13 N on ball_b from the little finger),
which wedged both balls. That confirms the wedge hypothesis for one case. In the others ball_b rests against the index/middle roots (0.2 N).
Force control: the grip servo was essential for the pin (1.5 N straight down held ball_a at 13-21 deg tilt for tens of
seconds). It cannot keep a single-pad pin on a sphere once the other ball pushes ball_a with a large sideways load.
FINAL = 4 (most rotation; all dev iterations 0/3).
