# pen_pickup_pivot at level force


Plan (before iter 1): thumb-index pinch near the +palm-y end of the pen (pinch line horizontal, across the pen),
held by api.grip (force servo, 1 N per pad). The pen has no torsional friction, so the pinch is a free hinge:
lift the pinch straight up, far end drags on the table, pen swings to vertical and stands on its end, keep lifting.
Then rotate the palm about the pinch to cancel the out-of-swing-plane tilt (set by the pinch-line slope).
If the stand was worse than 15 deg, lower the levelled pen onto its end and lift again.
Layers: move_joints (preshape of middle/ring/little), move_tips (thumb/index pinch pads, palm frame), carry (arm),
grip (force servo for the pinch closure and through all lifting).
Skills written in policy.py: level() rotates the palm about the pinch point to make a hinged pen vertical;
lift_until() lifts in 4 mm chunks while estimating the pen half-length from its pose while the far end drags
(h = (z - table - r)/|axis_z|), so the lower-end height is known without knowing the length.

iter 1 (0/3): pinch + straight drag-lift. Grip settled at ~1.1 N per pad (pad centres rest ~10 mm above the table,
~6 mm above the pen axis: the fingertips sit on the table). On lifting, the pen end rose ~10 mm (pivot 4-7 deg),
then slipped out (forces 0, servo at max travel 15 mm); the pen never left the table. Hypothesis: the pinch is not
a free hinge (pad faces give line contacts that resist rotation), so the drag torque peels the pads off.
Force control: the grip servo closed the pinch reliably (1.0 N reached in all cases) but cannot prevent slip.

iter 2 (0/3): arc the pinch about the far end while rolling the palm about the pinch line (+roll, little side down),
so the grasp need not hinge. Grip raised to 1.5 N: the index could not reach it (0.0-0.37 N at 14.4 mm servo travel,
it rides over the pen), and the pen dropped within the first 15 deg of the arc in all cases. The roll then hit arm IK
failure at 52 deg (palm swings low toward the robot base), and an unguarded retry carry crashed the policy.
Learned: the index fingertip can only press ~1 N on a pen lying on the table; asking more makes it ride over the
pen (force servo target too high is harmful). Roll direction matters for arm reach.

iter 3 (0/3): pinch the little-finger-side end and roll thumb-down instead (keeps the palm away from the robot base);
grip 1.0 N, a retry with distal links pointed straight down (move_tips directions). The index never touched the pen in
any case (0 N, servo travel 9.6 mm = its rate limit over the 1.2 s grip window); thumb ~0.75 N. Pen never lifted, so
the arc+roll was never exercised. The directions variant had 5 mm thumb IK residual and also 0 N.
Learned: pad centres read ~10 mm above the table while commanded ~8 mm: the fingertips were pressed on the table,
and table friction plausibly stalls the weak index as the servo tries to close it.

iter 4 (0/3): pads command-hover 10.5 mm above the table (they no longer press on it), re-centred on the pen's measured
pose in the measured palm frame, grip 0.9 N with one re-close, up to 3 grasp+pivot attempts (both ends), all moves
IK-guarded. The first grasp now closes reliably (thumb 0.9-1.0 N, index 0.65-1.07 N, servo travel 2-4 mm), but the pen
drops within the first ~11 deg of the arc in every case (forces 0 by step 3), so the roll/level/stand/lift phases were
never reached. Later re-grasps were worse (the pen had been nudged; thumb IK residual up to 76 mm on the +u end).
Conclusion: with ~10 mm thick tips resting at the table, the pinch contacts sit ~6 mm above the pen axis and cannot carry
the pen; the missing piece is a grasp that gets material below the pen's equator (or first raises one end), not force control.
Force control: grip() made closing robust once the tips were off the table (it found contact on both pads regardless of
pen offset), and logging its travel diagnosed the stalled index in iter 2-3; it did not help the lift itself.
FINAL = 4 (all iterations 0/3; 4 is the most robust: no crashes, reliable closure).
