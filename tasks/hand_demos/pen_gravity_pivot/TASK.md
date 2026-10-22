# pen_gravity_pivot

A pen lies level on a small wooden rest in front of the palm-down hand. Pick it up with the fingertips, holding it near one end, and lift it while it stays level. Then loosen the grip so that the pen swings down under gravity, turning in the fingers about the grasp, and catch it so it ends up hanging near vertical, still held in the fingers above the table.

**Objects.**
- `pen`: a capsule (rounded ends), nominal barrel diameter 7.5 mm, total length 140 mm (cylinder half-length 66.2 mm plus 3.75 mm end caps), mass 5 g, friction 0.9. Its long axis is the capsule's local z. Its pose is readable at any time.
- `pen_rest`: a fixed wooden block, 44 x 20 x 30 mm (long side along the pen, 30 mm tall), not scored. The pen lies across its top with its centre 12 mm from the rest's centre along the pen.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). Table top at z = 0.75. Pen centre near world (0.55, 0.00, 0.784), lying horizontal on the rest, its axis along (-sin 0.15, cos 0.15, 0), i.e. 98.6 deg from world +x (roughly along world y).

**Variation (severity 1).** Pen and rest move together: +-20 mm in x and y, +-25 deg yaw about the vertical through the pen centre. Pen diameter and length each +-10 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of, in this order:
1. *Level hold:* for at least 0.4 s without interruption, the pen centre is more than 70 mm above its start height and its axis is within 15 deg of horizontal.
2. *Pivot in the hand:* the pen's axis, expressed in the measured palm frame, ends up more than 60 deg away from its palm-frame direction at the end of the (latest) level hold, i.e. the pen turns relative to the hand; turning the wrist with the pen clamped does not count.
3. *Caught:* throughout the last 1 s (the bench's settle period), the pen axis is within 20 deg of vertical, the pen centre is more than 50 mm above its start height, the palm-frame pivot of step 2 stays above 60 deg, and at least two fingertip pad centres are within 20 mm of the pen's axis (it is held by the fingers, not resting on something).

Reported metrics for partial progress: max lift, longest level hold, max and final pivot in the palm frame, max angle swept in the world, peak turn rate of the pen axis, final tilt from vertical, and the last-second minima/maxima used by the rule.

**Physical facts.**
- Contacts between the pen and anything else use the pen's own contact parameters: sliding friction 0.9 and **no torsional friction** (nothing resists the pen spinning about a contact normal). Other hand contacts have torsional friction.
- MuJoCo-Warp physics at 1 kHz, control at 50 Hz. Finger position servos: kp 3 N m/rad, kv 0.12, torque limit 0.65 N m. Fingertips are 16-20 mm thick, about twice the pen's diameter.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
