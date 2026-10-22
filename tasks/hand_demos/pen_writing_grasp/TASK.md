# pen_writing_grasp

A pen stands in an open cup on the table. Take it out of the cup in a grasp you could write with, put its tip on the table away from the cup and draw a stroke with the tip on the table, then lift the pen and keep holding it.

**Objects.** `pen`: a capsule, nominal 7.5 mm diameter and 140 mm long end to end, mass 5 g, friction 0.8. Its pose is readable at any time; its long axis is the body z axis, and its tip is whichever end is lower. The cup is a fixture (four static walls, not a scored object): a 20 x 20 mm square bore with 3 mm walls, 60 mm tall, open at the bottom, so the pen stands on the table inside it. The table top is at z = 0.75, friction 1.0.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). Cup centred near world (0.55, 0.10, 0.75); the pen stands upright in it with its lower end 1 mm above the table, so its upper end is about 80 mm above the cup rim.

**Variation (severity 1).** Cup and pen position +-20 mm in x and y (they move together); pen initial lean +-3 deg about x and y, after which it falls against the cup bore and rests leaning up to ~8-10 deg from vertical; pen diameter +-10 %, length +-10 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal (pen exactly upright and centred).

**Success (check.py).** All of:
- the pen's tip rises more than 62 mm above the table (out of the cup) at some point;
- afterwards it draws a stroke: a run of control steps in which the pen touches the table with its tip (lower end within 3 mm of the table, pen within 60 deg of vertical, not touching the cup) with the tip more than 30 mm (xy) from where the pen stood in the cup; gaps up to 200 ms do not end a stroke. Some stroke must have a tip path of at least 25 mm, reach at least 20 mm from its starting point, and be in contact for at least 80 % of its duration;
- at the end the pen is held: its tip more than 30 mm above the table, touching neither table nor cup, and moving less than 5 mm over the last 0.5 s.

Reported metrics include the maximum tip height, the best stroke's path, extent, contact fraction and mean lean, and the final tip height.

**Physical facts.** Palm frame: +z runs along the fingers, +x is out of the palm face (fingers flex toward +x), +y is toward the thumb. Newton combines friction by taking the larger of the two contacting coefficients (pen 0.8 vs fingertip pads 0.9 -> 0.9; pen on table -> 1.0). Hand contacts have torsional friction (condim 4, coefficient 0.005) and rolling friction 0.0001. Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint. There is no ink model: "drawing" means the tip is in contact with the table while it moves.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
