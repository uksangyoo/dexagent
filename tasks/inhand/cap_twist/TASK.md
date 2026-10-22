# cap_twist

A screw-top jar stands fixed on the table under the palm-down hand. Unscrew its lid: turn the lid a full turn counter-clockwise (seen from above) using the fingers, while the palm holds still over the jar. Turning the lid by rotating or sweeping the wrist/arm does not count.

**Objects.** `jar`: fixed to the table (it cannot move or tip). Its pose (`object_pose("jar")`) is the centre of the jar's base on the table top; the jar axis is vertical.
- Jar body: cylinder 80 mm diameter, 100 mm tall; a shoulder ring 72 mm diameter, 7 mm tall on top of it.
- Lid: cylinder 70 mm diameter, 15 mm tall, 18 g, friction 1.0, centre 115.5 mm above the table (top face 123 mm up). Red with a white stripe and a dark dot on top.
- The lid sits on a single hinge about the jar axis. Thread friction is modelled as a joint friction loss of 0.02 N m (a lid already cracked loose) plus light damping. The thread lead is not modelled: the lid does not rise or come off, it only turns, either way, without limit.
- Contact forces between each finger/palm and the lid: `contact_forces("lid")` (`"jar"` includes the lid and the body). The lid's turn angle is readable with `api.object_joint("lid_twist")` [rad]; it increases when the lid turns counter-clockwise seen from above (the scorer measures the same coordinate).

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)), `hand_down(0.7)`, fingers open. Jar base centred near world (0.55, 0.00, 0.75).

**Variation (severity 1).** Jar position +-20 mm in x and y; lid radius 35 mm +-10 % (jar body and shoulder radii follow, staying 5 mm and 1 mm wider than the lid); lid mass +-50 %; thread friction loss 0.02 N m +-50 %. Lid height and jar height do not vary. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- at the end, the lid has turned at least 360 deg counter-clockwise from above (net, measured from its start angle; turning it back counts against you);
- the palm holds still while the lid turns: over every control step in which the lid moves faster than 5 deg/s, the palm body's rotation about the vertical spans at most 10 deg and its position spans at most 15 mm.

Metrics also report the max/min turn reached, how long the lid was turning, and the palm's yaw range and spread while it turned. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.** Newton combines friction by taking the larger of the two contacting coefficients (lid 1.0 vs fingertip pads 0.9 -> 1.0). The lid's torsional and rolling friction coefficients are 0.005 and 0.0001. Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint.
