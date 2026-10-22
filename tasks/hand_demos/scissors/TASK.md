# scissors

A pair of hinged office scissors lies flat and closed on the table in front of the palm-down hand. Pick them up so that they are entirely off the table, open and close the blades at least twice while holding them in the air, then set them back down on the table, lying flat with the blades closed.

**Objects.** `scissors`: an articulated two-part object, nominal length 190 mm, mass about 65 g (most of it in the steel blades). Its pose (`object_pose("scissors")`) is the frame of part A at the pivot screw: local +x runs along the blades toward the tips, local z is up when lying flat, and part A's handle loop is on local +y. Part B turns about the pivot (local z) by the opening angle and carries its loop on local -y, so the two loops spread apart as the blades open, as on real crossed scissors. Nominal geometry in the scissors frame:
- blades: steel, 96 mm from the pivot to the tips, 11 mm wide, 2.2 mm thick each (one above the other), friction 0.5;
- handle loops: plastic, friction 0.7, rectangular rings centred at x = -65 mm, y = +-24 mm, outer 45 x 37 mm, walls 5 mm thick and 11 mm tall, so each hole is 35 mm (along x) by 27 mm (along y);
- the hinge opens from 0 (closed, against a stop) to 1.2 rad (69 deg); the pivot has a friction loss of 0.04 N m and light damping, so the blades stay where they are put. The hinge angle is readable with `api.object_joint("scissors_hinge")` [rad].
- When lying flat the pivot frame is about 4.7 mm above the table top (z = 0.75).

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)), fingers pointing along the default palm yaw (0.7 rad from world +x), fingers open. The scissors lie flat and closed with their pivot near world (0.45, -0.20, 0.755) and the blades pointing along the default finger direction (world (0.77, 0.64, 0)), so the blade tips are near (0.52, -0.14) and the handle loops lie toward the arm, near (0.40, -0.24), with loop A (local +y) on the thumb side of the hand.

**Variation (severity 1).** Scissors position +-20 mm in x and y; yaw about the vertical +-25 deg; overall size +-10 % (all dimensions scale together, including the loop holes); mass +-50 %; pivot friction 0.04 N m +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** The opening angle is the hinge coordinate. "In the air" means the lowest point of the scissors (blade tips, loop corners, including their thickness, at nominal dimensions) is more than 20 mm above the table top. All of:
- at least 2 open/close cycles while in the air: the opening angle rises above 10 deg and then falls back below 3 deg, with the scissors in the air from the moment it passes 10 deg until it is back below 3 deg (touching down cancels a cycle in progress);
- at the end, the scissors rest on the table: pivot frame at most 15 mm above the table top, lying flat (local z within ~18 deg of vertical);
- at the end, the blades are closed (opening angle below 3 deg).

Where they are set down on the table does not matter. Progress metrics reported: highest pivot lift, highest clearance of the lowest point, time spent in the air, largest opening reached in the air, the number of completed cycles and the peak opening of each, and the final opening angle, rest height, flatness and xy offset from the start.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.**
- Part A's pose (`object_pose("scissors")`), the hinge angle (`object_joint("scissors_hinge")`) and contact forces with `"scissors"` are available.
- Hand contacts have torsional friction (condim 4, patch 0.005 m); the fingertip pads have sliding friction 0.9. Newton combines friction by taking the larger of the two contacting coefficients.
- The fingertips are rounded shells about 16-20 mm across.
- Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint.
