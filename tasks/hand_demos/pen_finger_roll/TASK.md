# pen_finger_roll

A pen stands upright in a pen cup on the table. Take it out of the cup and, holding it up in the hand, do two in-hand motions: swing it (tilt it back and forth relative to the hand, then bring it back to where it was) and roll it about its own long axis between the fingers. Keep holding it, clear of the cup, at the end.

**Objects.** `pen`: a capsule along its own z, nominal diameter 7.5 mm, total length 140 mm (tip to tip), mass 5 g, friction 0.8. Its pose is readable at any time. The cup is static and not scored: four walls 3 mm thick and 60 mm tall around a 20 x 20 mm square bore, standing on the table (cup rim 60 mm above the table).

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). The cup is centred at world (0.55, 0.00) on the table (table top z = 0.75), walls square to the world axes. The pen stands vertically in the centre of the bore on its lower tip, so its top is about 140 mm above the table, 80 mm above the cup rim.

**Variation (severity 1).** Cup and pen position +-20 mm in x and y (they move together); cup yaw +-30 deg; pen diameter 7.5 mm +-12 %; pen length 140 mm +-10 %; pen mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal. The pen always starts upright in the centre of the bore.

**Success (check.py).** Angles are measured in the measured palm frame (forward kinematics of the arm joints; +z along the fingers, +x out of the palm face, +y toward the thumb). "Rest" is the pen's axis in the palm frame at the first moment its lowest point is 10 mm above the cup rim. While the pen is held above the rim, the checker tracks
- swing tilt: the angle of the pen axis in the palm x-z plane (rotation about palm +y) relative to rest;
- roll: the accumulated rotation of the pen about its own long axis, relative to the palm.

All of:
- the swing tilt reaches at least +10 deg and at least -10 deg, with a total range (max - min) of at least 30 deg;
- after both sides have been reached, the swing tilt comes back within 5 deg of rest and stays there for at least 0.5 s continuously;
- the roll has a range (max - min of the accumulated angle) of at least 30 deg;
- at the end the pen is still held: its lowest point is at least 10 mm above the cup rim height (70 mm above the table).

Order between the swing and the roll does not matter. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.** Palm frame: +z runs along the fingers, +x is out of the palm face (fingers flex toward +x), +y is toward the thumb. Newton combines friction by taking the larger of the two contacting coefficients (pen 0.8 vs fingertip pads 0.9 -> 0.9). Hand contacts have torsional friction (condim 4, coefficient 0.005) and rolling friction 0.0001. Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint. The pen is a perfectly round capsule, so its roll shows only in its orientation; a dropped pen falls onto the table or back into the cup.
