# cube_palm

A foam cube rests on the table in front of the palm-down hand. Pick it up with the fingers, turn the hand over palm-up and transfer the cube from the fingers onto the open palm, so that for a while the palm alone carries it with the fingers not gripping it. Then turn back and set the cube down on the table near where it started, and let go of it.

**Objects.** `cube`: a box, nominal edge 40 mm, mass 12 g, friction 0.9. Its pose is readable at any time, as are contact forces between each finger / the palm and the cube (`contact_forces("cube")`: thumb, index, middle, ring, little, palm).

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)), `hand_down(0.7)`, fingers open. Cube centred near world (0.54, -0.024, 0.77), resting on a face, turned 0.7 rad (40 deg) about the vertical, i.e. square to the default palm yaw.

**Variation (severity 1).** Cube position +-20 mm in x and y, yaw +-25 deg, edge 40 mm +-12 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- **carried on the open palm** for at least 1.0 s in total (control steps need not be consecutive). A step counts when all hold at once:
  - the palm faces up: the palm-frame +x axis (out of the palm face) has a world-z component above 0.7 (within about 45 deg of vertical);
  - the cube centre is more than 100 mm above its start height;
  - the palm link pushes on the cube with more than 0.02 N, while the five fingers together push on it with less than 0.5 N;
  - the cube lies on the palm: its centre, in the palm frame, is less than 60 mm out from the palm origin along +x and within 45 mm of it along y;
- at the end the cube rests on the table on one of its faces (some cube axis within about 8 deg of vertical, centre height within 5 mm of its start height), with its centre within 80 mm (xy) of where it started;
- at the end the hand has let go: total hand contact force on the cube below 0.05 N.

Metrics also report the max lift, the most palm-up orientation reached while the cube was lifted, total and longest continuous time on the open palm, the final offset and height error. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.** Palm frame: +z runs along the fingers, +x is out of the palm face (fingers flex toward +x), +y is toward the thumb. Newton combines friction by taking the larger of the two contacting coefficients (cube and fingertip pads are both 0.9). Hand contacts have torsional friction (condim 4, coefficient 0.005) and rolling friction 0.0001, so a two-point pinch resists only a little twisting about the pinch axis. Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint.
