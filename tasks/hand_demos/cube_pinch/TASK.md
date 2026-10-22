# cube_pinch

A foam cube rests on the table in front of the palm-down hand. Pick it up with a fingertip grasp, lift it, turn it about the vertical axis while it is held, and put it back down where it started.

**Objects.** `cube`: a box, nominal edge 40 mm, mass 12 g, friction 0.9. Its pose is readable at any time.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). Cube centred near world (0.55, 0.00, 0.77), one face square to the default palm yaw (0.7 rad).

**Variation (severity 1).** Cube position +-25 mm in x and y, yaw +-30 deg, edge 40 mm +-15 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- the cube rises more than 80 mm above its start height at some point;
- while it is more than 60 mm up, its yaw changes by more than 25 deg (modulo 90 deg);
- at the end it rests flat on the table (within 5 mm of start height) with its centre within 15 mm (xy) of where it started.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
