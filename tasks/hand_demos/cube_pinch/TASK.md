# cube_pinch

A foam cube rests on the table in front of the palm-down hand. Pick it up with a fingertip grasp, lift it, turn it about the vertical axis while it is held, and put it back down where it started.

**Objects.** `cube`: a box, nominal edge 40 mm, mass 12 g, friction 0.9. Its pose is readable at any time, as are contact forces between each finger / the palm and the cube (`contact_forces("cube")`: thumb, index, middle, ring, little, palm).

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). Cube centred near world (0.55, 0.00, 0.77), one face square to the default palm yaw (0.7 rad).

**Variation (severity 1).** Cube position +-25 mm in x and y, yaw +-30 deg, edge 40 mm +-15 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- **held clear of the table** for at least 0.4 s in total. A step counts when the cube centre is more than 40 mm above its start height and the hand has an opposition fingertip grasp on it (thumb contact > 0.15 N and at least one other finger > 0.15 N);
- while held and clear, the cube centre rises more than 60 mm above its start height at some point;
- while held and clear, its yaw changes by more than 20 deg (modulo 90 deg, cube symmetry);
- at the end it rests flat on the table (within 8 mm of start height), with its centre within 40 mm (xy) of where it started, and the hand has let go (total hand contact force on the cube below 0.1 N).

A bat, flick, or free tumble does not count: lift and turn credit require the opposition grasp. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
