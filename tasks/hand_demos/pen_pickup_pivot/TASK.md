# pen_pickup_pivot

A pen lies flat on the table in front of the palm-down hand. Pick it up and bring it upright: pivot it until it stands vertically on one of its ends on the table, then lift it clear of the table and hold it upright in the air.

**Objects.** `pen`: a capsule (cylinder with hemispherical ends), nominal barrel diameter 7.5 mm, total length 140 mm (cylinder half-length 66.2 mm plus 3.75 mm end caps), mass 5 g. Its pose is readable at any time; its long axis is the pose's local z.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)), fingers pointing along the default palm yaw (0.7 rad from world +x). The pen lies on the table, centred near world (0.515, -0.070, 0.754), its long axis horizontal and perpendicular to the default finger direction (across the fingers, along world (-0.64, 0.77, 0)). The table top is at z = 0.75.

**Variation (severity 1).** Pen position +-20 mm in x and y, yaw about the vertical +-25 deg, barrel radius +-10 %, total length +-12 %, mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** All of:
- at some point the pen stands on one end on the table: its lowest point is within 3 mm of the table top while its axis is within 20 deg of vertical;
- at the end, its lowest point is more than 30 mm above the table (held clear, so it is carried by the hand);
- at the end, its axis is within 15 deg of vertical.

Progress metrics reported: largest pivot from the starting axis, smallest tilt from vertical while standing on the table, highest lift of the pen's lowest point, and the final tilt and height.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.**
- Every contact involving the pen uses the pen's own contact parameters: sliding friction 0.9 (against the fingers and the table alike) and no torsional friction, so a pen held between two contact points can turn freely about the line joining them. (The hand's own contacts elsewhere have torsional friction.)
- The fingertips are rounded shells 16-20 mm thick, much thicker than the pen.
- The pen is light: gravity on it is about 0.05 N, so small contact forces dominate its motion.
