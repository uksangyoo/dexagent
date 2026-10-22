# card_pickup

A thin, flat tile (a plastic coaster) lies on a hardcover book on the table in front of the palm-down hand. Get it off the book and hold it up in the hand, between the thumb and the index finger.

**Objects.**
- `tile`: a box 90 x 60 x 3 mm (long side along world y), mass 17 g, friction 0.3. Its pose is readable at any time.
- `book`: a static (welded) box, 160 mm (x) x 220 mm (y) x 45 mm thick, friction 0.3. It cannot move. Its pose (box centre) is readable with `api.object_pose("book")`, and contact forces between the hand and the book with `api.contact_forces("book")`.
- The table top is at z = 0.75.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)). Book centred at world (0.55, 0.00, 0.7725), so its top is at z = 0.795 and its near (robot-side) edge runs along x at y = -0.11. Tile centred on the book at (0.55, -0.02, 0.7965), square to the book, so its near end is 45 mm from the book's near edge.

**Variation (severity 1).** Book position +-20 mm in x and y, book thickness 45 mm +-10 %. The tile moves with the book and is additionally offset on it by +-15 mm in x and +-20 mm in y, yawed +-25 deg about the vertical, its length and width scaled +-12 % (thickness stays 3 mm), and its mass +-50 %. The tile always starts lying fully on the book. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** After the bench's 1 s hold of the final targets, both:
- the tile's centre is more than 80 mm above its start height;
- the tile's centre is within 60 mm of the midpoint between the thumb and index fingertip pads.

Reported progress metrics: farthest the tile was slid while still lying on the book (`max_slide_on_book_mm`), largest tilt from flat (`max_tilt_deg`), largest and final lift above the start height, and the final tile-to-pinch distance.

**Physical facts.** Newton combines friction by taking the larger of the two contacting coefficients, so the tile slides on the book at 0.3 but grips the fingertip pads at 0.9. Hand contacts also carry torsional friction (0.005 m patch). Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint. A 3 mm tile is thin compared with the fingertips: contact on its edges is not reliable. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
