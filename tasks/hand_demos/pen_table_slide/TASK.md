# pen_table_slide

A pen stands upright on its tip in a guide block on the table. Grasp the pen while its tip rests on the table and lower the hand so the table pushes the pen up through the fingers: the pen must slide at least 15 mm through the grasp while its tip stays on the table. Then lift the pen clear out of the guide and hold it in the air.

**Objects.**
- `pen`: a capsule (round barrel with hemispherical ends), nominal diameter 7.5 mm, overall length 140 mm (centre to either end 70 mm), mass 5 g, friction 0.8. Its pose is readable at any time (its long axis is the body z axis), as are contact forces between each finger / the palm and the pen (`contact_forces("pen")`: thumb, index, middle, ring, little, palm).
- Guide block (static, welded to the table; four parts `block_e`, `block_w`, `block_n`, `block_s`): a 40 x 40 mm block, 60 mm tall, with a 14 x 14 mm square through-bore. The pen stands in the bore with its tip on the table. The bore only keeps the pen upright; it carries no vertical load. The block cannot move.

**Start.** Palm down about 300 mm above the table (xyz ~ (0.45, -0.30, 1.05)), fingers pointing along the default palm yaw (0.7 rad). The guide block and pen are centred near world (0.61, 0.13), table top at z = 0.75; the pen is vertical, its upper end about 140 mm above the table, about 80 mm of it above the top of the block.

**Variation (severity 1).** Block and pen position +-20 mm in x and y (they move together); block yaw about the vertical +-30 deg; pen overall length +-10 %; pen diameter +-10 %; pen mass +-50 %. Severity scales all ranges linearly; severity 0 is nominal. The pen always starts vertical, centred in the bore.

**Success (check.py).** All of:
- **Slide.** While the pen's lower end is within 3 mm of the table and the pen is held (thumb and at least one other finger each pressing on it with more than 0.2 N), the pen travels at least 15 mm up along its own axis relative to the thumb tip. Only travel while those conditions hold counts; travel while the pen is released or off the table does not.
- **Lift out.** At the end the pen's lower end is more than 60 mm above the table (clear of the guide).
- **Held.** At the end the thumb and at least one other finger are each pressing on the pen with more than 0.2 N.

Metrics reported for partial progress: `slip_on_table_mm`, `max_tip_height_mm`, `final_tip_height_mm`, `final_tilt_deg`, `final_held`, final per-finger contact forces, peak total finger force.

**Physical facts.** Newton combines friction by taking the larger of the two contacting coefficients, so pen-fingertip contacts use the fingertip pads' 0.9. Hand contacts carry torsional friction (condim 4, 0.005 m patch), so a two-point pinch is not a free hinge about the pinch axis. Finger servos: kp 3 N m/rad, kv 0.12, torque limited to 0.65 N m per joint. The guide block's walls are 6.5 mm thick and its top is 60 mm above the table; fingers can collide with it.

The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.
