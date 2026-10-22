# pen_table_slide at level contacts


## iter 1 (0/3)
Plan: palm-down thumb-index pinch (hand.py PINCH centre) opposed along palm +z, open pads straddle pen above top, descend,
grasp 2 mm squeeze ~16 mm below pen top; lower palm 2 mm chunks with a live copy of the scorer's slide metric (Tracker), then lift 135 mm.
Result: slide works in all 3 (24-25 mm scored), but at ~12 mm lowered both forces dropped to 0 (pen rolled out of the 2-pad pinch sideways —
while sliding axially, friction has no lateral reserve) -> re-grasp put the pen leaning on the bore wall (tilt 8-9 deg, forces 4.4/1.3), and the lift left the pen behind.
Learned: a 2-point pinch is laterally unstable *while slipping*; API squeeze()/release() push toward the object origin (pen centre, 70 mm below), so I avoid them and re-call grasp() with explicit radial normals.
Skills: pen_axis/to_obj/pinch_points (world->object-frame contact points+normals); Tracker (live scorer-equivalent slide metric).

## iter 2 (3/3)
Changed: thumb opposes a V of index+middle (contacts at +-55 deg about the far side of the pen) so the pen sits in a groove and cannot roll out
sideways while slipping; grasp centre palm (0.08, 0.03, 0.118) chosen by api.reach() test (all < 0.2 mm); lift in 10 mm chunks with a lag check (re-grasp tighter if the pen lags).
Result: 3/3; slide 25-26 mm with forces steady 3-5 N per finger throughout (no LOST events), pen lifted 140 mm, final forces 0.7-1.9 N, tilt 7-15 deg (pen swings in the grasp while lifting).
Learned: a 3-contact V-cage fixes the lateral instability; 2 mm squeeze ~1.5-3 N per pad, rises to 3-5 N while sliding (pads dragged).
Skills: frame_uv/pinch_points (thumb vs index/middle V, object-frame points+normals), palm_dirs + reach scan to pick a reachable grasp centre.

## iter 3 (incomplete: 1/1 finished case passed)
Same policy as iter 2 (confirmation run). My shell `timeout 890` wrapper killed iterate.py while it waited for GPU slots: only the severity-1 case
finished (PASS, slide 25.6 mm, tip 141 mm, held); the 0 and 0.5 cases never ran. Harness/operator error, not a policy result. Re-running unchanged as iter 4 without the wrapper.

## iter 4 (3/3)
Unchanged policy from iter 2. 3/3 again: slide 24.7-24.9 mm, final tip 140 mm, held (thumb ~1.6 N, index ~0.7 N, middle ~1.7 N), tilt ~7.5 deg.
Stop rule met (iters 2 and 4 = two consecutive completed 3/3; iter 3 was killed while queued and does not count). FINAL = 4.
