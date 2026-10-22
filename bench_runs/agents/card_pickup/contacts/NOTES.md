# card_pickup at level contacts


## iter 1: 2/3 (sev0 PASS, sev0.5 PASS, sev1 fail)
Plan: palm-down, index pad presses tile centre (grasp w/ squeeze 6 mm -> ~2.7 N), carry palm -y to drag the tile until its centre is 6 mm inside the book's near edge; then droop the hand (R=diag(-1,1,-1): fingers down, palm faces -x, so the index finger lies in a plane parallel to the edge), slide in from -y with thumb/index open +-20 mm around a point 32 mm beyond the tile centre, approach+grasp (normals +-z, squeeze 4 mm -> 3.2 N each), lift 170 mm. Works cleanly when the tile stays on the book.
Failure: api.release() on the pressing finger dragged the tile ~5 mm further (to <1 mm from edge in all cases); with the shortest tile (sev1) it tipped off the edge. Fix: target 9 mm inside, back palm off 3 mm and lift the palm straight up instead of release(), closed-loop re-drag (either direction) until within +-4 mm.
Skills: pads_world/pads_palm = place pads at arbitrary world/palm points by feeding approach() virtual tile-frame points with explicit normals (contacts level has no direct finger move).

## iter 2: 3/3
Change: drag target 9 mm inside the edge, then back palm off 3 mm and lift the palm straight up (no release()); closed-loop re-drag if off by >4 mm (never needed). Tile ended 10.8-11.0 mm inside the edge in all cases (lift-off creep now +1.7 mm, toward safety). Pinch point 33 mm beyond the tile centre along -y (22 mm past the edge); thumb/index forces ~3.2 N; lift 163-170 mm, tile-to-pinch 34 mm. The sev-1 case had a 23 deg yawed tile and passed with no special handling.
Learned: a single pressed index pad (6 mm squeeze -> 2.7 N) drags the tile with no slip or rotation. The drooped-hand pinch (fingers down, palm facing -x) keeps the whole index finger beyond the book edge.
Iter 3: unchanged policy, run to confirm the stop rule.

## iter 3: 3/3 (same policy as iter 2), so the stop rule is met (two consecutive 3/3). FINAL = 3.
Lift 161-170 mm, tile-to-pinch 34 mm, slide 72-90 mm, with tile yaw up to 12 deg and mass up to +42 % in this draw.
Level notes: grasp() with explicit normals did both the one-finger press (drag) and the two-sided thumb-over/index-under pinch on a 3 mm tile in a single call each; carry() with fixed finger targets dragged the tile. Gaps: no direct pad placement for non-contact postures (I worked around it with virtual points in approach()); squeeze() pushes toward the object origin rather than along the contact normal, so it is wrong for pinching a flat tile near its end (I re-called grasp() with a larger squeeze instead); release() moves pads away from the origin, which dragged the tile 5 mm when the pad sat just above the centre.
