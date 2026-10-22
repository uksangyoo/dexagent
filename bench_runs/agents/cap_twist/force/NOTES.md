# cap_twist at level force


## Iter 1: 3/3 (turns 415/417/409 deg, palm spread 1.4 mm, yaw 0.1 deg)
- Strategy: top-press dial turning. Index+middle pads at r=20 mm from the lid axis (+-35 deg about the finger direction) press DOWN on the lid top with the force servo (`grip("jar", {index:2.5, middle:2.5}, directions=(0,0,-1) jar frame)`), then `rotate(... frame="world", centre=lid top)` sweeps them 64 deg CCW about the jar axis; ungrip, lift 25 mm, rotate back in the air, repeat until `object_joint("lid_twist")` >= 380 deg (closed loop on the lid angle). Palm fixed for all strokes. Radius-independent (pads always inside the smallest lid), so lid size/mass/friction variation does not matter.
- Placement chosen at runtime with `api.reach` over a grid of lid-top palm-frame positions (depth D x along-finger zc); feasible = all sweep/lift/press pads < 2.5 mm residual; picked centroid of feasible set (D=90 mm, zc=120 mm, alpha=32 deg). Ring/little straightened (joints) to stay at palm level; thumb far away (143 mm).
- Result: ~57-59 deg per 64 deg stroke (little slip), 7 strokes, done at t=24 s. Measured servo force 2.4-2.6 N, travel 3-4 mm.
- Layers used: joints (idle fingers straight), fingertips (move_tips + reach for placement search and lift/lower), contacts (rotate for the sweep and the return arc), arm (carry), force (grip/ungrip during each sweep). Force control mattered as a convenience: constant ~2.5 N normal load without knowing exact contact depth; a position press would likely also work given the large torque margin (~0.1 N m friction torque vs <=0.03 N m needed).
- Skill: `pads(top, axis, phi, h)` palm-frame pad targets on a rotated dial; `check()` reach-based feasibility of a whole stroke.
## Iter 2: same policy, re-run to confirm (stop rule)
- Result: 3/3 (turns 417/416/415 deg, palm spread <=1.5 mm), incl. lid radius -9 % and thread friction +43 %. Same 7 strokes of ~57 deg; per-stroke slip unchanged by friction, so the torque margin is large. Stop rule met (two consecutive 3/3). FINAL = 2 (identical policy to iter 1).
