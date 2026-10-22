# cube_pinch at level contacts


## iter 1: 3/3
Plan: tuck middle/ring/little; align palm yaw to the cube face nearest yaw 0.7; palm 50 mm above grasp, `approach` pre-shapes thumb/index
(points computed for a virtual cube at the pinch centre), `carry` down, `grasp` thumb (wrist-side face) + index (fingertip-side face)
5 mm above centre, squeeze 4 mm; lift 115 mm; turn 35 deg by rotating the WRIST about the vertical through the cube (5 carry steps);
closed-loop place to start xy (2 stages), release, retreat. Half-edge read from resting height (z - 0.75). Result 3/3, offset ~2 mm, turn ~39 deg.
Learned: thumb/index pinch along palm z is solid (2.6 N each while lifted, IK residual 0.1 mm). Rotating the wrist was simpler than `rotate`.
Skills: `face_points(api, obj, h, centre_palm)` -> opposition contact points+normals in object frame from a palm-frame layout (also used
to pre-shape before the palm descends, by placing a virtual cube at the pinch centre); closed-loop placement = carry palm by (p0 - p_cube).

## iter 2: 3/3
Change: approach standoff = clip(0.037 - h - 0.008, 0.004, 0.008) so the pre-shape stays inside the pad aperture for 46 mm cubes
(no effect at nominal). Result 3/3 (turn 32.7 deg on the negative-turn cases vs 38.8 positive: a few deg of rotational lag/slip, still
well above 25). Two consecutive 3/3 -> stop. FINAL = 2.
