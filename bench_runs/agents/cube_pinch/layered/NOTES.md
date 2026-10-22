# cube_pinch at level layered


## Iteration 1: 3/3 (sev 0, 0.5, 1 all pass)
Plan: palm yaw snapped to the cube face nearest default yaw; preshape thumb/index pads +-33 mm about the hand's measured pinch centre (constant copied from dexagent/hand.py) with the fingertip layer (move_tips) while the arm moves above the cube; carry down; contact-layer grasp on the two faces normal to the finger direction (assumed half-edge 18 mm, squeeze 2 mm, so smaller/larger cubes are just more/less penetration on the compliant fingers); force check + squeeze loop; 30 mm test lift with retry; arm lift to 110 mm; arm turns the palm 0.65 rad about the vertical through the cube in 4 chords; lower with measured xy correction; contact-layer release; back off.
Result: lift 112 mm, turn 35-40 deg, final offset 0.5-1.4 mm, flat. Grip forces 1.4-3.8 N with no extra squeeze needed. Lesson: the weak fingers are fine for a 12 g cube; turning with the arm (not in-hand) keeps the grip trivially stable.
Layers: fingertips (preshape), contacts (open, tuck, grasp, squeeze-if-needed, release), arm (carry). No joints or synergies were needed.
Skills: grasp_points(api, theta) picks the object-frame face points normal to the palm finger axis; Rz-chord turn of the palm about a world-vertical axis through the object.

## Iteration 2: 3/3 (stop rule met: 3/3 on two consecutive iterations)
Change: preshape aperture is now chosen at runtime per finger as the widest of 40/38/36/34 mm half-aperture that api.reach() solves within 1.5 mm (IK check only, no motion), because at +-33 mm a 46 mm cube left only ~2 mm of clearance on the way down. Both fingers reach 40 mm in every case.
Result: lift 112-114 mm, turn 34-41 deg, final offset 1.1-2.2 mm, flat; grip 2.7-5.0 N. Edges seen so far 0.88-1.05 of nominal.
Skill: widest_preshape: probe api.reach over candidate pad targets and take the widest clean one.
FINAL = 2.
