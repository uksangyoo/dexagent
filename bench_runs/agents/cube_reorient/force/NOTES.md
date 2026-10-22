# cube_reorient at level force


Plan: two-pad "twirl" gait. Thumb + one opposing finger pinch the cube through its centre (antipodal points on
opposite faces), grip servo holds ~0.7 N, api.rotate sweeps both pads about the palm normal, release, repeat.
Skills written (policy.py): `reach()` cached api.reach on a 3 mm grid; `plan()` picks face pair / contact offset /
opposing finger / pad heights / sweep size from reach probes along the arc; `simulate()` looks ahead to choose the
turn direction; `T()` wraps any motion generator and tracks the cube's unwrapped yaw every step (scorer's method).
No offline kinematics: all geometry is probed at run time through api.reach.

## iter 1: 0/3
Twirl gait with antipodal thumb+finger pinch planned purely from api.reach. Failures: reach ignores collisions, so the
planner picked "hooking" configs where a finger (or the thumb) crosses over the cube to the far face -> 9-20 N contact
forces, cube pressed into the palm (palm force 11 N) and did not turn; grip servo saturated at its 6 mm back-off.
Sev-1 case did reach 87-94 deg (thumb wrapping around the wrist side, index on +y side) but the final joint-space
retreat to home swept the cube off the palm. Learned: open pads (palm mm) thumb (27,127,30), index (14,28,202),
middle (14,5,207), ring (13,-18,199), little (12,-43,166); palm face is at x~23.5 (not 21) so h_est was clipped high;
thumb reaches 110-310 deg around the cube (0=wrist side, 90=thumb side), middle ~160-180, ring ~200-230, little ~270-290.
Force: the servo worked mechanically but its 0.7 N target was meaningless while links collided.

## iter 2: 0/3 (nominal ended resting flat at 73 deg, 17 deg short)
Added a base-to-pad line-crossing heuristic (fingers only), arbitrary finger pairs, non-antipodal offsets, careful
lift-then-home retreat. Retreat now works (nominal ended resting, finger force 0.07 N). But many grasps pressed
the cube into the palm (finger/thumb 12-29 N, palm 15-23 N, servo pinned at its 6 mm back-off): hooking fingers
land on the cube's top edge, so the pinch only drags the cube 0-60% of the commanded sweep; two sev>0 cases were
flung off. Palm camera film: the 40 mm cube sits just below the finger bases; fingers come down on it from above.
Only grasp that turned cleanly in iters 1-2 (palm force ~0): thumb tucked in on the wrist/-y corner (~300-320 deg),
index on the opposite +y/+z corner (~120-130 deg), sweeping the thumb toward -y (direction -1).
Force: the servo cannot fix link collisions (it moves the pad only along the pinch line), so contact choice matters more.

## iter 3: 0/3 (sev 1: yaw 86 deg, flat, height ok, but cube had drifted onto the finger bases: palm force 0.01 N < 0.02)
Restricted twirl grasps to thumb on the wrist/-y corner + index/middle/ring on the opposite corner, one direction,
pads stop 1 mm short and the grip servo closes. Where it planned (sev 1) it worked: grip servo held 0.8 N on both
pads with palm force ~0.1 N, steps turned 35, 42, 19 deg; this is where force control clearly mattered (contact found
and held at a light force regardless of the cube size). But (a) nominal/0.5 found no plan at all (thumb standoff
point out of reach with the cube at palm z 67-73), (b) the twirls walk the cube +30 mm toward the fingers.

## iter 4: 0/3 (budget used)
Added `recenter()` (middle+ring pads push the cube's finger-side face back toward palm z 66 mm, between steps if
z > 84 mm and at the end if z > 78 mm), relaxed the standoff reach check to 8 mm and widened the thumb window to
270-335 deg. All three dev cubes sat at palm z 63-67 and the planner again found no grasp, so nothing moved (the
recenter path was never exercised). Root cause: the thumb cannot get its pad to the wrist-side corner of a cube
centred at palm z >~ 60; the corner pinch that works needs the cube near the wrist (z ~ 56). A fix would be to first
slide the cube toward the wrist with a finger push (same skill as recenter, reversed), then twirl.
FINAL = 4: it is iter 3 (the only near-success: 86 deg, flat, resting height ok) plus fixes that only act in states
where iter 3 fails anyway.

Layers used: fingertips (move_tips: approach, lift, push), contacts (rotate: rigid pad sweep about the palm normal;
release), joints (move_joints: home), force (grip: pinch at 0.7 N during every twirl). The force servo was the right
tool for making and holding contact on cubes of unknown size (iter 3: 0.8 N held, palm force ~0.1 N), but it cannot
help when finger links, not pads, collide with the cube (iters 1-2: 9-29 N, servo pinned at its 6 mm back-off).
