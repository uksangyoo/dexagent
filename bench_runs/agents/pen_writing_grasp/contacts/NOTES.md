# pen_writing_grasp at level contacts


## iter 1 (0/3)
Thumb-down sideways hand, 3-point grip (index low/+z, middle high/+z, thumb between/-z) via approach+grasp with explicit normals;
lift along pen axis, straighten pen vertical, closed-loop descent, 50 mm drag with deflection regulation.
Result: extraction works (tip 110 mm, forces T4/I2/M2.4 N); but touchdown rotated pen in grip 7->25 deg in a few steps (arm lag overshoot = hard press),
and during the drag the tip stuck while the pen pivoted in the grip (extent 2-9 mm, lean ->40 deg), pen then dropped. One case dropped pen during straighten/transport.
Learned: pinch grip is a pivot under tip friction (table mu=1); vertical trailing pen is hopeless. Contacts' normals must be given explicitly for a cylinder (default normal = from object origin).
Skill: `contact_points(R, up, Rg, w, lift)` builds object-frame points+normals for fingers at palm-frame sides of a cylinder.

## iter 2 (0/3)
Changed: squeeze 7 mm, slow touchdown, pen tilted 40 deg tip-leading (push stroke along fingers, in the 3-point braced plane) so the table reaction runs along the pen.
Result: grasp great (T5.7/I2.8/M3.2 N) but in all 3 cases the pen was flung away during the 3 s palm pitch to the 40 deg tilt (arm swung through a new configuration). Also found: on lift-off the pen rotates ~14 deg in hand in the braced (palm y-z) plane - the 3-point bending grip is elastically soft (~0.2 N m/rad).
Learned: big palm pitch with the rolled hand is not arm-friendly; the stiff axis of this grip is actually rotation about the pinch normals (torsional friction 0.005 m x ~12 N total = ~0.06 N m).

## iter 3 (0/3)
Changed: no tilt; pen straightened vertical; stroke along the palm-face normal (twist resisted by torsional friction); start spot chosen so the cup is ahead of the fingertips.
Result: pen dropped during straighten/transport in 2/3; sev-1 case was blocked 29 mm high and the pen spun to 64 deg at touchdown. Cause: after lift-off the pen settles ~14 deg tilted in hand (axis_in_palm z=+0.24, deterministic), so fully straightening pitches the fingers up and sinks wrist/thumb into the table.
Budget cut to 4: last iteration -> grip 12 mm higher on the pen, only half-straighten (slerp between level hand and full correction), lighter press (0.15 mm), 10 mm/s stroke.

## iter 4 (0/3) - budget exhausted
Half-straighten + 12 mm higher grip did not help: the pen was still dropped/flung during the 3.5 s straighten+transport carry in all 3 cases (tip ended 140-260 mm away), so the stroke never started.
The transport carry (palm rotation ~10-20 deg about the rolled-hand pitch axis plus ~0.1 m translation) is what loses the pen; the grasp itself (T~5.7/I~2.8/M~3.2 N) and the extraction (tip 105-113 mm) worked in every run of iters 1-4.
FINAL = 1: the only iteration that got the pen to the table and drew contact (ink 211-243 steps, 98-100% contact, lean ~33 deg), even though the tip only moved 2-9 mm because the pen pivoted in the grip.
Next steps if continued: transport without re-orienting the palm (keep R fixed and accept the in-hand tilt), and test the palm-face-normal stroke direction, which was never actually reached.
