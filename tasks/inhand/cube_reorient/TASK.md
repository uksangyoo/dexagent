# cube_reorient

The hand is held palm-up with a foam cube resting loose on the flat of the palm. Using the fingers, turn the cube 90 deg about the palm normal (either direction), so that it ends lying flat on the palm again, released, with the palm still where it started.

**Objects.** `cube`: a box, nominal edge 40 mm, mass 12 g, friction 0.9. Its pose is readable at any time (`api.object_in_palm("cube")` gives it in the palm frame).

**Start.** The arm starts palm-up with the palm about 230 mm above the table, world palm position ~ (0.48, -0.06, 0.98). The palm is tilted 5 deg so that the fingers point slightly upward (the palm slopes down toward the wrist). Palm frame: +x out of the palm face (up), +y toward the thumb, +z along the fingers. The palm face is at palm x ~ 20 mm over palm z ~ 45-85 mm. The cube starts resting on it, centred near palm-frame (41, 0, 65) mm, one face down on the palm and its side faces square to the palm axes. The fingers start open. Nothing holds the cube but the palm and gravity.

**Variation (severity 1).** Cube position on the palm +-10 mm across the palm (palm y) and +-15 mm along the fingers (palm z); cube yaw about the palm normal +-25 deg; edge 40 mm +-12 % (it rests at the matching height); mass 12 g +-50 %. Severity scales all ranges linearly; severity 0 is nominal.

**Success (check.py).** Measured in the measured palm frame, relative to the cube's own start. All of:
- the cube's net yaw about the palm normal, tracked continuously (unwrapped), is 90 deg +-10 deg in either direction at the end (the cube is 4-fold symmetric, so this is the only way a face swap shows);
- at the end it lies flat: some cube face within ~10 deg of the palm normal (|cos| > 0.985);
- at the end it rests on the palm, released: palm contact present, all five fingers together pressing it with less than 0.5 N (averaged over the last 0.2 s), its centre height above the palm within 5 mm of its resting height at the start, and |palm y| < 45 mm;
- it never leaves the palm (palm-frame x > 90 mm, |y| > 70 mm, or z outside -20..160 mm at any step);
- the palm stays put: at every step within 30 mm and 15 deg of its start pose.

Reported metrics include the signed final yaw, the largest |yaw| reached while flat, flatness (final and minimum), height change, drift on the palm, final finger/palm contact forces, and palm motion. The bench holds the final targets for 1 s before scoring. Sim time limit 60 s.

**Physical facts.**
- Newton combines friction by taking the larger of the two coefficients, so every cube-hand and cube-palm contact has friction 0.9 whatever the cube's value. Contacts also have torsional friction, so a fingertip pressed on a face resists the cube spinning about that contact.
- Finger servos are torque-limited (|tau| <= 0.65 N m per joint); arm servos sag slightly under load, and the palm settles by about 10 mm at the start.
- The palm surface is not a large flat plate: it is flat only near its middle, and the cube can tip or slide off its edges and over the heel.
