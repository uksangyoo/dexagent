# pen_finger_roll at level layered


## iter 1 — 0/3
Layers: joints (pre-shape/tuck middle-ring-little), fingertips (open thumb-index pinch at hand.py PINCH_CENTRE, palm-frame), contacts (grasp with explicit normals; rotate about pen axis for roll; rotate about palm -y through grip point for swing), arm via carry. Closed loop on my own mirror of the scorer (tilt/twist/bottom) built from object_pose/object_in_palm.
Result: grasp (1.7 N per pad), lift (151 mm) and roll (>50 deg within 0.9 s of a 35 deg pad rotation; pen rolls ~1.5x the commanded pad angle) all work on 3/3. Swing dropped the pen in all 3 cases at ~+18 deg tilt (tilt tracked command ~1:1 until loss). Grip force had sagged to 0.5-0.8 N after the roll-back move. Suspect: pen top (25 mm above the pads) hits the thumb links as it tilts, plus weak grip.
Learned: contacts.squeeze/release use the direction to the object ORIGIN, which for a long pen is along its axis -> unusable; pass explicit normals to grasp and squeeze along the pinch line myself.

## iter 2 — killed mid-run (no summary; not counted). 2 cases finished, both failed
Change was a pinch-line regrip (move_tips along the thumb-index line) before the swing. After the roll + straight-line roll-back the grip was already 0.43 N and the pen had slid ~50 mm down along its axis through the pads; the regrip squeezes pushed forces to 0 and the pen dropped before swinging. Lesson: the roll-back (straight move_tips back to saved targets) is what ruins the grip; never regrip a pen that has already slid.

## iter 3 plan (2nd counted)
Reorder: swing FIRST with the fresh post-lift grip (regrip to 1.35 N only now), closed-loop contacts.rotate about palm -y at 6 deg/s stopping on measured tilt (+16.5, then -16.5, then back to |tilt|<1.5 and fine servo until my scorer mirror says settled), THEN roll by orbiting the pads about the pen axis and simply hold (no roll-back). Grip point moved from 13 to 10 mm below the pen top: the pen top sticking out between the thumb/index distal links may pry the pinch open as it tilts (my guess for the ~18 deg loss in iter 1).
Layers: joints (pre-shape/tuck), fingertips (open pinch, regrip along pinch line), contacts (grasp with explicit normals, rotate for swing and roll), arm (carry).

## iter 3 (2nd counted) — 0/3
Swing-first with fresh grip, grip 10 mm below top. Pen dropped during the very first swing in all 3 cases: as the pads rotated about palm -y the pinch force fell monotonically (1.4 -> 0 N over ~10 deg of pad rotation) while the pen tilt lagged (dtilt ~0-5 deg for ~10 deg of pad rotation). Correction to the iter-1 notes: iter 1 actually gripped 25 mm below the top and its swing tracked 1:1 to ~18 deg even with a weak grip, so gripping near the top was the wrong move: the pen section above the pads, lying between the thumb/index distal links, is what couples the pen to the finger rotation.
Learned: commanded rigid rotation of a two-pad pinch about an axis normal to the pinch plane unloads the pinch (fingers do not track the rotated targets in force), so the squeeze has to be servoed during the motion.

## iter 4 plan (3rd counted)
Grip 28 mm below the top; swing as 2-deg contacts.rotate increments (0.3 s each) with a force servo between increments (skill `keep_grip`: if a pad force < 1.2 N, close the pinch 0.8 mm per finger along the commanded thumb-index line via move_tips, capped at 6 mm total). Same closed loop on measured tilt; roll last, no roll-back. Trace now logs measured thumb/index pad x,z.

## iter 4 (3rd counted) — 0/3
Grip 28 mm below top + 2-deg increments + force servo (squeeze when < 1.2 N). Tilt now tracks the command ~1:1 (8-17 deg reached; sev 0.5 reached the +16.5 goal), but the pinch force then bleeds to 0 within ~0.4 s, even while holding still at +16 deg, and the pen drops. Measured pads do not move (x,z within 1 mm) while the pen drifts ~5 mm in palm -y as it tilts: it is squirting sideways out of a convex-pad-on-cylinder pinch (laterally unstable, only friction holds it). The squeeze servo made it worse: squeezing an off-centre pinch pushes the pen out faster (same as the iter-2 regrip).
Learned: the swing limit is lateral stability of the two-pad pinch, not grip force; force alone is the wrong feedback signal.

## iter 5 plan (4th counted, last)
New skill `centre_pinch`: measure the pen axis at grip height vs the measured pad midpoint, take the offset normal to both the pinch line and the pen axis, and shift both pad targets by it (move_tips, <=1.5 mm per call, <=6 mm total). Called after every 2-deg swing increment, during all holds of the swing/settle phases (`hold_servo`), and before the post-lift regrip; squeezing is only allowed when the pinch is centred (<1 mm) and threshold lowered to 1.0 N, cap 4 mm.

## iter 5 (4th counted, last) — 0/3
centre_pinch backfired: right after lift the measured "offset" (pen axis vs pad-centre midpoint, normal to pinch line) is already 3.5-4.5 mm and stable, so the pad-centre midpoint is not where the contacts are (the pads touch on their sides). Chasing it dragged pen and pads together: offset grew 4 -> 7 mm while the shift ran into its 6 mm cap within ~1 s, so it was positive feedback. Even so, all 3 cases reached +16.3..16.8 deg (best swing so far) before the pinch bled out. sev 0.5 then dropped the pen 10 deg into the -swing.
Learned: a lateral-offset servo needs the contact location (not available, only pad centres), or a reference taken at a known-good moment (servo the change of offset vs the post-lift value, not its absolute value). Next things to try: an offset servo relative to the post-lift reference; a third finger (middle) backing the pen on the -y side during the swing; a smaller swing (+-15.5) with an immediate reversal and no dwell at the extremes.
FINAL = 5 (all iterations 0/3; iter 5 got furthest on the swing in all 3 cases).
