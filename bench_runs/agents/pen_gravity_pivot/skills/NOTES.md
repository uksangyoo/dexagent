# pen_gravity_pivot, phase 3 (force level + cross-agent memory, 6 iterations)

## Plan (before iter 1)
Reused: the layered agent's iter_3 policy (pen_gravity_pivot/layered: 2/3 dev, 3/6 held-out) for phases 1-4,
nearly verbatim (skills `snap`, `hinge`, reach-scan for the pinch centre, middle press on the stub, level lift,
lower beside the rest, release the press, arc the thumb-index hinge over the resting tip until the pen stands on
the table). In all 9 of its scored cases the pivot reached ~90 deg; every failure came after, when the two-pad hinge
bled grip (0.8 -> 0.2 N) while the vertical pen was lifted and held.
New: (1) pen radius estimated from its rest height (pinch depth uses it; layered iter 5 lost thin pens);
(2) while the pen stands on its tip, re-grasp it into a 3-contact V-cage: thumb vs index+middle at +-55 deg
(geometry from pen_table_slide/contacts `pinch_points`, which lifted a vertical pen 3/3), middle comes in first;
(3) force layer: thumb grip servo (1.0 N, object-frame direction refreshed from the palm frame every chunk, as in
pen_gravity_pivot/force `regrip`) pushes the pen into the V; (4) lift ~40 mm in 8 mm steps with a lag check.

## iter 1: 0/3 (no level hold in any case)
- The reused press left the long part 17 deg UP in all 3 cases (layered had 5-6 deg with the same code; only my
  0.05 mm radius correction differs, so the press equilibrium is very sensitive) -> level_hold 0 s. Arc still turned the
  pen to standing (piv ~66-75 deg vs the lift-phase reference).
- V-cage regrasp failed: after the arc the pen sits ~6 mm toward palm +y of the thumb/index line (squirted to the thumb
  side, all 3 cases); moving the index to the +y side dragged the pen over, thumb ended beside it, middle never touched
  (also the middle pre-move pushed the index: thumb/index force jumped 0.7 -> 1.8 N). Pen dropped on the lift.
- Changes for iter 2: skill `level()` closed-loop press correction (move press pad along its normal by 0.8*rs*tan(elev),
  up to 4x) before the hold; skill `cage()` puts contacts in the gaps instead: index stays, middle comes in on the -y
  side 18 mm lower on the pen (reach scan, waypoint beyond the index tip), thumb slides around the pen from -z to the
  +y/-z diagonal at light squeeze, then all three firmed 2 mm; thumb grip servo 1 N toward the axis.
- The "reference" camera is blocked by the arm; use overview.

## iter 2: 0/3 (level hold now 2.1-2.2 s in 3/3; max palm pivot 90 in 3/3)
- `level()` fixed the hold (one step took 17-20 deg -> 0.3-0.9 deg).
- sev 0.5: hinge pinch bled 0.79 -> 0.58 N during release/retract and the pen dropped in the arc.
- sev 0, 1: cage failed. The middle on the -y side could not reach the pen (stopped 24 mm from the axis): the thumb sits
  on that side (-150 deg) because the pen is offset +y; the middle pushing past opened the thumb/index pinch
  (0.95 -> 0.25 N) and the pen fell during the thumb slide. Lesson: after the arc the only open side is +y/+z, between
  the pinch finger and the thumb; only a finger on the thumb side of the pinch finger can close it.
- Change for iter 3 (skill `cage()` v3): pinch with thumb + MIDDLE, ring presses the stub, index parked high; after the
  arc the index comes down on the +y/+z side (phi = middle + ~85 deg, 14 mm lower on the pen, reach scan) - the natural
  gap. Force layer phase A: grip thumb+middle hinge at 0.8 N (max travel 3 mm) from lowering through the arc.

## iter 3: 3/3 (sev 0, 0.5, 1 all PASS)
- thumb+middle hinge + ring press + `level()`: level hold 2.2 s, pivot 90 deg in all. Hinge grip servo (phase A) held
  0.79-0.80 N through release and arc in all 3 (iter_2's position pinch had bled to 0.58 N).
- After the arc the pen again sat ~5 mm toward +y (thumb -155..-161 deg, middle -22..-27 deg); the index came down at
  phi 58-63 deg, 14 mm lower (reach 0.1 mm), touched at 0.3-0.7 N, the cage firm-up (2 mm) spiked forces to ~3 N and
  briefly tipped the pen to 22 deg, then the thumb servo (1.0 N target) backed off to its -6 mm limit and the pen settled
  at 10-13 deg from vertical; lifted to +82 mm, 3 pads within 20 mm, final palm pivot 77-80 deg.
- Margins: tilt 10-13 deg (rule 20), pivot 77-80 (rule 60), lift 82 (rule 50). Re-running unchanged for the stop rule.

## iter 4: 2/3 (same policy as iter 3; sev 0 and 1 PASS, sev 0.5 fail)
- sev 0.5 (yaw -7 deg, mass -20 %): the hinge pinch bled 0.82 -> 0.51 N while the hand LOWERED beside the rest (before
  the servo started), the pen tilted 10 deg in the pinch and squirted (pinch_d 3 -> 8 mm); the servo then hit its 3 mm
  cap at 0.7 N and the pen dropped in the arc.
- Passing cases ended at 13.8-14.9 deg from vertical (rule 20): the all-three 2 mm firm-up tips the standing pen from
  ~2 deg to ~20 deg and it only partly comes back.
- Changes for iter 5: hinge servo starts right after the level hold (before lowering), max travel 5 mm, and keeps running
  while the index comes in; drop the position firm-up, the thumb servo (0.9 N toward the axis) alone closes the cage.

## iter 5: 2/3 (sev 0, 1 PASS with the pen 2.3-2.6 deg from vertical, palm pivot 86-88; sev 0.5 fail)
- Dropping the position firm-up worked: the index touches, the thumb servo (0.9 N) closes the cage, the pen stays at
  ~2 deg from vertical through the lift and hold (was 11-15 deg). Forces at the end: thumb 0.9, index 0.5-0.6, middle 0.75.
- The hinge servo held 0.8 N through lowering/release/retract in all 3, but sev 0.5 (yaw +11 deg) still lost the pen at
  the END of the open-loop arc (film: pen leaning ~20 deg, then gone; servo at its 5 mm cap with 0 N). This is the third
  arc drop in 12 dev cases with the same open-loop circle about a tip point estimated once.
- Change for iter 6 (last): skill `arc_cl()` - closed-loop arc that re-measures the pen's lower end every step and moves
  the commanded hinge on the circle about the current tip at the current radius, stops at 88 deg; logs a compact trace.

## iter 6: 3/3 (last iteration; FINAL = 6)
- Closed-loop arc: in all 3 the measured tip never moved (trace: rho and tip height constant), so it acted like the
  open-loop arc but stopped at 88 deg about the measured tip; the pen ended the arc 0.6-3 deg from vertical (was ~8)
  and the hinge servo held 0.79-0.80 N throughout. Whether it fixes the 1-in-4 arc drop is not shown by 3 cases.
- Cage + thumb servo as iter 5; final tilt 5.9 / 8.2 / 12.0 deg (sev 0 / 0.5 / 1), lift +82 mm, palm pivot 79-84 deg,
  3 pads on the pen. Dev: 1: 0/3, 2: 0/3, 3: 3/3, 4: 2/3 (iter-3 policy rerun), 5: 2/3, 6: 3/3.

## Summary
- Reused helpers: pen_gravity_pivot/layered iter_3 policy almost verbatim for grasp/press/lift/lower/release/arc
  (`snap`, `hinge`, reach scan for the pinch centre, arc-about-tip); V-cage idea and +-55 deg geometry from
  pen_table_slide/contacts `pinch_points`; grip-direction refresh from the palm frame from pen_gravity_pivot/force
  `regrip`; mock api pattern from pen_pickup_pivot/force scratch/mock3.py.
- New skills: `level()` (closed-loop press levelling), `cage()` (third finger into the measured open gap of a 2-pad
  hinge on a standing pen), `regrip()` thumb servo toward the pen axis, `arc_cl()` (closed-loop arc), radius from rest height.
- Force layer: phase A grip on the thumb+middle hinge (0.8 N, 5 mm travel) from the level hold through the arc - stopped the
  pinch bleed that dropped pens (0.58 N in iter 2); phase B thumb-only servo (0.9 N, object-frame direction refreshed every
  ~0.2 s) closes and keeps the 3-contact cage; it replaced a position firm-up that tipped the pen to 11-15 deg.
