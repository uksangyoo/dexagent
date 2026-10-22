# pen_gravity_pivot at level layered


## Layers used (plan from the earlier session, kept)
- approach/pinch/press: fingertips layer (`move_tips`, `reach` to pick a reachable pinch centre) -- the task
  needs exact pad placement on a 7.5 mm pen relative to its axis, which contact primitives' "away from object
  origin" normals get wrong for an off-centre pinch. Arm: `carry`. Retract press finger: joints (`move_joints` to OPEN).
- pivot: no finger motion at all -- thumb+index pinch with pinch line along palm z is a free hinge (no torsional
  friction on the pen); the press finger releases so the long end rests on the table, then the arm carries the hinge on
  an arc about the resting tip until the pen hangs vertical (table-assisted, quasi-static "gravity" pivot).
- Skill `snap(tag)`: logs lift, tilt, palm-frame pivot vs level-hold reference and per-finger forces.
- Skill `hinge()`: projection of the thumb/index pad midpoint onto the pen axis (world).

## iter_1 (killed mid-run, not counted; 2 of 3 cases finished, both fail)
Level hold 2.6 s at 113 mm lift OK; arc pivot worked (pen vertical, palm pivot 77 deg) in both cases. Failure: pinch
forces fell 0.83 -> 0.63 (after turn) -> 0.19 N (after lifting 120 mm) and a final 8-deg hand roll about palm z
dropped the pen. Change for iter_2: remove the roll, re-squeeze thumb/index along their line (1.2 mm steps until
both >1.1 N) once the pen stands vertical, lift in 4 x 30 mm stages re-squeezing if a force drops below 0.8 N.

## iter_2: 0/3 (sev 0, 0.5, 1 all fail the same way)
Level hold 2.6 s and arc pivot (77-81 deg in palm, pen standing vertical on the table) worked in all three, incl.
sev 1 (yaw -7 deg, length -8 %, mass +31 %). Every extra thumb/index squeeze (1.2 mm or 1 mm) on the vertical pen was
followed by it sliding axially out of the pinch (forces -> 0); in sev 0.5 the second squeeze alone expelled it. So the
squeeze hurts: likely the pads then bear on the rounded cap / domed pad edge and push the pen along its axis.
Learned: the rule only needs the pen centre > start + 50 mm, i.e. ~15-30 mm above "standing on the table".
iter_3 change: no squeeze, one slow 1.5 s lift of 20-50 mm (target lift 68 mm), then hold; snap() now logs the pinch
position along the pen (pinch_s, from pen centre) and its offset from the axis (pinch_d) to locate any axial slip.

## iter_3: 2/3 (sev 0.5 PASS, sev 1 PASS, sev 0 fail)
No squeeze + short slow lift (to lift 67-68 mm) held in 2/3. New diagnostics: the pinch stays ~27-30 mm from the
pen centre (no axial slide), but after the arc the pad midpoint is 6-8 mm off the pen axis (pen sits on the flanks of
the domed pads) and the pen stands leaning ~9 deg. After lift-off it swings as a pendulum (tilt samples oscillate
0-11 deg) in the free hinge, the pinch force walks down 0.63 -> 0.24 N, and in sev 0 it dropped in the bench's 1 s settle.
iter_4 change: closed-loop "stand vertical" step after the arc (skill `vert`: with the tip on the table, carry the palm by
-L*horiz(axis) up to 4 times until tilt < 2 deg) so there is no swing after lift-off; lift 2 s, shorter final hold.

## iter_4: 0/3 (regression)
The closed-loop vert step overshot: one full-gain move took the lean from -7 to +6 deg (about 1 deg per mm of hinge
motion, gain ~1.5 effective), and further steps oscillated with growing amplitude (+-14 deg). Each sideways hand move
also bled pinch force (0.63 -> 0.55 -> 0.37 -> 0.19 N) until the pen fell, still standing on the table. Sev 1 (yaw +23 deg,
psi 0.56 rad, radius +10 %) failed earlier: the pinch weakened during release/retract (0.58 N) and the pen dropped in the arc.
Lesson: every hand motion with the pen in the two-pad pinch costs grip; minimise motions after the turn.
iter_5 change: back to iter_3 plus a single, slow (1 s) vert move at gain 0.6 (skip if lean < 3 deg).

## iter_5: 0/3 (budget exhausted)
The single 0.6-gain vert move worked as intended (sev 0: lean 9 -> 3 deg) but the pinch had already bled to 0.51 N and
the pen slid axially out during the 2 s lift (pinch_s -27 -> -53 mm). Sev 0.5 and sev 1 dropped the pen during the arc:
pinch force after the press-finger retract was only 0.55-0.68 N (sev 1 radius -7 %: the pinch depth is computed from the
nominal radius, so a thinner pen gets less squeeze). Same draw-dependent marginal grip that iter_3 got lucky on.
Main lesson: the two-pad thumb/index hinge needed for the free pivot is too weak/variable (0.5-1.0 N, decays with every
hand motion) to then carry the vertical pen; a force-closed loop on pinch force (or a third finger to re-grasp the standing
pen) is what this policy lacks. FINAL = 3 (best dev result, 2/3).
