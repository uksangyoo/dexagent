# pen_finger_roll at level joints


## iter 1 (0/3)
Thumb-index pinch along palm z at palm (80,40,118), pen along palm x (vertical). Finger IK built online (skill `probe`: numeric
3x4 Jacobian of api.tips() per finger from 0.06 rad joint probes; skill `ik`: DLS servo with Broyden updates; skill `solve`:
linearised pad targets -> q around the probed config). IK converged to 0.2 mm; grasp (F~2.2 N) + lift 120 mm worked on all 3 cases.
Swing by opposite axial pad translation (rolling about palm y): gain is ~d/17mm, not d/r (pads are convex, ~rho+r lever);
reached +15 deg at d=4.6 mm then the pen dropped (grip only ~1.1 N after lift). My wrong-sign detector misfired (bug).

## iter 2 (0/3)
Warm-started the IK from iter-1's converged posture (q_pc) -> setup 8 s shorter. Squeeze 5.5 mm -> 3.2 N in cup, but only
~1.7 N after lift and the pen sat ~3 mm off the pinch line in -y (two convex pads on a thin cylinder don't locate it in y).
Added index distal pitch (null-space of the probed index Jacobian) to make the swing a rigid pinch rotation: worse, the pen
twisted (-58 deg) and squirted out at only ~11 deg tilt. Lesson: a 2-pad pinch has no y stability; swing commands leak into roll.

## iter 3 (0/3)
Switched to a tripod: thumb pad on -z, index/middle pads on +z at y +-9 mm (V locates pen in y). Middle-finger IK converged
online (0.1 mm). Grasp much better (F 5.9/4.9/3.8 N in cup; ~2.4/0.7/2.6 after lift). Swing now reaches +12..13 deg (d clip 8 mm,
pos side saturates) and -16.6 deg, but tilt vs pad differential is a stick-slip curve (deadband ~4-5 mm, then jumps), and
the negative swing coupled ~-30 deg twist that the slow twist loop could not cancel; pen slipped out right after -16.6 (-> -22, drop).

## iter 4 (0/3) - budget cut to 4, this is the last
Pad-differential limit 11 mm, d rate-limited to 3 mm/s, tilt target ramp 12.5 deg/s, a P+I twist loop, squeeze 6.5 mm, and
range-based switching (neg phase ends when hi - tilt >= 31.5). Both sides were reached on all 3 cases (+15.9..16.8 / -26.6..-34,
range 42-50). But the negative side is a slip: tilt jumps from -8 to -25 in ~0.4 s at d ~ -10 mm, and the pen stays stuck there.
On the way back the grip unloads (F 6.5 -> 0.7 N) and the pen drops ~4 s later, so the swing never returned to rest and the roll
never ran. Remaining problem: tilt vs pad differential is hysteretic stick-slip; it needs force-aware squeeze regulation
(re-squeeze along the pad normals from measured contact forces) and a return path that does not unload the pads.
Skills: probe/ik/solve (online finger Jacobian IK from api.tips), Mon (mirror of the scorer from api observations),
stepc (closed-loop tilt via axial pad differential + twist via tangential pad rolling).
