# cube_pinch at level joints


## iter 1: 3/3 (turn 44/39/30 deg, offsets 2.5/2/4.8 mm)
- Plan: thumb-index pinch along palm z at the hand.py PINCH_CENTRE (0.08,0.04,0.118); palm yaw matched to cube yaw mod 90; lift 110 mm, turn the wrist 40 deg about the vertical through the cube, place back closed-loop on measured cube xy, open, retreat.
- No IK at this level, so I wrote `finger_ik` (skill): online finger IK in the air — finite-difference Jacobian (4 joint pokes, all fingers at once, measured with api.tips()) + damped-LS Newton steps with Broyden updates; ~10 s sim for open+squeeze postures. Also `settle(q)`: move+hold, return measured pads.
- Learned: open pinch reached 3.9 mm residual (aperture limit ~69 mm pad-to-pad); squeeze posture IK sometimes jumps to another thumb branch (sev1: thumb j2 -2.42). Grip 3.5-5 N. Sev1 cube slipped ~10 deg during the 40 deg turn -> only 30 deg (thin margin).

## iter 2: 3/3 (turn 45/45/35 deg, offsets 2.3/2.7/2.5 mm) -> stop (two consecutive 3/3)
- Changed: squeeze-posture IK now re-estimates the FD Jacobian at the open posture with more damping (lam 0.015, step <= 0.2 rad) -> thumb stays on one branch in all cases (j2 ~ -1.6). Added closed-loop turn top-up (skill): after the 40 deg wrist turn, read cube yaw mod 90 and turn up to 0.4 rad more if it is short of ~34 deg.
- Learned: sev1 (edge 34 mm, yaw +23 deg) still slips ~6 deg during the turn (34 deg) but well above 25; the top-up did not need to fire. Online finger IK costs ~12 s of the 60 s budget; whole run ~27 s sim.
