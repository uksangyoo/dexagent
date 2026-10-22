# scissors at level joints


## iter 1: 0/3 (all: never got off the table)
- Plan: thumb behind loop A back wall + index inside hole A pinching that wall along palm z; ring in hole B to open/close by abduction. Home-made in-air fingertip IK (finite-difference Jacobian from api.tips() + Broyden) converged to <=2 mm in ~6 s; hover xy correction from tips_world gave <=2 mm placement.
- Squeeze reached 8 N per pad but the lift failed: a pinch across one 5 mm wall is a point grip, so the scissors pitch/slip out (thumb pad only gets down to ~13 mm, wall top ~12 mm).
- Learned: pivot frame rests 6.8*s mm above the table, so s = dz/6.8mm matches the drawn size exactly. Ring abduction (q[12], range -0.21..0.84) sweeps its tip 53 mm in palm -y (dy/dq ~ -0.048 m/rad): lateral range is plenty. Index dy/dq(abd) ~ -0.052.
- Skill: `ik(api, targets, q0, lo, hi)` in-air fingertip IK from measured tips; `sweep(api, q_goal, sec, cond)` one-call ramp that stops on a sensor condition (hinge angle).

## iter 2: 0/3 (sev0 lifted 220 mm, clearance 171 mm, then dropped)
- Changed to a closing pinch: index in hole A + ring in hole B press the inner walls toward each other (abduction, about 6 N each), so the closed stop makes the scissors one rigid body. Lift works: the scissors hang at about 35 deg pitch from the pinch line. Then I curled the distal joints to form hooks, and that dropped them. Once the ring leaves the inner wall to open, nothing holds part A.
- sev0.5 and sev1 (yaw about 0.8): the hand ended about 60 mm high and 25 mm off at hover, so the pinch missed. Cause unclear (arm tracking or the hand landing on something); next iteration adds a measured-palm correction loop and keeps the little finger at OPEN instead of tucked (the tucked little hangs about 78 mm below the palm).
- Next: grasp further back (pads at palm z 120) so the thumb can reach. In the air, thumb + index pinch A's outer wall and hold part A while the ring cycles B.

## iter 3: 0/3 (sev1: lifted 233 mm, thumb+index pinched A in the air, one opening to 7.9 deg, then dropped while closing)
- Grasped further back (pads at palm z 120) so the thumb can reach. In sev1 the whole chain worked up to the cycles: closing pinch 11 N, lift, thumb IK from the measured scissors pose (2 mm error), thumb on A's outer wall at 7 N, index moved across hole A. The ring opened only 7.9 deg, then the close sweep (ring abduction to its joint limit, up to 25 N) shoved the scissors out of the thumb pinch.
- In sev0/0.5 the ring's probed Jacobian was wrong (dy/dq_abd +0.03 instead of -0.05; the OPEN little finger probably got in the way), so the squeeze went the wrong way and nothing was lifted. Lesson: in-air finite-difference probing is only as good as the free space around the finger.
- Fixes for iter 4: little spread and lifted out of the way; prior abduction gain whenever the probed one is implausible; opening uses the full ring Jacobian (MCP + abduction) for more lateral range; closing only goes back to the squeeze posture, with hinge feedback, instead of the joint limit.

## iter 4: 0/3 (lifted in all 3 cases, 15-20 s in the air; sev1 opened to 25 deg but never re-closed)
- The closing-pinch lift is now reliable: 12-13 N per pad, pivot about 230 mm up, clearance 180-200 mm in all three cases.
- sev0/0.5: when the index moved across hole A onto the outer wall it lost contact (F_index 0). Part A then hung from the thumb alone and the ring sweeps did nothing (hinge stayed at 0).
- sev1: thumb 8 N + index 8 N held part A, and the ring opened the blades 9 -> 16 -> 25 deg over three sweeps. The close sweep (back to the squeeze posture) never brought the hinge below 3 deg, so no cycle counted. The ring's in-air IK and Jacobian were poor again (7-10 mm residual, dy/dq_abd +0.006), so the closing target was not on the inner-wall side.
- What would be next: close by commanding the ring toward its abduction limit with a force cap (stop on contact force, not just hinge); move the index across hole A with tips feedback; calibrate the ring away from the other fingers.
FINAL = 4 (all dev iterations 0/3; iter 4 got furthest: lift in 3/3 cases and an opening of 25 deg in the air).
