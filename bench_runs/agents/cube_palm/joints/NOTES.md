# cube_palm at level joints


## iter 1 (lost)
Policy v1: in-sim joint IK skill (`solve`: servo measured pads via api.tips onto palm-frame targets, FD Jacobian + Broyden, joint-limit masking), random posture sweep for FK data, thumb-index pinch at palm (80,35,118)mm, roll palm-up about the cube, slide pinch toward palm until palm force, open, carry, re-close, roll back, place.
Result: none - my shell `timeout 890` killed the run while it waited ~13 min for a GPU slot; iteration counted with no results. Lesson: run iterate in background without an outer timeout.

## iter 2: 0/3 (first completed iteration)
Same v1 policy. IK skill works (thumb/index pads within 0.2 mm in the air). Failure: the IK index took a hyperextended branch, so its proximal link sat over the cube; at pre-grasp the index pushed the cube with 14 N and the cube slipped out on lift (max lift ~18 mm). Transfer pinch c2=(48,30,85) mm was unreachable (residuals 18/26 mm): the index cannot reach near the palm at z<0.13.
Learned: limits thumb j2 [-2.67,0], index j2 MCP [0,2.2]; index MCP ~1.4 rad points the finger straight down with the pad at x~0.10-0.11, z~0.09-0.12; straight middle pads sit at x=15 mm (palm face ~0.02-0.025). The closeup camera barely shows the hand.

## iter 3 plan (v2)
Index held at a fixed posture (MCP 1.45) as a vertical "wall" on the cube's fingertip-side face; only the thumb uses IK (seeded from iter-2 FK samples hard-coded in THUMB_SAMPLES). Pick: descend 12 mm off the wall, slide the palm so the wall touches, thumb squeezes. Palm up: the thumb slides the cube down the wall onto the palm (x 0.10 -> 0.045), backs off 6 mm, carry. Re-squeeze, roll back, set the index tip 18 mm above the table, and the thumb slides the cube down the wall onto the table.
Result: killed mid-run by the restart (does not count). Its one finished case (sev 0.5) is still informative: pinch pick + lift worked (dz 166 mm, thumb/index 1.2 N), but the cube fell out during the 180 deg roll about palm z. Thumb IK for the low "on palm" positions failed (res 30 mm): the thumb cannot reach x<60 mm at z<70 mm.

## iter 4 plan (v3)
Diagnosis: a roll about palm z (= the pinch axis) loads the pinch purely in shear and twists it. New turn-over path keeps palm y horizontal so gravity is always taken by the wall/shelf or palm: pitch fingers-down (cube on index/middle shelf, thumb on top), spin 180 deg about the vertical wrist-3 axis, pitch up. Thumb lets go at 30 deg; pitching to 60 deg (palm up, fingertips 30 deg down, palm_up 0.87) the cube slides along the wall into the palm/wall corner. Fingers 1-4 all at the wall posture (so nothing sticks out past the pads in fingers-down). Return: pitch to fingers-down, spin back, lower the shelf to ~table+20 mm and pitch palm-down about the outermost pad so the cube slides off onto the table against the wall at the start spot, retreat 6 mm, lift, open.
Skill added: `pivot(api, R0, R1, piv, W0, W1, n, seconds)` turns the palm by slerp while keeping a palm-frame point on a world line.

## iter 4: 0/3 (all three: cube flung out during the spin; sev 0 and 0.5 then crashed on set-down reach)
v3 result: pick + lift solid (thumb/index 1.5-2.8 N), pitch to fingers-down holds the cube (1.1-1.8 N), but the 180 deg spin about the vertical wrist axis flings the cube out in both cases (film: the cube turns/slides out of the pinch while the hand spins; palm z is again the pinch axis). Also the set-down pre-pose (fingers down, palm ~0.1 m beyond the cube, z 0.96) was out of arm reach (IK residual 4.2 mm at 0.72 m from base) -> crash in the nominal case.
Learned: never rotate about palm z while pinching along palm z, whatever the gravity direction. Little finger at [0.2,1.45,.25,.15] splays out to y=-128 mm (its j2 is not like the other MCPs) - harmless.

## iter 5 plan (v4)
Turn over by ONE pitch about palm y only: Rd@Ry(psi), psi 0 -> 90 (fingers down) -> 120 (thumb lets go) -> 150 deg (palm up, fingers pointing back toward the robot, tilted 30 deg down, palm_up 0.87). No spin at all. Return: psi back to 90 (cube on the shelf against the palm), then set down as before but with the landing spot shifted 30 mm toward the robot base (fallbacks 60/90 mm) for reach; approach the set-down pose directly at low height.
Skill added: `try_move` = api.move that catches arm-IK failure, re-anchors at the measured palm and returns False, so the policy can try a fallback pose instead of crashing.

## iter 5: 2/3 (sev 1 PASS, sev 0.5 PASS, sev 0 fail)
v4 works: the single pitch keeps the pinch (thumb/index ~1 N at fingers-down and at 120 deg), the cube drops along the wall into the palm corner, palm-only carry 3.0-3.2 s at palm_up 0.87 (palm 0.06-0.14 N, index wall 0.03-0.09 N). Set-down by pitching palm-down about the pads lands the cube flat; reach OK with the 30 mm shift. Final offsets 66/68 mm (limit 80): the cube tips over once when it leaves the shelf and ends ~0.8-1.0 edge from the wall toward -d, plus the 30 mm shift.
Nominal failure: letting the thumb go at 120 deg dropped the cube off the shelf tip (cube sat at palm x 110, deeper than the 103-104 of the passing cases; thumb's IK "pre" pose was on another branch, so its retreat swept the cube).

## iter 6 plan (v5, last budgeted iteration)
(1) Keep the pinch all the way to the carry pose (150 deg), then back the thumb 15 mm straight off the face (new small-step IK pose q_r, same branch as the grasp) and open it; gravity (0.87 g toward the palm) takes the cube down the wall. (2) Aim the set-down 0.85 edge further along +d to cancel the tip-over drift. (3) Start the set-down pivot at 70 deg instead of 90 (palm 42 mm nearer the robot) for reach; fallbacks back off 2/4/7 cm toward the base.

## iter 6: 1/3 (sev 1 PASS with offset 44 mm; sev 0 and 0.5 fail)
The set-down changes worked: the 70 deg start reached with no back-off in all three cases, and the tip-over compensation brought the passing offset from 66 to 44 mm. The thumb release broke it: the small-step "back off 12 mm" IK pose q_r jumped to another thumb branch (j1 -1.4 vs -0.3 at the grasp, residual 13-20 mm), so the thumb swept or flicked the cube out of the hand at the carry pose in 2 of 3 cases. The nominal case failed in iter 5 for the same root cause (thumb pose on another branch). Lesson: the thumb's IK is multi-branched and its residuals are ~10 mm. The release should be a joint-space move from q_g that I know is monotone (e.g. open only the thumb's distal flexion joints), not a new IK target. The palm transfer itself (pinch -> pitch -> drop along the wall) is reliable whenever the release is clean.
Budget used (iters 2, 4, 5, 6). FINAL = 5 (2/3; its release via the "pre" pose at 120 deg worked whenever that pose was on the grasp's branch). Workdir policy.py restored to the iter-5 version; v5 is kept in scratch/policy_v5_iter6.py.

Skills summary: `solve` (in-sim FD-Jacobian + Broyden finger IK on measured pads), `pivot` (slerp the palm about a palm-frame point on a world line), `try_move` (arm-IK-failure recovery + fallback), `on_palm_ok` (mirror of the scorer's carry test for live feedback).
