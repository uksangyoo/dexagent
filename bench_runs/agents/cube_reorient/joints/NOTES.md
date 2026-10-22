# cube_reorient at level joints


## iter 1 (probe; run was killed by my own shell timeout after 2/3 cases, no score)
- Policy only swept each finger joint (all fingers at once) + 10 random postures, logging (api.q(), api.tips()) pairs (1404 samples). Cube barely moved (yaw 2.5 deg).
- Offline (scratchpad, numpy/scipy only, on api data) fitted a product-of-exponentials model per finger: 4 screw axes + pad point; rms 0.006 mm. Found: index/middle/ring MCP at palm z~105-113, abduction axes along palm x at z~80-90 (so abduction barely moves a curled tip); thumb joint0 rotates about palm normal through (y18,z30), joint1 about palm z.
- Reach analysis: at cube mid-height only the thumb reaches all sides of the cube (+y face, far face, -y face); index/middle/ring reach the far face only hooked and ring reaches the -y face only for some poses.
- SKILL fk(i,q)/IK.solve: PoE FK + least-squares pad IK with posture term (in policy.py). SKILL line(): straight palm-frame pad path, IK per 20 ms step.

## iter 2: 0/3 (yaw 27, 18, 16 deg)
- Thumb-only stroke pushing: pad on the most-+y face, 0.4h toward +z from its middle, pushed along -n with my IK; closed loop on tracked yaw; 14 strokes.
- First stroke rotated 11 deg but drifted the cube 7 mm to -y, where it hit the little finger's base (forces on finger 4). That contact sits on the far half of the -y face, so it makes the opposite torque: later strokes gained ~5 deg and lost ~3-4 deg when the thumb backed off (spring-back). Stalled.
- Learned: thumb can not reach the -y face (its links would lie over the cube; checked with a thumb skeleton built from the fitted axes) and can not reach the near (-z) half of the +y face; it can push the +y face far half and the far face. Ring/little can not reach beside the cube's -y face (abduction limits).

## iter 3: 1/3 (sev1 PASS yaw -85.7; sev0 stalled at 32 deg; sev0.5 pushed off over the heel)
- Changes: little finger spread (abduction 0.6) to remove the wall; strokes may use any face looking +y or toward the fingers (planner scores IK feasibility, predicted drift vs a box on the palm, transit length); thumb transits orbit the cube.
- Wall is not the little finger itself but its knuckle (abduction axis at palm y~-41 near the palm): thumb and finger 4 squeeze the cube at 1.3-2 N and the yaw stalls at 32 deg. In sev1 the cube's corner slipped past the knuckle at ~33 deg, then strokes gave 8-20 deg each and it ended at 85.7 deg, at palm (y -29, z 30) - still resting flat.
- sev0.5: the first planned face was the far face; the transit knocked the cube and far-face pushes walked it to z=16 off the heel (tipped).
- SKILL thumb_skeleton/thumb_hits: knuckle points on the fitted thumb axes, carried through the PoE chain; rejects pushes whose thumb links would sit over/in the cube.

## iter 4 (killed mid-run by the pause; not counted): sev0.5 PASS (yaw -87), sev1 77 deg (no feasible push left), sev0 not run
- Changes vs iter 3: drag along the face tangent, thumb_hits link check, stall penalty (switch faces after a stroke gains < 3 deg).
- Both cases stalled at ~30-38 deg on faces looking toward the fingers (thumb+little-knuckle squeeze 2-3 N), then switched to the wrist-side face turned toward +y (n ~ (y .78, z -.62)): 18-21 deg per stroke with no finger-4 force, cube walking to z ~ 96-100 (still flat on the palm).
- Post-pause edit before iter 5: relax pass when no push is feasible (ylo -46 mm, zlo 30 mm, fmin 0.1).

## iter 5: 3/3 (yaw -86.4, -88.6, -85.8)
- Changes: stall penalty 3 -> 8 per stalled stroke (switch faces after one stalled stroke, not three); BOX_HI z 84 -> 92 mm plus hard pred z < 106 mm (the cube rests fine at palm z ~100); +1.5*max(0,-n_z) bonus for wrist-side faces; the relax pass from the pause.
- sev0/0.5: 4 strokes on +y/finger-ward faces up to ~50 deg, one stall (thumb+knuckle squeeze 1.5-1.8 N), then the wrist-side face (y .76, z -.65) gave 15 deg per stroke to 86-89 deg. sev1 (cube started at z 59) went the other way: finger-ward faces walked it toward the heel (z 59 -> 31) with no stall, 5 strokes. Offline check (planner only, no sim): the thumb reaches the wrist-side face only once the cube is >= ~37 deg and sits far toward the fingers (z ~90), so the bonus matters only late.
- Running iter 6 unchanged to confirm (stop rule: two 3/3 in a row).

## iter 6: 2/3 (same policy as iter 5; sev0 -86.0 PASS, sev0.5 -88.6 PASS, sev1 -77.1 fail)
- sev1 cube started at z 58 (toward the heel): finger-ward faces gave 16-23 deg per stroke with no stall, but walked the cube to z 30, and at 77 deg no push passed the planner even in the relax pass (zlo 30 mm), so it stopped 3 deg short of the 80 deg band.
- Failure mode left: the heel-side route runs out of reachable faces near the end. A likely fix (untested, out of budget): when the cube centre is near the heel, push the face looking toward the fingers at its -y end, or accept a smaller zlo/stroke for the last ~15 deg.
- FINAL = 5 (identical code to 6; best dev result).
