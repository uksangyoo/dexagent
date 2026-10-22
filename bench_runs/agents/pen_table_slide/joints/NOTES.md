# pen_table_slide at level joints


## iter 1: 0/3
Thumb-down side grasp (palm vertical, fingers horizontal at yaw 0.7, pen along palm y so it slides freely through a thumb-index pinch along palm z). Joint-level IK built in-policy: finite-difference pad Jacobian (probe all fingers' joint j at once, read api.tips()) + damped Gauss-Newton on measured tips; converges to <1 mm in ~8 s sim. Squeeze loop: step pads 0.8 mm inward via J until each force > 0.8 N.
Result: slide 18-22 mm in all cases (descend 32 mm in 8 chunks), but on lift the pads slid up off the pen; pen stayed in the guide (final tilt ~7 deg, which is the free lean in the 14 mm bore). Suspect the pen wedged in the bore (pinch not over the tip) — grip 3 N vs 5 g weight can't otherwise slip.
Skills: `jacobian(api,q,fingers)` FD pad Jacobian; `tips_ik(api,targets,q)` measured-tip IK; `pad_step(J,d)` DLS joint step for a pad displacement.

## iter 2: 0/3
Added post-slide "tilt fix" (move pinch over the pen's lower end) and chunked lift. Still 0/3: slide 20-27 mm but the pen is already leaning 5-8 deg right after closing (one pad reaches the pen first and shoves it; the side film shows it), it wedges in the bore; the tilt fix moved the pinch 1 big step and lost contact.
Learned: pad-centre-symmetric IK targets do not give a symmetric contact (pad-surface offsets differ thumb vs index); need force-guarded closing.

## iter 3 (run killed by my own 590 s timeout while the 3rd case ran; not counted; 2 cases finished, 0/2)
Tried force-guarded closing (thumb then index advance until 0.12 N) + verticalize() (palm steps <=3 mm to put the pinch over the pen's lower end). Learned: guarded single-finger closing fails — the free-standing pen is pushed ~7 mm with no measurable force until it hits the bore wall, so the thumb over-travelled to an extrapolated posture (u=1.6); verticalize then made the pen vertical (tilt 0.1 deg at grasp — works). But during the slide grip was lost (slide 8-10 mm), likely because squeeze steps used the q_close Jacobian far from the extrapolated thumb posture.
Skill: `verticalize(qg, xyz)` palm-correction loop using the pen pose in the palm frame; `pen_points(api, mid)`.
Measured: both pads' contact surface sits ~6.5 mm from the tip-site pad centre (thumb and index alike).

## iter 4 (3rd completed): 0/3, but close
Simultaneous close to an IK'd posture ~1 mm into the pen (D_CLOSE 9.5 mm; contact surface ~6.5 mm from pad site), verticalize at grasp (tilt 0.1 deg), slide 27-34 mm in all cases, pen lifted to tip 128-140 mm in 2/3 cases. Fails: grip force decays steadily (2 N -> 0.2 N over ~4 s) while the pen drifts toward the palm (palm -x) relative to the pads: two convex pads squeezing a thin cylinder squeeze it out sideways; palm-level verticalize only chases it. Final force 0.2/0.18 N (just under the 0.2 N rule) at sev 0; dropped at sev 1 when the tip left the bore.

## iter 5 (killed mid-run when the study paused; not counted; 2 cases finished, 0/2)
Added `centre_pads` (both pads chase the pen along palm x every step). Worse: the pen still ran to dx -9..-11 mm during the slide, the chase never caught it, and the pen dropped at bore exit (tip 60-74 mm then fell). Chasing the pen with the pads does not stop the sideways creep. Reverted to the iter-4 policy for the resumed run.

## iter 6 plan (resumed study, last budget iteration)
Base = iter 4. New skill `regrasp(qg, xyz)`: open both pads 3.5 mm (pad_step along palm z), let the pen stand free in the bore, re-centre the open pinch on the pen axis at pinch level (palm x,z, <=3 steps), close to the calibrated q_close, squeeze to 0.8 N. Used once mid-slide (k=2..5) if the pen offset >3 mm or a force <0.5 N, and always before lift. The lift no longer verticalizes the palm (a lean <=6 deg in the bore clears the wall when lifted straight up); it lifts 140 mm in 10 mm chunks with squeeze top-ups, then a 0.8 s hold loop tops forces up to 1.2 N. Slide D capped at 30 mm (was 32).

## iter 6 (4th completed, last budget): 3/3
Result: all pass. Slide 24-29 mm, tip lifted to ~141 mm, final grip 1.0-1.4 N on thumb and index, final tilt ~16 deg. The mid-slide regrasp fired at k=4 in every case (offset ~3.4 mm). After each regrasp the grip held steady (1.0-1.3 N through lift and hold, no decay). The pen still tips about 7 to 16 deg when it leaves the bore, but it stays held.
Learned: the grip dies from slow sideways creep of the thin pen out of the two-pad pinch, plus the torque that palm-level verticalizing builds up between pinch and pen. Fighting it continuously (iter 4 palm verticalize, iter 5 pad chasing) loses. Resetting it with a release and re-close on the free-standing pen (the bore keeps it upright) is cheap and works. Skill: `regrasp()`, as described in the iter 6 plan above.
