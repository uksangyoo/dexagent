# cap_twist at level joints


## iter 1 (0/3)
Plan: palm still over far half of lid; index/middle/ring curl onto lid top (MCP force loop on contact_forces) and sweep abduction toward thumb (+y) = CCW from above; lift, swing back, repeat. Calibrated pad positions in the air via api.tips() (hook / abd+0.3 / lifted postures).
Result: 0 deg on all cases, zero lid force. Bug: engagement guard used the *lifted* pad position, which sits ~20+ mm further forward (MCP extension moves the pad forward), so no finger ever engaged. Also the bench keeps only the last 60 log lines and every per-step api.move logs one, so my notes were lost.
Skill: `notes` buffer flushed via api.note in a finally at the end (keeps the first 10 + last 45 lines).

## iter 2 (0/3; 103/101/114 deg)
Fixed the engagement guard (predict the pressed pad position from the lifted one using the calibrated offset). The lid now turns ~6.5 deg per 3 s stroke, the same on every case, so 16 strokes give ~105 deg. The palm stays still (spread 1 mm, yaw 0.16 deg).
Learned: positive abduction moves the pad toward -y (little side), at ~65 mm/rad. Limits: index abd [-0.84, 0.26], middle [-0.58, 0.73], ring [-0.21, 0.84]. MCP -0.35 lifts the pad 22 mm and moves it 21 mm forward (a 45 deg diagonal). Pressing on the lid top gives a tiny moment arm (pads cross 8-18 mm from the axis) and mean forces of only ~0.5 N, so the fingers mostly slip.

## iter 3 (3/3; 438/430/425 deg)
Switched from pressing on the lid top to a rim hook. Index/middle/ring reach past the lid (middle pad placed at R_max + 8 mm forward of the axis, 4 mm below the lid top) and pull back toward the axis with a PIP+DIP force loop (DIP = 0.5*PIP, F_target 2.5 N, 0.005 rad/N/step). Abduction then sweeps them toward the thumb, +-0.45 rad, over 1.2 s. A pad pulls only while its lateral offset is within 28 mm of the axis, and retracts otherwise. Lift = MCP -0.35, swing back, lower, repeat.
Result: ~72 deg per 2.7 s stroke on every case, so 6 strokes take 23 s; palm spread 1 mm and yaw 0.04 deg. Pads settle at r ~42 mm (lid R + ~7.5 mm pad). The full lid radius as moment arm was the key change (12x per stroke).
Skill: `set_pip(q, f, pip)` couples DIP to PIP for one "pull-in" DOF per finger; lateral-window engagement based on tips_world() relative to the jar axis.

## iter 4 (3/3; 439/377/446 deg), the same policy as iter 3
Confirmation run: 3/3 again, so the stop rule is met. 71-75 deg per stroke on every case. The policy stops once it passes 371 deg (GOAL = 2*pi + 0.2), so one case ended at 377 deg after 5 strokes. That is a thin margin above 360, but max turn = final turn (no back-rotation during lift or settle). A larger GOAL (e.g. 2*pi + 0.6) would cost one more ~2.7 s stroke and add margin; I left it untested because the budget is used up.
FINAL = 4.
