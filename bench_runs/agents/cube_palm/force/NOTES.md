# cube_palm at level force


## iter 1: 3/3 (sev 0, 0.5, 1 all pass)
Plan: thumb-index pinch along palm z at (80,40,118) mm palm frame, held by api.grip (1.2 N each, servo) through lift and a 4x45deg wrist roll to palm-up; in-hand shift (api.translate, chosen by api.reach scan) then lower the pinch toward the palm in 4 mm steps until the palm feels >0.03 N; ungrip + release (12 mm); hold 1.8 s; regrip with the servo; roll back; place; release. On-palm time 2.6-3.1 s.
Weak margins: on the palm the index finger still touched (0.08 N) and the palm only carried 0.036 N, so a 0.5x mass cube could fall under the 0.02 N palm threshold. Placement stopped 6-9 mm above the table (the cube dropped flat on release).
Force control: grip servo used for the pick, the carry and roll, and the regrasp. It mattered: it held a steady 1.2 N through the 180 deg roll and the lowering onto the palm without tuning finger closure per cube size.
Skills: face_axis(R,d) picks the cube axis that faces a palm direction; feasible(api,targets) checks pad targets with api.reach (my own IK feasibility check); pinch_tips(c,a,w).

## iter 2: 3/3 (stop rule met: 3/3 on two consecutive iterations)
Changes: the in-hand shift is now chosen to move the cube toward the wrist and also had to pass a reach check 20 mm deeper; the index MCP is extended 0.4 rad after release (restored before the regrasp); the placement descent is guarded (2 mm steps while the cube still follows).
Result: 3.7 s on the palm in every case. The shift still came out (0,-20,-10) mm because deeper/wrist-ward pads failed the IK check. The index kept touching at about 0.085 N even after extending, and the palm carried only 0.022-0.05 N, so a light (0.5x) cube remains the main risk. The guarded descent was "blocked" (the fingertips reach the table first), so the cube still drops 8-13 mm on release. It landed flat every time, 9-22 mm off target (limit 80).
Force control mattered again: at the regrasp on the palm the index pad target had 14-15 mm IK residual, and the grip servo still closed to 1.2 N and held the cube for the roll back and the placement.
Layers used: move_joints for the other fingers and the index extension; move_tips to pre-shape and pre-contact; grip/ungrip (force) for every hold; translate (contacts) for the in-hand shift and lowering; release (contacts) to let go; carry for the arm.
