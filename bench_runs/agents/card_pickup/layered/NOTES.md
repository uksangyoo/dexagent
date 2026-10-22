# card_pickup at level layered


## iter 1: 0/3 (all ARM IK FAILED in the first move)
- Plan: palm-down index pad drags tile toward the book's near (-y) edge until its centre is 4 mm inside, then rotate hand to fingers-down/palm-facing +y so the thumb-index pinch axis (palm z) is vertical (thumb on top, index under the overhang), close with contacts `grasp`, lift.
- Failure: I used hand_down(yaw=-pi/2) (fingers toward the robot) for the slide, which puts the palm ~0.15 m behind the pad at y~+0.1, z~1.03: out of the UR5e's reach. Learned: keep the palm on the robot side (y < ~-0.1); use default yaw 0.7 for the slide.

(iter_2 folder: killed mid-run when the study paused; not counted. Next run is iter_3 = 2nd budgeted.)

## iter 3 (2nd budgeted): slide yaw fixed to default 0.7, first full run of the plan
- Layers: `tuck` (contacts) for middle/ring/little; `move_tips` (fingertips) to place the index pad for the drag and to open the thumb-index pinch (the pinch geometry is naturally a palm-frame pad-placement problem); `carry` for all arm motion; `grasp` (contacts) with explicit face normals to close thumb (top face) + index (bottom face) on the overhang. Synergies/joints not used: no whole-hand posture needed, only two pads.
- Retract pulls the palm back 4 cm and up 10 cm before the big wrist rotation; rotation is wrapped in try/except with a fallback via the home pose.
- Result: 3/3 (sev 0/0.5/1). Lift 121 mm, tile 25 mm from pinch mid, tilt ~14-17 deg while hanging; grasp forces thumb 8 N / index 5 N, 2.4-2.6 N while held. Rotation to fingers-down/palm+y reached directly (no IK fallback needed).
- Learned: index pressing 5 mm into the tile gives ~11 N normal force, so book friction (0.3*11 = 3.3 N) deflects the compliant finger ~10 mm: the drag consistently undershoots by ~10-12 mm, and a 10 mm corrective drag moves the tile only 2 mm (all absorbed by finger deflection). Undershoot is harmless here (centre ends ~10 mm inside the edge, ~35 mm overhang, plenty for a 17 mm-out pinch).
- Skills: `drag-with-pad` (palm-down index pad presses object centre, carry palm to slide; force-checked press), `vertical pinch` (Rp = hand_down(-pi/2) @ Ry(pi/2): fingers down, palm faces +y, thumb above index; open pads +-21.5 mm along world z around the hand.py pinch centre via move_tips), `face-normal grasp` (contacts.grasp with explicit top/bottom normals on a thin plate).

## iter 4 (3rd budgeted): unchanged policy, confirmation run for the stop rule (3/3 twice)
- Result: 3/3 again (sev-1 case had tile yaw 16 deg, size 0.91, book +16 mm in y). Same ~10 mm drag undershoot and identical lift/hold numbers in every case: the open-loop pinch geometry is insensitive to the variation, because every placement is computed from the measured book edge and tile pose.
- Stop rule met (3/3 on iters 3 and 4). FINAL = 4 (same policy as iter 3).
- Known weakness, not fixed: the drag needs ~11 N of pad force, so the compliant finger lags ~10 mm. A lighter press (2-3 mm) or overshooting the commanded drag by ~10 mm would land the tile closer to the edge if a longer overhang were ever needed.
