# cube_pinch at level force


## iter 1: 3/3 (sev0, 0.5, 1 all PASS; lift 129 mm, turn 39-42 deg, final offset 0.3-0.5 mm)
- Plan: thumb-index pinch along palm z at the hand.py pinch centre (80,40,118 mm palm frame), palm yaw = cube yaw (mod 90) nearest 0.7 so the pads land on opposite faces. Pinch height = max(half+3 mm, 22 mm) above the table (COM just below the pinch line, tips clear of the table).
- Layers: joints (OPEN as the IK seed), contacts `tuck` (middle/ring/little), fingertips `move_tips` for the pre-pinch (pad surfaces 6 mm clear) and the straight close (1.5 mm interference), **force `grip` 1.5 N thumb+index** right after the close, held through lift/turn/carry/lower, then `ungrip` before opening. Arm via `carry` with closed-loop corrections (commanded-vs-measured palm bias; cube world position vs target before lowering).
- Force control: the servo settled within 0.15 s and held 1.5 +- 0.05 N throughout with <=1.1 mm of travel. It mattered as a guarantee (the grip pressure doesn't depend on how far the pad is inside the cube, so cube size +-15 % needs no tuning), but the geometric squeeze would probably have worked too.
- Helpers (skills): `track(gen)` wraps any motion to log my own lift/turn estimate per step (matches the scorer exactly); the pinch-centre/aperture search uses `api.reach` at runtime (my own small use of IK feasibility, no offline precompute); `place_at(target)` moves the palm so the measured cube centre goes to a world target. Turn done as 4 x 10 deg palm slices about the measured cube centre.

## iter 2: 3/3 (policy unchanged; lift 129 mm, turn 42-43 deg, offset 0.3-0.6 mm; sev1 had edge -7 %, mass -26 %, yaw -15 deg)
- Same policy, run again to confirm. The force servo held 1.5 N again across the cube sizes and masses. Stop rule met (3/3 twice in a row), so I stopped at 2 of 4 iterations. FINAL = 2 (the same policy as iter 1).
- What I learned: with the palm yaw squared to the cube, the hand's built-in thumb-index pinch line (palm z) is a reliable opposition grasp for a 34-46 mm cube. `api.reach` showed the aperture is never the limit here. Turning the arm about the measured cube centre in 10 deg slices keeps the cube in place, and closed-loop cube-position placement puts it back within about 0.5 mm.
