# card_pickup at level synergies


## iter 1 (0/3)
- Plan: fingers pointing down (palm faces +y/book, thumb +x) so thumb-index pinch axis (palm z) is vertical. In-air calibration of a linear model tips(s) (central differences, 13 postures) + Broyden-refined least-squares "synergy IK" for open/closed pinch postures (built inside policy.py from api.tips(); my own tip-level controller on top of synergies). Drag tile to the near edge with the index, hook index under the overhang, thumb on top, close, lift.
- Result: drag works (75-82 mm slide, ends within 1-5 mm of target). Pinch fails: only thumb touched the tile, index never loaded; tile was pushed back +y ~20 mm onto the book.
- Learned: mean posture is already a wide thumb-index aperture (D~70mm along palm z). Thumb pad z barely controllable (<=13 mm/sd); index closing via s0 moves the pad out of the palm (+x, toward the book) 21 mm per 38 mm of closure -> index path is 29 deg off the pinch axis. Middle pad sat 14 mm above index at tile height -> likely hit the tile end and pushed it. Curled index presses with its dorsal knuckle ~20 mm behind the pad (tilted tile 24 deg during drag; press force 8 N too high).

## iter 2 (3/3)
- Changes: posture solve is now joint over (s_open, s_close) on the measured linear model: thumb static, index travels 14 mm straight toward the thumb pad, middle level with the index (co-support + second drag contact, no tile yaw), ring/little hinged up/back. The palm rotation is then derived from the solved pinch direction (index->thumb mapped to world +z, palm-x to +y) instead of fixing pinch axis = palm z. Pinch point 34 mm from the tile centre (+12 mm lateral so index/middle straddle the axis); drag pressing 30 mm behind centre, only 2 mm press.
- Result: 3/3 (sev 0/0.5/1), lift ~148 mm, tile tilts ~31 deg in the pinch, tile_to_pinch ~34 mm. Pinch forces ~10-11 N thumb/index.
- Learned: the solved posture keeps the thumb as a fixed anvil and closes with mostly s0 (+0.28) plus small corrections; residual 4 mm. Index rests against the book (14-30 N) during the slide (its pad measured 12 mm higher than planned) but still closes under the tile; little finger touched the book in sev 1. Fit is deterministic across cases.
- Skills: `_refine_pair` (joint open/close synergy IK with Broyden updates from measured pads), `_frame(n)` (palm rotation from a palm-frame pinch direction), `_move_until` (palm move with early stop on contact force).

## iter 3 (3/3) — resumed after pause; earlier iter_3 attempt was killed mid-run and discarded
- Changes: none (re-ran the iter-2 policy unchanged to confirm robustness; stop rule = two consecutive 3/3).
- Result: 3/3 (sev 0/0.5/1), lift 146-150 mm, tile_to_pinch 32-36 mm, max tilt 31-39 deg. Sev-1 case had a smaller tile (0.95x), light (0.58x) and book +11 mm in y; same behaviour as iter 2.
- Learned: the in-air calibration + Broyden-refined synergy IK is repeatable across runs; drag-to-overhang then pinch is insensitive to the drawn offsets. Untested so far: yaw near the +-25 deg extreme (max seen ~12 deg). Stopped early (stop rule met), FINAL = 3.
