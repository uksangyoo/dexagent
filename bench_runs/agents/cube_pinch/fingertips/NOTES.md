# cube_pinch at level fingertips


## iter 1: 3/3
Thumb-index pinch along palm z (PINCH_CENTRE/AXIS from hand.py docs) with palm yaw squared to the nearest cube face; pads 4 mm above cube centre, 7 mm squeeze (forces ~10-13 N); lift 120 mm; turn by rotating the palm 38 deg about the world vertical through the cube (4 sub-steps); carry back to the start xy using the measured cube pose; lower; open.
Result 3/3, turn ~40 deg, but final offset 9-11 mm (limit 15): the cube was still ~6 mm up when released and slid ~10 mm in -y on release/drop. Other fingers left at their start (straight) posture stay clear of the cube.
Skills: `pinch_tips(hw)` thumb/index palm-frame pad targets at half-width hw; palm-turn about a world vertical axis via xyz=c+Rz(xyz-c), R=Rz R.

## iter 2: 3/3
Closed-loop lowering on the measured cube height (to ~2 mm) plus a two-stage release (loosen to light contact, then open). Result 3/3 but final offset 11.7-12.2 mm, slightly worse than iter 1: the cube still slides ~5 mm while loosening and ~11-12 mm in total, always along palm -y (away from the thumb side), whatever the yaw. So the slide comes from the compliant thumb/index springing back sideways as the squeeze is released, not from the drop. Lowering stalls ~2-3 mm above the start height (the pads seem to bottom out / the arm z lags), not a problem.
Stopped here (3/3 on two consecutive iterations). A likely next fix would be to place the cube about +11 mm along palm +y before releasing, or use a lighter squeeze. FINAL = 1 (lower worst-case offset, 11.3 mm vs 12.2 mm).
