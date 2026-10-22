# cube_reorient at level synergies


## iter 1 (0/3) — calibration probes
Policy probed each of the k=6 synergies +-2 sd from api.open_coeffs (cube on palm), logging api.describe(), mean/basis, api.tips() at each probe, and cube response; then would exploit the best single-synergy push.
Result: no single synergy at +-2 sd from open moves the cube (only s0+ makes the ring press down on the cube top, 4.6 N, no yaw). 0/3, cube untouched (yaw ~0).
Learned: single synergies are whole-hand blends; the thumb stays at y>60 mm for every single-axis probe, so pushing the cube needs coordinated multi-coefficient postures. Contact-sheet camera "closeup" showed nothing useful.

## iter 2 (killed mid-run, not counted) — thumb sweep
Thumb pad servoed in synergy space (quadratic model of pad positions fitted to iter-1 probes + online bias) across the cube's +y face. Only the sev-0.5 case finished: yaw -20.5, cube pulled 47 mm toward the wrist (z 70 -> 24) by ring/little (3-12 N) instead of the thumb.
Learned: the thumb cannot get below x~70 anywhere near y<60 inside +-2.6 sd (pad errors 30-80 mm vs targets); ring/little curling hook the far top edge and drag the cube toward the wrist; cube still rests flat at z=24 (heel region OK).

## iter 3 plan — one-sided hook pull, joint-space targets -> synergies
Resumed after pause. New helper `Syn.solve(q_des, w)`: bounded weighted least squares from a desired 20-joint posture to the 6 coefficients using api.mean/api.basis (joint limits read from api.posture at +-60 sd). This is my own joint-level layer built on the synergy API.
`hook_target(fingers, u)`: listed fingers curl (MCP/PIP/DIP grow with u), others held open; index+middle alone or ring+little alone are well reproduced by 6 synergies (middle-alone too).
Policy: pull the side of the cube's far edge that is away from its centre (index+middle if cube y<=2 mm, else ring+little), ramp u slowly, stop when tracked yaw reaches 88 deg, retract, open. Up to 3 pushes.

## iter 3 (0/3) — one-sided MCP-led curl
Result: yaw -9, -1.7, +14 deg; cube barely moved. The MCP-led curl (MCP 1.9-2.3, PIP 1.4-1.6) lands the middle/ring pads on the cube TOP near its centre (pads at x~70, z~65) and presses 15-20 N straight down (palm force = finger force), which only pins the cube. Index curled fully ends beside the +y face at (30, 27, 66) mm. Sev-1 case: index+middle got 14 deg over 3 pushes.
Learned (palm-camera sheet is useful): the cube's far face sits right at the finger knuckles, so fingers can't get beyond it; joint limits from api.posture: flex 0..2.2 (MCP), +-1.6 (PIP/DIP), thumb j2 <= 0. Posture class that dragged the cube toward the wrist in iter 2 is PIP-led (MCP ~1.5, PIP 1.6, DIP 1.1-1.3).

## iter 4 plan — PIP-led far-edge pull (last budgeted iteration)
`hook_target` rewritten: pull fingers first close PIP/DIP to 1.6/1.3 with MCP low, then MCP ramps 0.6 -> 1.9 (the iter-2 class that dragged the cube); middle spread biased toward +y. Always pull with index+middle (+y half) unless cube y > 9 mm; switch sides if yaw goes the wrong way by >12 deg; up to 5 pulls, stop at 88 deg.

## iter 4 (0/3) — PIP-led far-edge pull
Result: yaw -21 (sev 0), -33 (sev 0.5, max 39), +16 (sev 1; cube dragged 40 mm to z=34 and then out of reach). Best yaw so far. The PIP-led class does drag/turn the cube; ring+little turn it ~5-8 deg per 8 s pull; index+middle gave -12 deg in 4 s but in the "wrong" (negative) direction (the thumb also gets pushed into the cube, F_thumb up to 10 N), so my wrong-way rule threw away the fastest pulls.
Learned: which way a pull turns the cube is not predictable from finger y alone (both sets turned it negative in sev 0/0.5). Rotation per pull is small and each pull also drags the cube toward the wrist, so the budget of travel runs out after ~4 pulls.

## iter 5 plan
Direction-agnostic: adopt whichever yaw direction appears first (|yaw|>6), keep pulling with the set that last gained in that direction, switch sets on a stall (<3 deg) or an 8-deg reversal. Faster ramps, up to 8 pulls.

## iter 5 (1/3) — direction-agnostic pulls  [FINAL]
Result: sev 0 PASS (yaw -84.7: two index+middle pulls -25, -64, then ring+little pulls creep to -84; cube dragged 40 mm to z=28 but still flat). Sev 0.5: -26 then -57 in pull 2, but the cube tipped (flat 0.848) and my guard stopped it; it fell back flat at -32. Sev 1: cube starts far out (z=80, y=+6); index+middle pull squeezes the cube against the thumb (thumb 10 N) without turning it; ring+little pulls only press the top (17 N) -> -6 deg.
Learned: index+middle PIP-led pulls are the strong rotator (25-40 deg per pull) but turn the cube negative and tip it; ring+little pulls give a few degrees. The tip guard (flat<0.85) was too strict: the cube fell back flat in sev 0.5, so a softer guard (pause rather than stop) would likely have kept going.
Budget used (4/4). FINAL = 5.
