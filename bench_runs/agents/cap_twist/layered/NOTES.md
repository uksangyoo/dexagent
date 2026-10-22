# cap_twist at level layered


## iter 1: 0/3 (turn 385-388 deg in all 3, failed only on palm spread 15.0/16.1/17.9 mm > 15)
- Layers: joints (flat hand / preshape / thumb parking), fingertips (`reach` IK-feasibility search for palm placement + stroke angle, guarded press, lift/return), contacts (`rotate` about world z for the stroke), arm `carry`.
- Strategy: palm fixed over lid, index+ring pads press on lid TOP face either side of the axis (hinge carries the normal load, no opposition needed), 90 deg arcs, lift, swing back, press; closed loop on lid angle. Search picked xp 70 mm, zc 60 mm, rc 23 mm, alpha 90.
- Learned: gains only 33-61 deg per 90 deg stroke (contact point != pad site, compliance); my "press harder when weak" rule ramped normal force to 20-24 N/finger, and the upward reaction + arm integral windup made the palm bob ~8 mm -> spread fail. Fingers give ~1.2 N per mm of press.
- Skill: `pad(phi, x, yc, zc, rc)` palm-frame point on a circle about the lid axis; runtime `api.reach` grid search = IK-feasibility planner for palm placement.

## iter 2 (killed mid-run, not counted; 2 of 3 cases had finished): sev1 PASS (382 deg, spread 11.4), sev0.5 FAIL (222 deg, max 283)
- Changed: stroke rewritten as per-step move_tips along the arc with a force servo on pad depth (target 3 N). Palm spread fixed (11-12 mm) but gains decayed 50 -> ~0 deg/stroke and finger forces climbed to 45-75 N even with the depth at its upper clamp; in one case strokes turned the lid backwards. Abandoned the force servo.

## iter 3: 3/3 (turn 382-384 deg in 9-10 cycles by t=31-33 s; palm spread 13.6/13.8/14.0 mm, limit 15)
- Changed: back to iter-1 design (contacts `rotate` strokes about the world-vertical jar axis, fingertips layer for press/lift/swing), but constant 5 mm press (no force escalation), 0.5 s settle after the first press, and the rotate stroke issued as per-step increments (smoothstep split into 70 rotate calls of 0.02 s) so lid angle/palm can be logged each step. Escalation only if the time projection says the goal is out of reach (never triggered).
- Result: per-stroke gain 25-76 deg/90. Own palm spread over stroke steps only 7.4 mm (z within +-1.3 mm), but scorer says ~14 mm => the excess comes from non-stroke steps where the lid moves (2-3 deg backdrag per lift/re-press, ~2 deg on the final release) while the palm springs ~4 mm as the 15-25 N finger load comes off (arm integral loop wound up). Index force creeps 13 -> 25 N over cycles at constant commanded depth.
- Skill: `stroke()` = per-step contacts-layer rotate with lid/palm logging.

## iter 4: 3/3 (turn 380-387 deg; palm spread 13.3/11.2/13.2 mm) -> second consecutive 3/3, stopped. FINAL = 4
- Changed: slow unload (0.4 s, back to touch depth) before each lift and a slow end release (0.8 s); a `tracked(gen, phase)` wrapper that steps any motion generator and records lid angle + measured palm every control step exactly as check.py does. Own spread estimate now matches the scorer to 0.1 mm.
- Learned: stroke steps alone give ~8 mm spread; the 13 mm comes from lift/swing steps (palm 6.5 mm off the mean while the finger load comes off and the arm's integral loop unwinds; the lid still jiggles >5 deg/s there) and the end release. Backdrag per cycle: ~2-3.5 deg on unload+lift, ~1.6-3 deg on re-press, 0 on swing. The slow unload did not reduce it. Gains decay after cycle 4 (60-75 -> 15-35 deg) as index force creeps to 25-30 N; a mid-severity case needed 12 cycles (44 s of 54). Remaining risk for held-out cases: spread margin ~2 mm and the time margin at high thread friction.
- Layers used: joints (flat hand, preshape seed, thumb parking), fingertips (runtime `reach` IK-feasibility search for palm placement/arc, guarded press, unload/lift/swing/re-press), contacts (`rotate` for the stroke, per step), arm `carry`. Synergies not used: nothing in the task maps onto a whole-hand posture.
- Skill: `tracked(gen, phase)` = per-step monitor wrapper around any api motion generator (scorer-replica palm-spread bookkeeping by phase).
