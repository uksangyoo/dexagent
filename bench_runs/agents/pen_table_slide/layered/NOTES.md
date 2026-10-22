# pen_table_slide at level layered


## Plan (resumed 2026-10-22; earlier agent left policy.py, never ran an iteration)
Layers: joints layer for the initial posture (open thumb/index, middle/ring/little tucked = IK seed);
fingertips layer (move_tips every 20 ms step) for approach, closed-loop force-regulated thumb-index
pinch (common half-width h about the pen axis, pinch axis = palm z, horizontal in world), slide and lift;
arm via move_tips(xyz=...). Contacts/synergy layers unused: closed-loop grip regulation needs per-step
pad targets, which the fingertips layer gives directly.
Skills in policy.py: `Tracker` (replica of check.py's slide measure from api observations),
`regulate()` (bang-bang grip-force band on h), `close()` (close until both pads > 0.6 N),
`slide()` (lower palm 8 mm/s, stop at 21 mm tracked slip / 40 mm / pads near block top), restroke loop.

## Iter 1: 2/3 (sev0 PASS, sev0.5 PASS, sev1 fail)
Ran the inherited policy unchanged (filmed sev1 closeup; the sheet was too far to read).
In all 3 cases the single stroke lost the pen after ~15.5-17 mm of lowering (forces -> 0 even as h shrank),
the same travel regardless of pen length -> geometric: the pen top rises ~27 mm above the pinch centre and
meets the curled index/thumb links, which push it sideways (re-grasp centre moved 8 mm in palm -y).
Restroke rescued slip (>21 mm) in all cases, but in sev1 the displaced, tilted pen slipped out at lift start.
Lesson: slide through a two-pad pinch works well (slip ~= lowering after ~1 mm finger deflection), but travel
per grasp is limited to ~15 mm by hand geometry -> do short strokes (<= 9 mm) and re-grasp in between.

## Iter 2: 3/3 (all severities PASS, slip 22.1-22.2 mm, final tip 118 mm, forces 0.5-0.6 N)
Change: short strokes (stop each grasp at 9 mm of pen travel or 14 mm lowering, or if the grip is lost),
then open h by 5 mm, climb back to the grasp height re-centred on the pen axis (`reposition`), re-close;
3 strokes reach 21 mm. Added `lift()` with lost-pen detection and one retry from above (not needed).
Grip stayed balanced and centred (re-grasp centre moved < 2 mm); first-contact closes at h ~10.3 mm.
Skills added: `slide(stroke_goal)` (stroke-limited), `reposition()` (open + palm re-centre), `approach_from_above()`, `lift()` (returns ok).
Iter 3: same policy unchanged, to check the stop rule (3/3 twice).

## Iter 3: 3/3 (same policy as iter 2; slip 22.1-22.3 mm, final tip ~118 mm, final grip 0.41-0.57 N)
Stop rule met (3/3 on iters 2 and 3). FINAL = 3 (identical policy to iter 2).
Layer usage: joints layer once (initial tucked posture); fingertips layer for everything else (one move_tips per
20 ms step during close/slide/lift, plus a few timed moves for approach/re-centre). Synergy and contact
primitives unused: the task needs a force-regulated pinch updated every step, and per-pad targets plus
contact-force feedback gave that directly; nothing in the higher layers would have addressed the real
difficulty (pen top running into the curled fingers after ~15 mm of travel), which only showed in the metrics.
