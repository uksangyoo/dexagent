# cap_twist at level contacts


## iter 1: 0/3
Strategy: palm held fixed over the lid, 3-4 pads press DOWN on the lid top (r=21 mm; hinge => no force closure needed),
stroke = grasp(press, normals=up) -> rotate(world z, +dth) -> release (lift) -> rotate(-dth) in air -> repeat, closed loop on lid_twist.
Placement (palm-frame point at lid-top centre) + per-finger start angles found at runtime by a grid search with api.reach (fingertip IK
residual query, zero sim time) -- built myself, the level has no "where can my fingers reach" planner.
Result: each stroke turned the lid ~36-42 deg, but release could not lift middle/ring (IK residual 14 mm: plan picked palm only 55 mm
above the lid, pads could not retract upward), so the air return dragged the lid straight back. Net ~0. Contact forces 15-25 N (jammed).
Also wall time ~13-14 min/case (close to the 900 s timeout): the search (2.7k IK calls) + per-step IK is expensive.
Fix: plan now requires pads reachable both pressed and lifted 20 mm above across the stroke; deeper palm candidates (70-100 mm);
coarse-to-fine search (~700 IK calls); re-release fingers still touching after release.
Skill: plan(api,R) -- runtime reachability search over palm placement / finger arcs using api.reach.

(iter_2 directory: run killed while queued after a coordinator rule change -- it was wrapped in timeout; no results, not a completed iteration. Same policy rerun as iter_3.)

## iter 3 (2nd completed): 0/3 -- lid turned 384/391/398 deg in ~24 s, but palm_spread 25 mm > 15 mm
Plan chose palm 85 mm above lid, lid centre 135 mm along fingers; fingers middle+index+thumb, 60 deg strokes (lift check fixed release reach).
Puzzle: pressed CCW stroke moved lid -15 deg, the "in-air" return moved it +57 deg (middle still touching after release; slip-based
press growth had driven extra press to 15 mm -> 30-40 N jammed fingers; that load pushed the palm/arm ~25 mm). Net +42 deg/cycle.
Change: cap extra press at 6 mm (no slip-based growth), hold 0.3 s after palm placement, per-phase lid-angle logging, and a
sign-adaptive stroke: if a full cycle nets < -5 deg the stroke direction is mirrored (pads re-positioned in air).

## iter 4 (3rd completed): 0/3 -- turns 382/394/396 deg by t=29 s, palm spread 16.9-17.7 mm (limit 15) -> all fail on palm_still only
With extra press capped at 6 mm the intended mechanism shows cleanly: per cycle grasp -1..-8, pressed stroke +22, release +10 (finger
spring-back), air return 0 -> ~31 deg/cycle, no sign flip needed. Iter-3's weird return turning was jammed fingers (15 mm extra press).
Palm spread scales with press force (30-40 N -> 25 mm, 15-20 N -> 17 mm): the arm yields under the finger reaction.
Change (last iter): grasp inset 4->2 mm, extra cap 3 mm, squeeze only if F < 0.5 N, hold 0.6 s after palm placement; log palm deflection.

## iter 5 (4th completed): 3/3 -- turns 389/395/393 deg by t=29 s, palm spread 11.0-11.7 mm, palm yaw 0.1 deg
Lighter press (inset 2 mm, extra <= 3 mm): forces now 2-13 N, palm deflection ~2.3 mm (mostly vertical), ~31 deg/cycle, 13 cycles.
Middle finger often reads ~0 N at press time yet the stroke still gains +25 deg (index + thumb carry it).
Palm-spread margin is modest (11-12 vs 15 mm); press force is the knob. FINAL = 5 (budget of 4 completed iterations reached).
