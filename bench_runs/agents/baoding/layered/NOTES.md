# baoding at level layered


## iter 1 — 0/3 (swap 36/54/98 deg, all dropped)
- Approach: arm layer only (api.carry each 20 ms). Closed-loop palm tilt (<=20 deg) sweeping the downhill direction around the cup, hoping fingers/thumb act as walls.
- Learned: after the drop both balls sit side by side in a trough at the finger bases (z~95-103 mm, a at y~-25 against ring, b at y~+18 against index; phi ~77 deg at t=1). Palm face at x~21 mm (ball centre x~40).
  Home thumb pad is far out at (60,101,74) mm -> no +y wall: tilting toward +y (>=10-20 deg) eventually lets both balls slip past the index and off the thumb side. Tilting toward +z (fingers) is always safe (trough).
  Thumb pad reach (api.reach): good for y 35-50 / z 70-110 and y 20 / z>=90; poor below z~50. Little finger pad reaches (55,-30..-55,60) and z 120 only.
- Skill: tilt_R(R0, v) — palm rotation whose in-plane downhill direction/magnitude is v (deg), pivoting about a palm point via api.palm_for.

## iter 2 — 0/3 (swap 43/61/58; sev1 dropped)
- Approach: tilt bias (arm layer, via move_tips' palm pose) + thumb pad closed-loop pushing (fingertips layer). Plan: thumb pushes b down out of the trough, then left under a.
- Result: thumb pressed b with ~4 N but b never moved: a had not climbed into the trough (b sat at y~0, blocking a), so a was right below b and blocked the push. With tilt having a -y component while a is low on the palm, a rolled off the little-finger side (sev1). Thumb reach map (x 45-62): y 0-40 at z>=80 all reachable; z<=60 not.
- Learned: with the thumb out of the way and tilt toward the fingers, b goes to the right pocket (y~12-18,z~103) and a rises to the left pocket (y~-25,z~95). Never tilt toward -y when a ball is low.

## (killed run, not counted) — index/middle MCP "paddle" + thumb push
- Only the sev-0 case finished before the pause: a stuck at (-23,75) against b (jammed by ball-ball friction at 8 deg tilt), then the index/middle MCP flex flung BOTH balls out of the hand at t~4.3 s. Lesson: finger flexion near the trough is violent at these gains; do not paddle balls with the fingers.

## iter 3 (folder iter_4/, because the killed run kept iter_3/) — 0/3 (swap 118/112/94 deg, all dropped)
- Approach "pin and orbit": thumb pad pins b in the finger trough from above (fingertips layer; pressing direction chosen at runtime with api.reach over 10 normals; ~4-5 N, rock solid), palm tilt (arm layer) with downhill = into the pinned ball + PI-controlled tangential part walks a under b toward the thumb side. Plan was to then swap roles (re-pin a, orbit b).
- Result: phase 1 works in all 3 cases (+80..+120 deg counted incl. settling), but a always stops at about (y 20, z 70) mm against the thumb's own links (Fa thumb 0.2-0.9 N). The role swap killed every case: as soon as the thumb pad moved sideways toward a, the thumb links swept a off the +y (thumb) side; then my edge-guard tilt (up to 14 deg) threw b out as well.
- Learned: the thumb pin is easy and strong; the thumb's proximal links occupy the palm's +y half around z 60-90 and act as a wall; a pure +x lift of the pad (10-20 mm) does not disturb the wedged ball, any y/z motion of the thumb does. Thumb reach at x=60 (residual mm): y-20..40 fine for z>=70-80; low z only on the -y side.
- Skills: press_targets(api, P, r) = best thumb pressing pad target/hover/normal via api.reach; pin(): lift -> hover -> press with force check; orbit(): PI tangential-speed tilt controller for one ball circling a post ball; open_arc(): how far a ball can circle a post inside the palm box.

## iter 4 (folder iter_5/) — 0/3 (swap 10/33/87 deg, all dropped) — budget used up
- Change: no thumb re-pinning. Phase 1 as before; on stall keep the thumb where it is, tilt so the wedged ball leans on the thumb links, lift the pad 10 mm straight up, and orbit the formerly pinned ball around the wedged one (+y bias).
- Result: worse. sev0/0.5: after the lift, b stayed put in the finger trough (the trough holds it) while the phase-2 tilt (toward -z/-y, i.e. "b into a" + tangential) rolled the supposed post a off the wrist/little-finger side; the thumb wall only resists +y. sev1: my role heuristic (open_arc + reach score) picked a as the pinned ball (a was low at t=1), b could not circle a (trough blocks it), and in phase 2 a ran off the +y side with no thumb next to it. Bug in my plan: I meant to always pin the upper (trough) ball.
- Learned: the finger trough is the only reliable holder; an unpinned ball at mid-palm has no support against any tilt except +z and (with the thumb pinning at the trough centre) +y. A working scheme probably needs a second finger to hold a ball (e.g. a curled finger as a post), or the thumb to drag the pinned ball, not tilt-only motion of a free ball across mid-palm.
- FINAL = 4 (folder iter_4, the 3rd counted run): highest swap on every severity (118/112/94); policy.py restored to that version (iter-5 version kept in scratch/policy_iter5.py).

## Layers used (layered level)
- Arm layer (palm tilt via move_tips' palm pose, closed loop on ball poses): all ball motion, every iteration.
- Fingertips layer (thumb pad IK + api.reach to choose a pressing direction): thumb pin / pushes. Joints layer only in the killed run (index/middle MCP paddle; violent, abandoned). Synergies and contact primitives unused: the task is non-prehensile rolling of two free balls, where object-frame grasp primitives and posture synergies do not map onto "press this ball here while the palm tilts"; pad-position IK was the useful abstraction.
