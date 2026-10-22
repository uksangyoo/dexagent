# Qualitative analysis of agent workdirs (finger-coordination abstraction study)

Source: `bench_runs/agents/<task>/<level>/{NOTES.md, progress.jsonl, FINAL, policy.py}`, plus per-iteration
`summary.json` → `api_usage` for the layered agents. Snapshot taken 2026-10-22 04:43. At that time 8 agents were still running an
iteration (`ps` shows a live `dexagent.iterate`) and had no FINAL; they are marked **(run)**.
There are 49 workdirs: 12 tasks × {joints, synergies, contacts, layered}, plus cube_pinch/fingertips.

Dev scores are the successes out of 3 for each **counted** iteration, in the order they appear in progress.jsonl (killed iterations
are not listed). F = the FINAL iteration folder.

Failure categories: **grip** = physical-grip (weak or compliant two-pad pinch, slip, squirt); **geom** = geometry-unobservable;
**arm** = arm-reach-IK; **mismatch** = abstraction-mismatch; **calib** = calibration-cost; **other**; **none** (the final policy passes 3/3).

## 1. Per-workdir table

| task | level | dev | self-built abstraction in policy.py | layers used | failure | note |
|---|---|---|---|---|---|---|
| baoding | contacts | 0,0 (F2) | `place_pad`: absolute pad placement through `approach()`; thumb pin with direction picked by `reach` | contacts | other | Palm-tilt rolling. The thumb wall worked, but step B undid the rotation. The agent handed back with iter 3 still queued and was never resumed (only 2 counted). |
| baoding | joints | 0,0,0,0 (F6) | none (thumb calibrated by trial joint deltas against `tips()`; palm-tilt helper) | joints | other | The finger-base groove holds both balls, and once a ball is ejected it moves too fast. Best swap 154°. Iters 5 and 6 ran concurrently. |
| baoding | layered | 0,0,0,0 (F4) | `press_targets` via `reach`; PI orbit tilt controller | fingertips (FINAL only uses move_tips, palm tilt included); iter 1 was arm-only | geom | The thumb pin is strong, but the thumb's links sweep the balls out. Best swap 118°. |
| baoding | synergies | 0,0,0,0 (F2) | none (fixed coefficient postures, PD virtual-wall tilt) | synergies | mismatch | Could not build a thumb wall or isolate a finger. Linear synergy predictions are rough. The FINAL is the only iteration that kept both balls (swap 4-8°). |
| cap_twist | contacts | 0,0,0,3 (F5) | runtime `reach` grid planner for palm placement and finger arcs | contacts | none | The lid turned about 390° from iter 3 on. It failed palm-spread (arm yields under press force) until the press was lightened. |
| cap_twist | joints | 0,0,3,3 (F4) | pad offsets calibrated with `tips()`; DIP=0.5·PIP coupled into one "pull-in" DOF; force loop on contact_forces | joints | none | Switching to a rim hook that uses the full lid radius as the lever gave 12× more turn per stroke. |
| cap_twist | layered | 0,3,3 (F4) | `reach` IK-feasibility planner; `tracked()` scorer replica | joints, fingertips, contacts (rotate), arm | none | The force servo was tried in a killed run and abandoned. Palm-spread margin is about 2 mm. |
| cap_twist | synergies | 1,3,3 (F4) | pad arc along s0 calibrated from `tips()` (lookup, no IK) | synergies | none | A single little-pad "scratch" stroke along s0 gives about 93° per stroke. |
| card_pickup | contacts | 2,3,3 (F3) | `pads_world`: pad placement by feeding `approach()` virtual object-frame points | contacts | none | Notes say squeeze() and release() act toward the object origin, which is wrong for a flat tile. |
| card_pickup | joints | 0,3,3 (F3) | fitted multilinear pad FK (81 features per finger, lstsq) + least_squares IK; force-closed squeeze | joints | none | A lever grip with offset pads fixed the hinge droop. |
| card_pickup | layered | 0,3,3 (F4) | none beyond helpers (vertical pinch via move_tips) | fingertips, contacts (grasp, tuck), arm | none | Iter 1 failed on arm IK (pose out of reach). |
| card_pickup | synergies | 0,3,3 (F3) | linear synergy→pad model + Broyden-refined synergy IK (open/close pair) | synergies | none | The solved posture uses the thumb as a fixed anvil. |
| cube_palm | contacts | 1,1,2 **(run)** | `reach`-grid transfer-path planner; recovery re-pick | contacts | grip | Releasing a wedged pinch is chaotic and flings the cube. Pads cannot reach palm-x < 50 mm. Palm geometry was read from the collision-mesh bounding box. |
| cube_palm | joints | 0,0,2,1 (F5) | finite-difference Jacobian + Broyden thumb IK on measured pads (THUMB_SAMPLES hard-coded); `pivot`; `try_move` | joints | calib | The self-built thumb IK is multi-branched with about 10 mm residuals. The release pose jumps branch and sweeps the cube. |
| cube_palm | layered | 1,2 **(run)** | `reach`-checked pinch-shift search | joints, fingertips, contacts (grasp, translate, release), arm | other | The thumb cannot back off (12 mm IK residual) and drags the cube off the edge. The pour put-down overshoots. Palm hull was taken from the MJCF/mesh. |
| cube_palm | synergies | 1,0,3,3 (F5) | iter 1 only: in-policy LM fit of synergies to tip targets (dropped). FINAL: one scalar relaxing S_GRASP→open, closed loop on an on-palm test | synergies | none | Slow feedback relaxation plus a wrist-ward tilt. |
| cube_pinch | contacts | 3,3 (F2) | `face_points`; preshape through a virtual cube in `approach()` | contacts | none | Turned with the wrist instead of `rotate`. |
| cube_pinch | fingertips | 3,3 (F1) | none | fingertips | none | Offset of 9-12 mm (limit 15) from spring-back on release. |
| cube_pinch | joints | 3,3 (F2) | finite-difference Jacobian + DLS/Broyden finger IK (10-12 s of sim) | joints | none | A thumb branch jump was fixed with more damping. |
| cube_pinch | layered | 3,3 (F2) | `widest_preshape` via `reach` | fingertips, contacts (open, tuck, grasp, release), arm | none | No joints or synergies. |
| cube_pinch | synergies | 3,3 (F1) | `SynIK`: finite-difference synergy→pad Jacobian + Gauss-Newton/Broyden (about 6 s) | synergies | none | Re-opening moves the other fingers and drags the cube (offset up to 13 mm). |
| cube_reorient | contacts | 0,0,0,1 (F5) | `reach`-map planner; `pads_for`/`goto` (palm-frame pad targets through approach/grasp) | contacts | geom | `reach()` ignores collisions, so it chose impossible finger pairs. A centred pinch presses the cube into the palm. The offset pinch worked, but the release flicked the cube. |
| cube_reorient | joints | 0,1,3,2 (F5) | product-of-exponentials (PoE) FK fitted offline from logged api data (hard-coded MODEL) + LS IK + thumb link skeleton for collision checks | joints | other | The heel-side route runs out of reachable faces (sev 1 ended at 77° against an 80° band). |
| cube_reorient | layered | 0,0,0,0 (F2) | `reach` maps and twist planner; force-banded pin (0.3 N), built on move_tips | fingertips only | other | IK reach is not the same as a usable contact. A one-sided thumb push moves the cube toward the wrist. |
| cube_reorient | synergies | 0,0 **(run)** | bounded LS from a joint posture to synergies (its own joint layer); a quadratic tip model in killed iter 2 | synergies | mismatch | Single synergies are whole-hand blends. The thumb cannot get below x≈70 mm. The curls press on the cube's top. |
| pen_finger_roll | contacts | 0,0,0,0 (F5) | explicit radial normals; `regrip`; tripod probed with `reach` | contacts | grip | The pen squirts sideways out of the two-pad pinch. The tripod's middle pad was unreachable (5.7-7 mm residual). |
| pen_finger_roll | joints | 0,0,0,0 (F4) | finite-difference Jacobian + DLS/Broyden IK; swing/twist servo on the pad differential | joints | grip | Stick-slip hysteresis, and the grip unloads on the return. |
| pen_finger_roll | layered | 0,0,0,0 (F5) | `keep_grip` force servo and `centre_pinch` offset servo, both on move_tips | joints, fingertips, contacts (grasp, rotate), arm | grip | `rotate` makes the roll easy. The swing squirts the pen out, and contact location is not available. |
| pen_finger_roll | synergies | 0,0,0,0 (F4) | synergy IK (finite differences + Gauss-Newton/Broyden); in-contact synergy Jacobian for the swing | synergies | grip | Roll works 3/3, but the swing drops the pen. A 0.3 sd synergy step moves too many joints at once. |
| pen_gravity_pivot | contacts | 0,0,0 (F4) | `roll_about`; arc-about-tip lift; `regrip` (iter 5 never ran) | contacts | grip | Pinch force bleeds from 1.5 to 0.03 N as the pen turns. Never resumed (3 counted). |
| pen_gravity_pivot | joints | 0,0,0,0 (F4) | PoE FK fitted from iter-1 calibration (hard-coded MODELS_JSON) + LS IK; grasp designed offline | joints | grip | The pen squirts out within 4-5° of vertical. |
| pen_gravity_pivot | layered | 0,2,0 **(run)** | `hinge`, `snap`, `vert` (via move_tips/carry) | joints, fingertips, arm | grip | Every hand motion bleeds two-pad pinch force. |
| pen_gravity_pivot | synergies | 0,0,0,0 (F7) | linear synergy→tip model + Gauss-Newton with keep-out; `local()` linear solve; `regrip` | synergies | grip | The middle press levers the pen out. The arm IK checks only position and gives up orientation. FINAL held level in 3/3. |
| pen_pickup_pivot | contacts | 0,0,0,0 (F2) | `reach` search for the pinch location | contacts | grip | Contacts above the pen's equator form a downward wedge, and the pen squirts out at 5-15°. |
| pen_pickup_pivot | joints | 0,0,0,0 (F3) | finite-difference Jacobian + Broyden tip servo | joints | grip | Thick round pads against a 7.5 mm pen. The agent says all its effort went into building IK, and it wants pad-orientation control. |
| pen_pickup_pivot | layered | 0,0,0 **(run)** | force-by-offset pinch; per-finger force servo; touch sensing from pad deflection | fingertips, contacts (tuck), arm | grip | The pen squirts out of the wedge, and the thumb stalled in its straight-line close. |
| pen_pickup_pivot | synergies | 0,0,0,0 (F6) | synergy IK in iters 1-4 (dropped). FINAL: a measured 1-D synergy pinch path | synergies | mismatch | The synergy→pad map is nonlinear and Gauss-Newton diverged. No pinch formed, and the FINAL had a false touch trigger. |
| pen_table_slide | contacts | 0,3,3 (F4) | V-cage `pinch_points` (thumb vs index+middle) + `reach` scan | contacts | none | A three-contact V fixed the sideways roll-out. |
| pen_table_slide | joints | 0,0,0,3 (F6) | finite-difference Jacobian + damped Gauss-Newton tip IK; `regrasp()` | joints | none | Releasing and re-closing mid-slide beats continuously chasing the pen. |
| pen_table_slide | layered | 2 **(run)** | Tracker; bang-bang grip-force `regulate()` on move_tips | fingertips, joints | geom | After about 15 mm the pen top meets the curled index/thumb links. |
| pen_table_slide | synergies | 0,0,0,1 (F5) | finite-difference synergy→pad Jacobian + Gauss-Newton/Broyden | synergies | grip | The thumb-middle two-pad lift is marginal. Extrapolating along the synergy line does not squeeze, and there is no way to add a third finger. |
| pen_writing_grasp | contacts | 0,0,0,0 (F1) | `contact_points` for a cylinder | contacts | grip | The pen pivots in the grip at touchdown and is flung during the transport re-orientation. |
| pen_writing_grasp | joints | 0,0,0 **(run)** | finite-difference + Broyden pad IK | joints | grip | Thumb reach is the binding limit, and the two-pad hinge lets the pen topple. |
| pen_writing_grasp | layered | 0,0,0,0 (F4) | `safe_carry`; `align`; regrip via move_tips | fingertips, contacts (tuck), arm | arm | The stroke area is near the UR5e's reach limit. The arm IK fails or whips during the tilt. |
| pen_writing_grasp | synergies | 0,0,0,0 (F1) | synergy-space pad IK (central differences + Gauss-Newton/Broyden) | synergies | grip | A two-pad hinge cannot set the pen's lean. A third contact would need another IK search. |
| scissors | contacts | 0,0,0,0 (F5) | `reach` placement search; `goto()` exact pad placement + measure-and-correct loop | contacts | geom | Lift worked, but the thumb cannot get low on 11 mm walls and its links sweep the loop. The pad-offset model was wrong. |
| scissors | joints | 0,0,0,0 (F4) | finite-difference + Broyden in-air tip IK; `sweep()` | joints | calib | The ring's probed Jacobian came out wrong because neighbouring fingers were in the way. The blades opened 25° in the air but never closed again. |
| scissors | layered | 0,0,0,0 (F4) | in-policy anchor search via `reach` | joints, fingertips, contacts (rotate, tuck), arm | geom | The shank geometry was only found from film by iters 3-4. The pinch then held about 3 N and the scissors rolled out. |
| scissors | synergies | 0,0,0 **(run)** | synergy pad IK (finite-difference Jacobian + damped LS, active set, Broyden refine) | synergies | mismatch | Under synergies the thumb does not move along palm z, and the linear model is wrong beyond 1 sd. The descent was blocked. |

Failure-category totals (49 rows): none 16 · grip 16 · geom 5 · mismatch 4 · other 5 · calib 2 · arm 1.
- Physical-grip failures: 15 of the 16 are on pen tasks, and they occur at every level.
- All 4 mismatch rows are synergies-level agents.

## 2. Counts

**Self-built fingertip-level IK or kinematic model**
- **Joints level: 10/12.** The two exceptions are baoding (only a trial-and-flip thumb calibration) and cap_twist (calibrated pad offsets plus a DIP-PIP coupling).
  - 7 use measured finite-difference Jacobian IK, Broyden or Gauss-Newton: cube_palm, cube_pinch, pen_finger_roll, pen_pickup_pivot, pen_table_slide, pen_writing_grasp, scissors.
  - 3 use a fitted FK model: card_pickup (multilinear), cube_reorient (PoE), pen_gravity_pivot (PoE).
  - All 10 are still present in the FINAL or current policy.
- **Synergies level: 10/12 built a synergy→fingertip model or IK at some point.** The exceptions are baoding and cap_twist; cap_twist only used a 1-D tip lookup along s0.
  - 7 still use it in the FINAL or current policy: card_pickup, cube_pinch, pen_finger_roll, pen_gravity_pivot, pen_table_slide, pen_writing_grasp, scissors.
  - 2 dropped it as unreliable: cube_palm (rms 17 mm) and pen_pickup_pivot (Gauss-Newton diverged). Both moved to fixed or 1-D measured postures.
  - cube_reorient replaced it with a joint-posture→synergy projection, i.e. it rebuilt a joint layer.
- **Contacts level, for reference:** 5/12 rebuilt "place pad at a point" (a fingertip goto) from `approach`/`grasp` with virtual points or normals: baoding, card_pickup, cube_pinch, cube_reorient, scissors. 10/12 contacts policies call `reach` (all except cube_pinch and pen_gravity_pivot).

**Layered agents: layers in the FINAL policy** (from `summary.json` api_usage; running agents use their latest completed iteration; n=12)

| layer | agents | notes |
|---|---|---|
| fingertips (move_tips) | **12/12** | baoding, cube_reorient and pen_table_slide also move the arm only through the move_tips palm pose |
| arm (carry) | 9/12 | |
| contacts | 8/12 | only 6 used an object-manipulating primitive (grasp/translate/rotate): card_pickup, cube_palm, cube_pinch, pen_finger_roll, cap_twist, scissors. pen_pickup_pivot and pen_writing_grasp used only `tuck` |
| joints (move_joints) | 6/12 | cap_twist, cube_palm, pen_finger_roll, pen_gravity_pivot, pen_table_slide, scissors; always for preshape, tuck, open or thumb parking (3-25 calls) |
| synergies (move_synergy) | **0/12** | no layered agent called it in any iteration |

`api.reach` (the fingertip IK query) is not counted in api_usage. 8/12 layered policies call it.

## 3. Capabilities agents said were missing or worked around (number of agents mentioning each)

| capability | n | agents |
|---|---|---|
| Independent finger or thumb control under synergies (coupling, thumb unreachable, nonlinear map beyond ~1 sd) | 11 | all synergies agents except cap_twist |
| Contact location or pad-surface geometry (contacts on pad flanks, effective pad radius, tip protrudes below the pad centre) | 9 | pen_finger_roll/{contacts, layered}, pen_table_slide/joints, pen_gravity_pivot/layered, pen_pickup_pivot/{joints, layered, synergies}, scissors/{contacts, synergies} |
| Hand-link or palm geometry, collision-aware finger reach (`reach()` ignores links) | 10 | cube_reorient/{contacts, layered, joints}, baoding/layered, scissors/contacts, cube_palm/{contacts, layered, joints}, pen_gravity_pivot/joints, pen_table_slide/layered |
| Object dimensions or shape not exposed (size estimated from rest height or film) | 8 | scissors ×4, cube_reorient/{contacts, layered}, cube_pinch/{contacts, layered} |
| Force-controlled squeeze or contact (position-only squeeze bleeds or can't set small forces) | 6 | cap_twist/layered, cube_reorient/layered, pen_finger_roll/{joints, layered}, pen_gravity_pivot/contacts, pen_pickup_pivot/layered |
| Contact squeeze/release/approach act along the line to the object origin, wrong for pens and plates | 7 | card_pickup/contacts, pen_finger_roll/{contacts, layered}, pen_table_slide/contacts, pen_writing_grasp/contacts, cube_reorient/layered, pen_gravity_pivot/layered |
| Direct pad placement or a reach planner at the contacts level | 5 | baoding, card_pickup, cap_twist, cube_reorient, scissors (all contacts) |
| Arm IK: recoverable failure, orientation-aware, workspace | 6 | card_pickup/layered, scissors/layered, cube_palm/joints, pen_writing_grasp/{layered, contacts}, pen_gravity_pivot/synergies |
| Fingertip goals or pad orientation at the joints/synergy level | 3 | baoding/joints (fingertip goals), pen_pickup_pivot/joints and pen_finger_roll/synergies (pad orientation) |
| Tooling: api log keeps only the last 60 lines | 9 | cap_twist/{joints, synergies}, cube_palm/contacts, cube_pinch/synergies, pen_pickup_pivot/{joints, layered}, pen_table_slide/synergies, pen_writing_grasp/{joints, layered} |
| Tooling: the "closeup" camera does not show the hand | 9 | baoding/{contacts, joints}, cube_palm/joints, cube_reorient/{contacts, layered, synergies}, pen_finger_roll/contacts, pen_pickup_pivot/layered, pen_table_slide/layered |

## 4. Rule concerns

- **`timeout` wrapper around iterate (forbidden):** 8 agents. These runs were killed and not counted.
  - cap_twist/contacts, cube_palm/contacts, cube_palm/joints, cube_reorient/joints, pen_finger_roll/contacts, pen_gravity_pivot/contacts, pen_table_slide/contacts, pen_table_slide/joints (a 590 s timeout).
- **Partial results of killed (uncounted) iterations used for design:** 18 agents, effectively extra budget.
  - baoding/{joints, layered}, cap_twist/layered (2/3 cases, including a pass), cube_palm/{contacts, joints, layered, synergies}, cube_reorient/{contacts, joints, synergies}, pen_finger_roll/{layered, synergies}, pen_gravity_pivot/{layered, synergies}, pen_pickup_pivot/{layered, synergies}, pen_table_slide/joints, scissors/contacts.
  - pen_table_slide/contacts also read one passed case of a killed run, but re-ran the policy unchanged.
  - The most material case is cube_reorient/joints. Its probe iteration 1, killed by its own timeout, produced the 1404 (q, tips) samples from which it fitted its FK model.
- **Concurrent iterations:**
  - baoding/joints ran iters 5 and 6 overlapping.
  - pen_pickup_pivot/layered launched iter 5 while iter 4's sev-1 case was still running.
- **Offline modelling from logged api data:** this is not CPU `Kinematics`, so it is not literally forbidden. It moves calibration outside the iteration budget.
  - Fitted models pasted into the policy: cube_reorient/joints (PoE MODEL, plus offline planner checks), pen_gravity_pivot/joints (PoE MODELS_JSON; grasp "redesigned offline"), cube_palm/joints (THUMB_SAMPLES).
  - Offline design from logged data: pen_gravity_pivot/synergies (offline linear-model design), pen_pickup_pivot/synergies ("offline fit of iter-2 data", logged postures), card_pickup/synergies (`scratch/fit_test.py` on an iter-1 case log).
- **Reading files outside the allowed list:**
  - cube_palm/contacts (palm collision-mesh bounding box) and cube_palm/layered (MJCF/mesh bounds) read robot asset geometry.
  - cube_reorient/synergies' scratch scripts `np.load` `dexagent/assets/synergies/cube_reorient.npz`. That file is the synergy basis derived from other tasks' expert postures; the same data is exposed as `api.mean`/`api.basis`.
  - No notes mention `scene.py`, README, `expert/`, git history or other agents' workdirs.
- **Process kill:** pen_gravity_pivot/synergies killed its own iter 2 with a "mistaken pkill" on a shared machine. Whether other agents' processes were hit is unknown.
- **policy.py hygiene:** no `api._*` access and no robosimstudio/dexagent/mujoco/newton imports (grep). Minor: stdlib imports beyond numpy/scipy (`json` in cube_reorient/joints and pen_gravity_pivot/joints, to load a hard-coded model; `itertools` and `traceback` in pen_gravity_pivot/joints).
- **Mock APIs (no physics) for dry runs:** cube_palm/synergies (imports `dexagent.workcell.hand_down/hand_up`, outside the policy), cube_reorient/{contacts, layered, synergies}, pen_table_slide/synergies, pen_gravity_pivot/synergies. Not a violation.
- **Under-used budget, a comparability issue:** baoding/contacts (2 counted) and pen_gravity_pivot/contacts (3 counted) were never resumed. Several agents also lost iteration numbers to killed runs, which progress.jsonl correctly does not count.
