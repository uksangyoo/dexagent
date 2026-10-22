# Abstraction study: COMPLETE 2026-10-22 08:15 (phases 1-3)

Write-up: bench_runs/REPORT.md. Tables: RESULTS.md / results.json (tools/analyze.py). Per-run audit: QUALITATIVE.md.
Best policy per task + films: bench_runs/best/<task>/{reference,overview}/ (tools/film_best.py).
Phase 1: 49 runs (12 tasks x joints/synergies/contacts/layered + cube_pinch fingertips). Phase 2: 12 runs at level "force"
(= layered + grip-force servo, dexagent/levels.py Force). All have FINAL and test/summary.json (6 held-out cases).

Log of the session follows.
# Abstraction study: RESUMED 2026-10-22 03:25, user gave 8 hours (deadline ~11:20)

Wave 1 (20 agents, launched 03:25): first 20 lines of bench_runs/RESUME_ORDER.txt.
Wave 2 (launch one per free agent slot, in order): the remaining 12 lines of RESUME_ORDER.txt
  cube_palm/synergies, cube_reorient/joints, cube_reorient/layered, pen_writing_grasp/joints,
  pen_writing_grasp/layered, baoding/synergies, cube_reorient/contacts, pen_gravity_pivot/synergies,
  pen_pickup_pivot/synergies, pen_table_slide/joints, pen_writing_grasp/synergies, scissors/layered
Final eval: `.venv/bin/python bench_runs/tools/final_eval.py [workers]` (lock-protected; rerun as more FINALs appear).
Then: analyze.py, best policy per task, write-up.
Hard stop for agents: ~09:30, so final eval + write-up fit before ~11:20.
03:35 launched wave-2 cube_palm/synergies, cube_reorient/joints, cube_reorient/layered.
Still to launch (in order): pen_writing_grasp/joints, pen_writing_grasp/layered, baoding/synergies, cube_reorient/contacts,
  pen_gravity_pivot/synergies, pen_pickup_pivot/synergies, pen_table_slide/joints, pen_writing_grasp/synergies, scissors/layered
03:37 launched pen_writing_grasp/joints
03:38 launched pen_writing_grasp/layered
03:39 launched baoding/synergies
03:40 launched cube_reorient/contacts
03:41 launched pen_gravity_pivot/synergies
03:42 launched pen_pickup_pivot/synergies
03:43 launched pen_table_slide/joints
03:44 launched pen_writing_grasp/synergies
03:33 launched scissors/layered (all 32 now launched). DONE: cap_twist/synergies FINAL=4 (1,3,3)
03:35 DONE: card_pickup/synergies FINAL=3 (0,3,3); built own fingertip IK over synergies
03:52 DONE: cube_reorient/layered FINAL=2 (0,0,0,0); fingertips only; wants force-target contact
04:00 DONE: card_pickup/layered FINAL=4 (0,3,3); fingertips+grasp+tuck+carry, no synergies/joints
04:02 DONE: pen_writing_grasp/synergies FINAL=1 (0000); 2-pad hinge grip, synergies couple all fingers
04:04 DONE: pen_pickup_pivot/synergies FINAL=6 (0000); synergies couple fingers, built+dropped own IK
04:11 DONE: cube_palm/synergies FINAL=5 (1,0,3,3); 1-D grasp->open scalar + closed-loop release
04:11 DONE: cap_twist/layered FINAL=4 (0,3,3); joints+tips(reach)+contacts rotate, no synergies; wants force control
04:12 DONE: pen_table_slide/joints FINAL=6 (0,0,0,3); own FD fingertip IK; regrasp trick. CAVEAT: several agents read partial case results from killed (uncounted) iterations
04:12 DONE: cube_reorient/contacts FINAL=5 (0,0,0,1); offset antipodal pinch + rotate; built fingertip goto from grasp
04:14 DONE: cube_reorient/joints FINAL=5 (0,1,3,2); fitted own FK+IK from logged api data
04:15 DONE: scissors/layered FINAL=4 (0000); fingertips+reach main; object geometry unobservable, weak grasp
04:15 DONE: pen_gravity_pivot/synergies FINAL=7 (0000); own fingertip IK over synergies, arm IK position-only issue
04:16 final_eval #2 started (FINAL_AFTER=03:23, 3 workers, log logs/final_eval_2.log). Later: rerun with FINAL_AFTER=<#2 start> for remaining FINALs
04:19 DONE: cube_pinch/layered FINAL=2 (3,3); tips preshape+reach, contacts grasp, arm carry; no joints/synergies
04:20 DONE: baoding/layered FINAL=4 (0000, best swap 118deg); arm tilt + thumb tip press; links not exposed
04:24 DONE: scissors/contacts FINAL=5 (0000); lift worked, finger handover never solved; pad offset model wrong for short walls
04:27 DONE: baoding/synergies FINAL=2 (0000); palm-tilt PD; synergies can't isolate one finger
04:29 DONE: pen_writing_grasp/layered FINAL=4 (0000); move_tips chuck + tuck + carry; arm reach limit
04:32 DONE: pen_table_slide/synergies FINAL=5 (0,0,0,1); own synergy->pad IK; 1-D squeeze limit
04:33 DONE: pen_finger_roll/layered FINAL=5 (0000); roll easy via rotate, swing loses 2-pad pinch; no contact locations exposed
04:35 DONE: baoding/joints FINAL=6 (0000, best 154deg); ran iters 5&6 overlapping (budget ok); wanted fingertip goals
04:36 DONE: pen_finger_roll/synergies FINAL=4 (0000); own synergy IK; roll ok, swing fails
04:36 final_eval #3 started (FINAL_AFTER=04:16, 3 workers, -u, log logs/final_eval_3.log)
04:39 DONE: cube_palm/joints FINAL=5 (0,0,2,1); own thumb IK (FD+Broyden), wall posture; IK branch flips
04:40 DONE: cube_pinch/synergies FINAL=1 (3,3); own synergy->pad pinch solve (6s)
04:44 DONE: scissors/synergies FINAL=4 (0000); own constrained synergy->pad solver; ring lateral not in span
04:46 DONE: pen_pickup_pivot/layered FINAL=5 (0000); move_tips + own force servo; overlapped iters 4&5
04:47 DONE: pen_gravity_pivot/layered FINAL=3 (0,2,0,0); move_tips+reach+carry; wants force hold
04:49 DONE: pen_writing_grasp/joints FINAL=4 (0000); own FD IK; 2-pad hinge
04:59 DONE: cube_palm/contacts FINAL=5 (1,1,2,1); read palm mesh bbox once (noted). QUALITATIVE.md written by analysis agent (49 workdirs total)
05:02 PHASE 2 launched: level "force" (= layered + grip-force servo, dexagent/levels.py Force) on all 12 tasks, fresh workdirs bench_runs/agents/<task>/force, budget 4. Smoke test OK (cube_pinch: holds 2.5 N +-0.05 through lift). Hard stop for phase-2 agents ~08:45; then final_eval with FINAL_AFTER=<phase2 start> and analyze.
05:06 DONE: cube_palm/layered FINAL=4 (1,2,2,1); joints shapes + reach + contacts grasp/translate/release; no synergies
05:09 DONE: pen_table_slide/layered FINAL=3 (2,3,3); joints preshape + move_tips per-step force band; no synergies/contacts
05:14 final_eval #4 started (FINAL_AFTER=04:36:18). NOTE: force workdirs will also get picked up by later evals — run final eval #5 after phase 2 with FINAL_AFTER=now
05:21 DONE: cube_reorient/synergies FINAL=5 (0,0,0,1); built joint-space layer ON TOP of synergies (lstsq to 6 coeffs). PHASE 1 COMPLETE (49/49 FINAL)
05:50 DONE P2: cap_twist/force FINAL=2 (3,3); grip down + rotate world; all layers but synergies; force convenient not essential
05:57 DONE P2: cube_palm/force FINAL=2 (3,3) [layered was 1,2,2,1]; grip decisive: one setting for any cube size
06:01 DONE P2: card_pickup/force FINAL=2 (3,3) [layered 0,3,3]; grip for drag press + thin plate pinch
06:03 DONE P2: cube_reorient/force FINAL=4 (0000); servo good for contact but link collisions unseen (collision-aware reach missing)
06:06 DONE P2: pen_pickup_pivot/force FINAL=4 (0000); thick tips vs 7.5mm pen on table: geometry, not force
06:09 DONE P2: cube_pinch/force FINAL=2 (3,3); grip insurance not necessity
06:25 DONE P2: baoding/force FINAL=4 (0000, best 106deg); grip pin on sphere useful; links unseen
06:28 DONE P2: pen_finger_roll/force FINAL=2 (0000); grasp solved by grip, swing unstable 2-pad (needs 3rd contact / shear force)
06:31 DONE P2: pen_writing_grasp/force FINAL=4 (0000); grasp effortless, grasp shape (tripod) is the bottleneck
06:31 final_eval #5 started (FINAL_AFTER=05:13, 4 workers): cube_reorient/synergies + finished force cells. Need #6 for last 3 force cells.
06:50 DONE P2: pen_gravity_pivot/force FINAL=4 (0000) [layered 0,2,0,0]; servo travel cap hides low force; object-frame dirs feedback issue
06:54 DONE P2: pen_table_slide/force FINAL=3 (0,3,3); lean geometry fix, grip kept force on moving pen
06:55 final_eval #6 started (FINAL_AFTER=06:30): pen_gravity_pivot/force, pen_table_slide/force. scissors/force still running -> #7
07:06 DONE P2: scissors/force FINAL=3 (0000); grip lifted scissors (>=6.5N), 2-contact twist weakness. ALL AGENTS DONE.
07:18 PHASE 3 launched: bench_runs/agents/<task>/skills for pen_finger_roll, pen_pickup_pivot, pen_writing_grasp, scissors, baoding, pen_gravity_pivot. force level + read all prior workdirs (SKILLS_BRIEF.md), 6 iters, seed_tag=skills, no new iteration after 10:15. Then final_eval FINAL_AFTER=07:20, analyze, film_best, update REPORT.md section 6.
07:45 DONE P3: pen_finger_roll/skills FINAL=2 (3,3) — first agent solve; 'cigarette' grip between index/middle via abduction + grip servo, palm rotates about MCP axis. Read hand MJCF/mesh bbox (noted).
07:49 DONE P3: baoding/skills FINAL=2 (000000, best 113deg); reused layered helpers; thumb links block swap
07:56 DONE P3: pen_gravity_pivot/skills FINAL=6 (0,0,3,2,2,3); reused layered pivot + V-cage + regrip; servo fixed grip bleed
07:56 DONE P3: scissors/skills FINAL=5 (0,0,0,0,1,0) — first scissors pass; reused contacts+joints helpers; servo edges
08:01 DONE P3: pen_writing_grasp/skills FINAL=4 (000000); valid strokes 2-3/3 per iter, final hold fails; 3-pad chuck slides sideways
08:02 DONE P3: pen_pickup_pivot/skills FINAL=4 (000000, pivot up to 74deg, stood_on_end once); middle-finger ledge via servo. grip settle bug: 0N counts settled for target<=0.3N. ALL PHASE 3 DONE.
