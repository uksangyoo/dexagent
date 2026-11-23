# pinch_then_palm (long horizon)

The `cube_pinch` scene, with a two-skill goal: first do what `cube_pinch` asks (fingertip pick, lift, turn the cube while held, put it back where it started), then do what `cube_palm` asks (pick it up, transfer it onto the open palm, hold it there, set it back down and let go).

The two skills were learned in isolation and can be run with `yield from api.run_skill("cube_pinch")` and `yield from api.run_skill("cube_palm")`. Read `skills_lib/` for their policies and notes.

**Objects.** `cube`: as in cube_pinch (box, nominal edge 40 mm, mass 12 g, friction 0.9).

**Start.** As cube_pinch. **Variation (severity 1).** As cube_pinch: position +-25 mm, yaw +-30 deg, edge +-15 %, mass +-50 %.

**Success (check.py).** Both component checks pass on the same rollout:
- cube_pinch: lift > 80 mm; yaw change > 25 deg while > 60 mm up; at the end resting flat and within 15 mm (xy) of the start;
- cube_palm: carried on the open palm for >= 1.0 s in total (palm up, > 100 mm above start, palm force > 0.02 N, fingers < 0.5 N); at the end resting on a face within 5 mm of start height, within 80 mm (xy) of the start, with total hand force on the cube < 0.05 N.

Sim time limit 60 s; the bench holds the final targets for 1 s before scoring.
