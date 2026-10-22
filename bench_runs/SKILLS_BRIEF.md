# Phase 3 brief: solve a hard task, with memory of everyone before you

Read `bench_runs/POLICY_BRIEF.md` first. Everything in it applies, with three changes:

1. **Budget: 6 completed iterations.** The stop rule is the same: 3/3 on two consecutive iterations.
2. **You may and should read earlier agents' work, for any task and any level:**
   - `bench_runs/agents/*/*/NOTES.md`, `policy.py`, `iter_*/policy.py`, `iter_*/case_*.json` and contact sheets;
   - `bench_runs/QUALITATIVE.md` (a cross-run audit);
   - `bench_runs/REPORT.md` (the study write-up).

   Copy and adapt their helpers freely. That reuse is what this phase measures: how much accumulated skills help.
   Still forbidden: `tasks/**/scene.py` and its helpers, `tasks/**/README.md`, `README.md`, `docs/`, `artifacts/`, `bench_runs/expert/`, `bench_runs/best/`, git history.
3. **Your level is `force`:** every layer, plus the grip-force servo. Your workdir is `bench_runs/agents/<task>/skills/`.

## What 61 earlier agents learned (start here)

- **Two-pad pinches on thin or long objects fail.**
  - A thumb–index pinch on a 7.5 mm pen acts as a hinge about the pad line.
  - It squirts sideways under any shear, tilt or push along the pen.
  - Squeezing harder makes it worse, with or without the force servo.
- **Three contacts work.** The one pen task that was solved reliably used three contacts: a V-cage, thumb opposing index plus middle (`pen_table_slide/contacts`). A brace or fulcrum along the object (another finger, the palm, the table or a bore wall) is the fix every failed pen agent named and none had the budget to try.
- **Pad centre is not the contact point.**
  - The fingertips are about 16–20 mm thick, with an effective radius of about 10 mm.
  - On a thin object lying on the table the contacts sit above its equator and wedge it down.
  - `pad_offset` is 8 mm.
- **`api.reach` checks only the pad centre.** Finger links and the palm can collide with the object even when reach says fine, and several agents lost iterations to this. Check measured contact forces on all five fingers and the palm, not just the intended ones.
- **The force servo:**
  - It acts only along each pad's closing direction, with travel up to `max_travel` (default 15 mm).
  - If `grip_state()` shows travel at the cap and force below target, the pad cannot reach. It is not "holding".
  - Directions given in the object frame rotate with the object. During re-orientation, re-issue `grip` with palm-frame-consistent directions.
  - It does not resist twisting about the pinch line.
- **Arm.**
  - The arm follows palm poses through IK, and big wrist rotations near the edge of the workspace (about 0.75–0.8 m from the base) can fail or swing.
  - Check `measured_palm()` against the commanded pose after large rotations.
  - Pivoting the palm about a fixed point in small steps is reliable (`pen_gravity_pivot/layered` helpers `hinge`, `vert`).
- **Process.**
  - The api log keeps only the last 60 lines, so buffer diagnostics and `api.note` a summary at the end.
  - Test new code paths against a mock api for Python errors before spending an iteration (several agents have `scratch/mock.py`).
  - Film the case you do not understand.
