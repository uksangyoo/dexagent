# pen_table_slide at level synergies


Resume (2026-10-22): earlier agent's iter_1 was killed mid-run (no summary, not counted); next folder is iter_2 = 1st budgeted iteration.
Skills in policy.py: `probe` (move to synergy s, settle, read pads); runtime finite-difference synergy->pad Jacobian (k probes, 0.5 sd)
+ `solve` (Gauss-Newton with Broyden updates on measured pads) to find a thumb-index pinch opposing along palm z at C_DES;
close along pre->tight synergy line until both pads > 0.8 N; lower palm in 3 mm steps, re-squeeze if a pad < 0.5 N; lift 130 mm.
(This is my own synergy-space "IK" built from api.tips(), per rule 1.)
Added before iter_2: lowering capped so pads stay >= 12 mm above guide top; stop if palm touches pen (> 1 N).

iter_2 (1/4): 0/3. Calibration picks thumb-MIDDLE pinch (k=6; tight pads 18 mm apart, well opposed). Slide works (scorer 19.7/14.7/5.6 mm;
my coarse estimate said ~24 -> overestimates because scorer counts only while both pads > 0.2 N). Every lift dropped the pen (max tip 23 mm,
then back in bore); final tilt 5-8 deg ~ bore limit (6 deg) => pen wedged against bore by off-centre grasp; one case lost thumb contact.
Change for iter_3: per-step slide tracker mirroring the scorer (`track`/`tracked` wrapper); xy centring servo (`centre_err`: move palm so
the pen axis at pad height is over the bore centre) at grasp, during slide and lift; force-regulated squeeze (`squeeze_to`); stepwise lift
with re-squeeze if pen lags palm; multi-stroke slide and regrasp-on-drop loops.

iter_3 (2/4): 0/3. Slide tracker now matches scorer (18.4 vs 18.5, 27.6 vs 28.3 mm); centring gives tilt 0.1 deg at grasp. Sev1 slid 28 mm and
lifted pen to 97 mm, then dropped it: lift squeezes extrapolated the synergy line to u=2.6 (tight posture is u=1), which moves the middle
pad off the pen -> learned: synergy-line extrapolation beyond ~1.4x is not a squeeze. Sev0.5: pen top hit palm/knuckles after ~18 mm slide.
Nominal timed out (60 s) in repeated strokes (tracker threshold 0.3 N undercounted vs scorer's 0.2). api_log keeps only the last 60 lines.
Change for iter_4: u capped at 1.4, slide goal 21 mm, grip 14 mm below pen top (more room before the palm), palm-touch stop 0.3 N,
tracker threshold 0.25 N, gentler lift re-squeeze, time guards on strokes/regrasps.

iter_4 (3/4): 0/3, worse. Gripping 14 mm below the top lost contact mid-slide (slides 10.6/0.4/7.1); when grasps failed, the
centring servo (driven by an unheld pen) walked the palm 12 mm off and regrasps aimed at the bore centre missed the tilted pen.
But a fresh regrasp 40 mm below the top lifted the pen to 133 mm (u 1.03, 0.56 N each) -- it then fell during the final hold.
Change for iter_5: decouple phases -- slide with the iter_3 grip (top-20 mm, which slid 18-28 mm), stop at 21 mm tracked; then
release and take a fresh grasp 35 mm below the top aimed at the pen's measured axis (`axis_at`), lift with u <= 1.4, settle loop
re-squeezing to >= 0.7 N. `centre_step` now only acts while both pads press (> 0.3 N) and its offset is bounded to 8 mm.
Smoke-tested the generator logic against a mock api (no simulator) to catch Python errors before spending the last iteration.

iter_5 (4/4): 1/3 (nominal PASS: slide 25.8 mm, tip 131.7 mm, held 0.69/0.69 N). Slide now passes in all three cases (25.8/22.4/25.4 mm);
decoupling slide grasp and lift grasp was the key fix. Remaining failure = the lift: sev0.5 carried the pen to 100 mm, then it fell
(tilted out of the two-point pinch, flung off the table); sev1 (short pen, -9%) pulled the pen ~50 mm up the bore, then it slipped back
(two regrasps failed, pen tilted 6-8 deg against the bore). A thumb-middle two-pad pinch at ~0.7-1 N on a 7 mm pen is marginal: torsional
friction alone has to keep the pen from swinging about the pinch axis, and the synergy line gives no clean way to raise force or add a
third finger without moving the pads off the pen. FINAL = 5 (only iteration with a pass; most robust slide).
