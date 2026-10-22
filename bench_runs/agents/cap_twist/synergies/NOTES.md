# cap_twist at level synergies


## iter 1 — 1/3 (sev0 fail -2132deg, sev0.5 fail -2126deg, sev1 PASS +438deg)
- Policy: "scratch" gait. Synergy s[0] alone (others 0) is a near-pure index..little flexion; calibrated pad arcs in free air at the start (s0 = -3..2), placed lid centre 4 mm on the little side of the little pad's lowest point, palm height so the arc dips 8 mm into the lid top, sweep, lift palm 16 mm, re-extend, lower, repeat.
- Finger spacing is wide (~25-30 mm): only the little (plus ring when lid big/offset) reaches the lid. With 4 mm lever the sign was fragile: nominal turned the lid CW ~130 deg/stroke (pad slightly past the centre). Sev1 had ring+little on the + side: +36 deg/stroke, PASS. Contact forces 19-26 N on the little pad at 8 mm depth (plenty).
- Palm spread 11-27 mm: palm-height "deepen" adaptation (+6 mm) and lid creeping while the palm lifted. api log keeps only the last 60 lines, so early calibration notes were lost — fewer move calls per stroke needed.
- Skill: calibrate-arc (sample api.tips() along a synergy line in free air, interpolate, find each pad's lowest point) — used to place the palm relative to the lid.

## iter 2 — 3/3 (+489, +484, +478 deg; palm spread 7.8-8.8 mm, yaw 0.35 deg)
- Changed: little pad only, lid axis placed 16 mm beyond it on the -y side (lever 16 mm), fixed palm height (no deepening), depth 7 mm, 0.9 s sweep in 3 segments, shorter return; notes summarised at the end so they survive the 60-line log.
- Measured pad lowest points (palm frame, mm): I [91,34,136], M [94,8,137], R [95,-18,128], L [88,-58,93]; finger spacing ~26-40 mm, so only one pad fits on a 70 mm lid at a useful lever. Little gives ~93 deg/stroke at ~19 N; 5 strokes in ~13 s. Ring sits 56 mm off-axis (beyond the rim), others far.
- Skill: scratch-stroke (sweep one synergy through a pad's lowest point with the palm fixed, lift-return-lower between sweeps) — reusable for any flat rotating top.

## iter 4 — 3/3 (+478, +485, +476 deg; palm spread 8.8-9.4 mm, yaw 0.35 deg) — iter_3 folder is a killed run (study paused), not counted
- No change: re-ran the iter-2 policy unchanged to confirm the stop rule. Passed again with the same margins (about 115 deg of extra turn, about 5.6 mm under the 15 mm palm limit). Stop rule met (3/3 on two consecutive completed iterations), FINAL = 4.
- Margin to watch: the palm spread of about 9 mm comes from arm/IK compliance while the little pad presses about 19 N into the lid; the turn per stroke is steady at about 93 deg.
