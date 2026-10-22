# Results

Generated 2026-10-22 08:08.

## By level

| Level | Cells | Dev AUC (mean success over 4 iters) | Test success | Test sev 0 / 0.5 / 1 | Tasks with test >= 80% | Median iters to 2/3 | Never 2/3 |
|---|---|---|---|---|---|---|---|
| joints | 12 | 0.271 | 0.486 | 0.417 / 0.458 / 0.528 | 6 | 5.0 | 6 |
| fingertips | 1 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 1 | 1.0 | 0 |
| synergies | 12 | 0.278 | 0.347 | 0.5 / 0.333 / 0.306 | 4 | 5.0 | 8 |
| contacts | 12 | 0.285 | 0.375 | 0.333 / 0.375 / 0.389 | 4 | 5.0 | 7 |
| layered | 12 | 0.34 | 0.431 | 0.417 / 0.417 / 0.444 | 4 | 4.0 | 6 |
| force | 12 | 0.396 | 0.361 | 0.417 / 0.375 / 0.333 | 4 | 5.0 | 7 |
| skills | 6 | 0.269 | 0.278 | 0.333 / 0.25 / 0.278 | 1 | 7.0 | 4 |

Dev success by severity (all dev cases, all iterations), 0 / 0.5 / 1:

- joints: 0.2 / 0.222 / 0.244
- fingertips: 1.0 / 1.0 / 1.0
- synergies: 0.227 / 0.182 / 0.227
- contacts: 0.244 / 0.268 / 0.195
- layered: 0.279 / 0.233 / 0.279
- force: 0.256 / 0.256 / 0.256
- skills: 0.188 / 0.125 / 0.219

Mean dev success per iteration (fresh cases at severities 0, 0.5, 1 each iteration):

- joints: [0.083, 0.194, 0.389, 0.417]
- fingertips: [1.0, 1.0, 1.0, 1.0]
- synergies: [0.139, 0.25, 0.333, 0.389]
- contacts: [0.167, 0.278, 0.306, 0.389]
- layered: [0.167, 0.444, 0.389, 0.361]
- force: [0.333, 0.417, 0.417, 0.417]
- skills: [0.167, 0.167, 0.333, 0.278, 0.333, 0.333]

## By task (held-out test successes; expert out of 10, agents out of 6; dev per iteration in brackets)

| Task | Expert | joints | fingertips | synergies | contacts | layered | force | skills |
|---|---|---|---|---|---|---|---|---|
| baoding | 0/10 | 0/6 (0000) | - | 0/6 (0000) | 0/6 (00) | 0/6 (0000) | 0/6 (0000) | 0/6 (000000) |
| cap_twist | 10/10 | 6/6 (0033) | - | 6/6 (133) | 6/6 (0003) | 6/6 (033) | 6/6 (33) | - |
| card_pickup | 10/10 | 6/6 (033) | - | 5/6 (033) | 6/6 (233) | 6/6 (033) | 6/6 (33) | - |
| cube_palm | 7/10 | 5/6 (0021) | - | 5/6 (1033) | 2/6 (1121) | 4/6 (1221) | 5/6 (33) | - |
| cube_pinch | 10/10 | 6/6 (33) | 6/6 (33) | 6/6 (33) | 6/6 (33) | 6/6 (33) | 6/6 (33) | - |
| cube_reorient | 4/10 | 5/6 (0132) | - | 1/6 (0001) | 1/6 (0001) | 0/6 (0000) | 0/6 (0000) | - |
| pen_finger_roll | 10/10 | 0/6 (0000) | - | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 6/6 (33) |
| pen_gravity_pivot | 8/10 | 1/6 (0000) | - | 0/6 (0000) | 0/6 (000) | 3/6 (0200) | 0/6 (0000) | 4/6 (003223) |
| pen_pickup_pivot | 4/10 | 0/6 (0000) | - | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (000000) |
| pen_table_slide | 10/10 | 6/6 (0003) | - | 2/6 (0001) | 6/6 (033) | 6/6 (233) | 3/6 (033) | - |
| pen_writing_grasp | 7/10 | 0/6 (0000) | - | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (000000) |
| scissors | 7/10 | 0/6 (0000) | - | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (0000) | 0/6 (000010) |

## Layered level: commands per layer in the last iteration (summed over its 3 cases)

| Task | joints | fingertips | synergies | contacts | arm | dev |
|---|---|---|---|---|---|---|
| baoding | 0 | 558 | 0 | 0 | 0 | 0000 |
| cap_twist | 18 | 178 | 0 | 2170 | 3 | 033 |
| card_pickup | 0 | 6 | 0 | 6 | 39 | 033 |
| cube_palm | 25 | 6 | 0 | 32 | 15 | 1221 |
| cube_pinch | 0 | 3 | 0 | 12 | 33 | 33 |
| cube_reorient | 0 | 400 | 0 | 0 | 0 | 0000 |
| pen_finger_roll | 3 | 18 | 0 | 33 | 6 | 0000 |
| pen_gravity_pivot | 6 | 15 | 0 | 0 | 87 | 0200 |
| pen_pickup_pivot | 0 | 240 | 0 | 12 | 36 | 0000 |
| pen_table_slide | 3 | 1750 | 0 | 0 | 0 | 233 |
| pen_writing_grasp | 0 | 14 | 0 | 3 | 1362 | 0000 |
| scissors | 3 | 9 | 0 | 105 | 15 | 0000 |

## Best policy per task (held-out test)

| Task | Best | Test | Policy |
|---|---|---|---|
| baoding | contacts | 0/6 | `bench_runs/agents/baoding/contacts/iter_2/policy.py` |
| cap_twist | force | 6/6 | `bench_runs/agents/cap_twist/force/iter_2/policy.py` |
| card_pickup | force | 6/6 | `bench_runs/agents/card_pickup/force/iter_2/policy.py` |
| cube_palm | force | 5/6 | `bench_runs/agents/cube_palm/force/iter_2/policy.py` |
| cube_pinch | contacts | 6/6 | `bench_runs/agents/cube_pinch/contacts/iter_2/policy.py` |
| cube_reorient | joints | 5/6 | `bench_runs/agents/cube_reorient/joints/iter_5/policy.py` |
| pen_finger_roll | skills | 6/6 | `bench_runs/agents/pen_finger_roll/skills/iter_2/policy.py` |
| pen_gravity_pivot | skills | 4/6 | `bench_runs/agents/pen_gravity_pivot/skills/iter_6/policy.py` |
| pen_pickup_pivot | contacts | 0/6 | `bench_runs/agents/pen_pickup_pivot/contacts/iter_2/policy.py` |
| pen_table_slide | layered | 6/6 | `bench_runs/agents/pen_table_slide/layered/iter_3/policy.py` |
| pen_writing_grasp | contacts | 0/6 | `bench_runs/agents/pen_writing_grasp/contacts/iter_1/policy.py` |
| scissors | skills | 0/6 | `bench_runs/agents/scissors/skills/iter_5/policy.py` |
