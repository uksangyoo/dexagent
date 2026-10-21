# Hand demos

Each scene reproduces one or two phone clips from `reference/hand_demos` with the UR5e + DG5F-S right hand at the wood table. The `reference` camera matches the phone: about 350 mm over the table, pitched 35° down, with a 42° vertical field of view. Each folder's `ref/` holds stills from its clip.

```bash
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. .venv/bin/python -m dexagent.run tasks/hand_demos/<scene> --camera reference   # RTX film in artifacts/rss/<scene>/
tasks/live tasks/hand_demos/<scene> up                                                      # live page
```

A routine prints `HAND_RESULT <task> <outcome> <metrics>` and writes the full trace to `<scene>/results/`. Routines write only the 26 robot joint targets. They read object poses for planning and scoring, so they are privileged-state scripts.

## Status (Newton 1.7, `.venv`, October 21)

| Scene | Clip | Result | Runs | Key numbers |
| --- | --- | --- | --- | --- |
| `cube_pinch` | 4772 | success | 3/3 | lift 126 mm, turn while held 36°, placed within 1 mm |
| `cube_palm` | 4782 | success | 3/3 | 1.5 s on the open palm, placed at about 40° as in the clip, 8 mm offset |
| `pen_gravity_pivot` | 4774, 4775 | success | 3/3 | swing 90°, pivot 81° in the pinch, caught upright |
| `pen_pickup_pivot` | 4773, 4780 | success | 5/5 | pivot 88°, stands on its tip at 15° tilt, lifted 156 mm |
| `pen_finger_roll` | 4776, 4777 | success | 2/2 | swing ±25° (range 48–51°), roll ±20° (stick-slip), lift 80 mm |
| `pen_writing_grasp` | 4783 | success | 2/2 | 31 mm inked, 86% of the stroke in contact |
| `pen_table_slide` | 4781 | success | 3/3 | 26.5 mm slip through the fingers, regrasped and held 99 mm up for 2 s |
| `scissors` | 4778, 4779 | success | 4/4 | two open/close cycles in the air (13° each), held 130 mm up, set down |

## Where the scenes differ from the clips

- **Prepared starts.** On this hand a pen lying flat on the table cannot be grasped reliably: the 16 x 20 mm fingertip shells meet the 7.5 mm barrel above its equator and ride over it. So three scenes start the pen in a fixture the clip does not show:
  - `pen_gravity_pivot`: on a rest.
  - `pen_finger_roll` and `pen_writing_grasp`: standing in a cup.
  - `pen_table_slide`: in a 60 mm guide block.

  Only `pen_pickup_pivot` picks the pen off the table. It is mirrored relative to the clip.
- **Torsional friction.** The hand's contacts have torsional friction (condim 4). The two pivot scenes give the pen priority in contact (`grip.smooth_pen`) so it can turn freely in the pinch.
- **`pen_pickup_pivot` hold.** After the pivot, the lift is carried by the middle finger clamping the pen against the index, not by a planned grasp. It passed 5 of 5 runs on Newton 1.7 and 2 of 3 on Newton 1.6.
- **`pen_table_slide` regrasp.** A two-finger pinch squeezes the pen out within 1–2 s, so after the slide the hand lets go and closes a three-finger carry grip.
- **`cube_palm` set-down.** The cube is set down from a closed fist; the person returns it to a fingertip pinch first.
- **`scissors` grasp.** The index and middle fingers work against each other in the loops, because the DG5F thumb cannot point down into a 27 mm loop. The scissors are stood up on their blade tips and then hooked, because only about 9 mm of finger fits into each loop with the scissors flat. They open about 13°; the person opens them about 30°. The timing window is narrow: the stand-up must take about 3 s.
- **Nominal objects.** Object sizes, masses and frictions are nominal, not measured.
