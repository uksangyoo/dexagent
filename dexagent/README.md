# dexagent

One UR5e with a right DG5F-S hand at a wood table, simulated by RoboSimStudio (Newton collision, MuJoCo-Warp solver, on the GPU), and scripted routines that drive it.

| Module | Purpose |
| --- | --- |
| `workcell` | Writes `assets/ur5e_dg5fs_right.xml` (relative mesh paths), registers the embodiment, and provides CPU kinematics: palm IK and fingertip-pad IK in the palm frame |
| `scene` | `table_scene(name)`: the table, the robot, MuJoCo-Warp physics at 1 kHz (50 Hz control), and the `reference` (phone), `overview` and `closeup` cameras |
| `hand` | `Controller`: palm moves, finger poses, pinch grasps, contact forces and the result file. Routines are generators that yield once per 20 ms control step |
| `objects` | Objects from `reference/hand_demos` (the pen) |
| `run` | Headless runner: build a scene folder, play a routine, film it |
| `cycles_renderer`, `cycles_worker` | Blender Cycles/OptiX snapshot renderer for the live page (bpy interpreter in `DEXAGENT_RENDER_PYTHON`) |
| `retarget` | Naive track schema, MediaPipe/HSV `.mov` estimate, palm/tip IK retarget, object-teleport replay (see `retarget/schema.md`) |
| `teleop` | Self-contained Vive + Manus clutch teleop (live page: `vive manus collect`; see `teleop/README.md`) |
| `hom` | Minimal GCF / relative contact-velocity features for demo logging |
| `skills` | Demo recorder, pack N demos → mean HOM skill, open-loop replay |

**Palm frame.** +z runs along the fingers, +x is out of the palm face (fingers flex toward +x), and +y is toward the thumb. `hand_down(yaw)` faces the palm down with the fingers pointing `yaw` from +x; the default `YAW` = 0.7 rad matches the clips. The widest thumb-index pinch (8-70 mm) puts the thumb toward the wrist and the index toward the fingertips: `PINCH_CENTRE`, `PINCH_AXIS`.

**Physical contract.** A routine writes only the 26 joint targets: 6 arm position servos (Menagerie gains, with integral action against gravity sag) and 20 finger position servos (kp 3 N m/rad, kv .12, |tau| <= .65 N m). Object state is read for planning and scoring, never written. Planning reads object poses from the simulator, so the routines are privileged-state scripts, not perception policies.

```bash
export PY=/path/to/robosimstudio/.venv/bin/python     # tasks/live defaults to the physgram checkout
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:third-party/robosimstudio $PY -m dexagent.run tasks/hand_demos/cube_pinch --camera reference
tasks/live tasks/hand_demos/cube_pinch up             # live page: rebuilds scene.py on save, Run buttons per routine
```

`tasks/live <scene-dir> <command>` is `examples/live/ctl` for one scene folder, with its own `run/` directory, so several pages can be up at once (`up --port N`). Saving `scene.py` rebuilds the page. Edits to `dexagent` need `quit` and `up` again, because the page only reloads modules beside the scene file.
