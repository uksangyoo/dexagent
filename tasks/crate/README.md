# Crate

Repeat the human strategy in `reference/crate_human.mp4`. Two UR5e + DG5F-S arms grasp opposite rack rims, lift one rack clear of the stack, invert it by about 180°, place it on the washer entry conveyor, release, and repeat. This is work in progress on RoboSimStudio/Newton, separate from the Drake stack.

```bash
tasks/crate/ctl up --scene "$PWD/tasks/crate/scene.py"     # live viewer/agent server via third-party/robosimstudio
python tasks/crate/prepare_assets.py                        # rebuild the MJCF assets in assets/
```

`ctl` uses `$PYTHON` (a RoboSimStudio environment) and physical GPU 1. `scene.py` defines the live scene, `robot.py` the bimanual embodiment and kinematics, and `policy.py` the video-derived policy graph, which writes joint setpoints only. Snapshots render with the shared Blender Cycles/OptiX plugin in `dexagent/rss/cycles_renderer.py` (`DEXAGENT_RENDER_PYTHON`). `inspect_trace.py` renders recorded states. `task.json`, `policy_graph.json` and `reference/` hold the goal, the graph and the annotated reference frames. Cycle results go to `results/`, which is not tracked.

The generated MJCFs in `assets/` contain absolute paths for this workstation. Re-run `prepare_assets.py` after cloning elsewhere.
