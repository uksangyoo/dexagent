"""Record palm / tip / object poses from a scripted scene routine into tracks.json.

    PYTHONPATH=.:third-party/robosimstudio \\
      .venv/bin/python -m dexagent.retarget.record_from_routine \\
      tasks/hand_demos/cube_pinch --out tasks/hand_demos/cube_pinch/tracks/synthetic.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from dexagent.retarget.tracks import CONTROL_HZ, Track, TrackFrame, save_tracks
from dexagent.workcell import Kinematics

ROOT = Path(__file__).resolve().parents[2]


def _pose7(sim, name):
    p = np.asarray(sim[name].pose, float)
    return p[:3].copy(), p[3:].copy()


def record(sim, routine_gen, *, objects=(), every=1, max_steps=5000) -> Track:
    """Step ``routine_gen`` and sample kinematics + named object poses."""
    kin = Kinematics()
    handle = sim.scene.handles.arms["right"]
    arm_idx = list(handle.arm_coords)
    hand_idx = list(handle.gripper_coords)
    frames: list[TrackFrame] = []
    t0 = float(sim.time)
    for k, _ in enumerate(routine_gen):
        if k % every == 0:
            q = sim.state.joint_q.numpy()
            arm = q[arm_idx]
            hand = q[hand_idx]
            palm_xyz, palm_R = kin.palm_pose(arm)
            kin.set(arm=arm, hand=hand)
            tips = np.array([kin.data.site_xpos[s].copy() for s in kin.tip_site])
            objs = {name: _pose7(sim, name) for name in objects}
            frames.append(
                TrackFrame(
                    t=float(sim.time) - t0,
                    palm_xyz=palm_xyz,
                    palm_R=palm_R,
                    tips=tips,
                    objects=objs,
                )
            )
        if k >= max_steps:
            break
    return Track(
        frames=frames,
        fps=CONTROL_HZ / every,
        meta={"source": "scripted_routine", "every": every, "objects": list(objects)},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene folder or scene.py")
    ap.add_argument("--routine", help="routine name (default: first non-retarget)")
    ap.add_argument("--objects", default="cube", help="comma-separated object names to record")
    ap.add_argument("--out", type=Path, help="output tracks.json path")
    ap.add_argument("--every", type=int, default=1, help="record every N control steps")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--renderer", default="tiled")
    a = ap.parse_args(argv)

    scene_path = Path(a.scene).resolve()
    if scene_path.is_dir():
        scene_dir = scene_path
        scene_path = scene_path / "scene.py"
    else:
        scene_dir = scene_path.parent
    out = a.out or (scene_dir / "tracks" / "synthetic.json")
    objects = tuple(n.strip() for n in a.objects.split(",") if n.strip())

    os.environ.setdefault("ROBOSIMSTUDIO_RENDERER", a.renderer)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(scene_dir))
    from robosimstudio_real2sim_agent.scenefile import run_scene_file

    t0 = time.time()
    loaded = run_scene_file(scene_path)
    sim, ns = loaded.sim, loaded.ns
    print(f"built {scene_path.name} in {time.time() - t0:.1f} s", flush=True)
    routines = ns.get("routines") or ({"routine": ns["routine"]} if "routine" in ns else {})
    name = a.routine
    if name is None:
        name = next(n for n in routines if "retarget" not in n.lower())
    print(f"recording routine {name!r} → {out}", flush=True)
    track = record(sim, routines[name](sim), objects=objects, every=a.every, max_steps=a.max_steps)
    track.clip = str(ns.get("CLIP") or ns.get("CLIPS") or "")
    save_tracks(track, out)
    print(f"wrote {len(track.frames)} frames ({track.duration:.2f} s) to {out}", flush=True)
    if hasattr(sim, "close"):
        sim.close()


if __name__ == "__main__":
    main()
