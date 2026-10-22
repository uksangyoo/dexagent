"""Collect HOM skill demos via Vive + Manus teleop (self-contained, no rio-hw).

Live page (recommended — you see the hand)::

    tasks/live tasks/hand_demos/cube_pinch up
    # then press Run on \"vive manus collect\"

CLI (same loop; headless unless you already have the live page up)::

    python -m dexagent.teleop.collect tasks/hand_demos/cube_pinch --source hw

Keys: ``c`` clutch, ``n`` new episode, ``s`` save, ``q`` quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="scene folder or scene.py")
    ap.add_argument(
        "--source",
        default="hw",
        choices=("hw", "hardware", "vive", "glove", "mock", "routine"),
        help="hw=Vive+Manus, routine=scripted demos, mock=fake inputs",
    )
    ap.add_argument("--episodes", type=int, default=None, help="stop after N successful saves (default: until q)")
    ap.add_argument("--object", default="cube")
    ap.add_argument("--yaw-cal", type=Path, default=None)
    ap.add_argument("--manus-cal", type=Path, default=None, help="default: auto-find manus_glove_cal.json")
    ap.add_argument("--headless-keys", action="store_true")
    a = ap.parse_args(argv)

    if a.source == "routine":
        from dexagent.teleop.record_routine import record_episodes

        record_episodes(a.scene, episodes=a.episodes or 5, object_name=a.object)
        return

    scene_path = a.scene / "scene.py" if a.scene.is_dir() else a.scene
    scene_dir = scene_path.parent
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(scene_dir))

    from robosimstudio_real2sim_agent.scenefile import run_scene_file

    from dexagent.teleop.loop import run_teleop_collect

    loaded = run_scene_file(scene_path)
    sim, ns = loaded.sim, loaded.ns
    print(
        "NOTE: this CLI is headless. Prefer: tasks/live tasks/hand_demos/cube_pinch up → Run \"vive manus collect\"",
        flush=True,
    )
    try:
        for _ in run_teleop_collect(
            sim,
            None,
            scene_dir=scene_dir,
            source_kind=a.source,
            object_name=a.object,
            start_xyz=ns.get("START"),
            start_quat=ns.get("QUAT"),
            edge=float(ns.get("EDGE", 0.04)),
            squeeze=float(ns.get("SQUEEZE", 0.018)),
            helpers=tuple(ns.get("HELPERS", ())),
            tuck=tuple(ns.get("TUCK", (2, 3, 4))),
            yaw_cal=a.yaw_cal,
            manus_cal=a.manus_cal,
            max_episodes=a.episodes,
            headless_keys=a.headless_keys,
        ):
            pass
    finally:
        if hasattr(sim, "close"):
            sim.close()


if __name__ == "__main__":
    main()
