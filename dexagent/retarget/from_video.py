"""Estimate poses from a .mov and replay them in an existing scene twin.

    PYTHONPATH=.:third-party/robosimstudio \\
      .venv/bin/python -m dexagent.retarget.from_video \\
      --scene tasks/hand_demos/cube_pinch \\
      --clip reference/hand_demos/IMG_4772_cube_pinch_lift.mov

Writes ``tracks/from_video.json`` under the scene folder, films a naive
retargeted replay from the ``reference`` camera, and a time-aligned
``retarget_compare.mp4`` (human + hand overlay | sim).

``--object`` defaults from the scene folder (cube_pinch→cube, scissors→scissors).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="scene folder (must already be a twin)")
    ap.add_argument("--clip", required=True, help="phone .mov / video path")
    ap.add_argument(
        "--object",
        default=None,
        help="object name in the scene (default: inferred from the scene folder)",
    )
    ap.add_argument("--tracks-out", type=Path, help="where to write tracks.json")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth video frame")
    ap.add_argument("--max-frames", type=int, default=None, help="cap estimated frames (debug)")
    ap.add_argument("--camera", default="reference", help="camera to film (empty string to skip)")
    ap.add_argument("--every", type=int, default=2, help="film every N control steps")
    ap.add_argument("--out", default=str(ROOT / "artifacts/rss"), help="film output root")
    ap.add_argument("--renderer", default="tiled")
    ap.add_argument("--skip-estimate", action="store_true", help="reuse existing tracks-out")
    ap.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="playback speed vs the source clip (0.25 = 4× slower; more IK steps)",
    )
    a = ap.parse_args(argv)
    if a.speed <= 0.0:
        ap.error(f"--speed must be positive, got {a.speed}")

    scene_dir = Path(a.scene).resolve()
    if scene_dir.is_file():
        scene_dir = scene_dir.parent
    scene_path = scene_dir / "scene.py"
    clip = Path(a.clip).resolve()
    tracks_out = Path(a.tracks_out) if a.tracks_out else (scene_dir / "tracks" / "from_video.json")

    from dexagent.retarget.scene_objects import infer_object_name

    object_name = a.object or infer_object_name(scene_dir)

    os.environ.setdefault("ROBOSIMSTUDIO_RENDERER", a.renderer)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(scene_dir))
    from robosimstudio_real2sim_agent.scenefile import run_scene_file
    from dexagent.retarget.replay import play_track

    # Build first so non-cube objects can seed a static teleport from the twin pose.
    t0 = time.time()
    loaded = run_scene_file(scene_path)
    sim = loaded.sim
    print(f"built {scene_path.name} in {time.time() - t0:.1f} s (object={object_name!r})", flush=True)
    if object_name not in sim.names():
        raise KeyError(
            f"object {object_name!r} not in scene; available: {sorted(sim.names())}"
        )

    pose0 = np.asarray(sim[object_name].pose, float)
    xyz0, quat0 = pose0[:3].copy(), pose0[3:].copy()

    if not a.skip_estimate:
        from dexagent.retarget.estimate import estimate_and_save

        print(f"estimating {clip} → {tracks_out}", flush=True)
        track = estimate_and_save(
            clip,
            tracks_out,
            object_name=object_name,
            object_xyz0=xyz0,
            object_quat0=quat0,
            detect_color=(object_name == "cube"),
            stride=a.stride,
            max_frames=a.max_frames,
        )
        print(
            f"track: {len(track.frames)} frames, hand_hits={track.meta.get('hand_hits')}, "
            f"palm_up_hits={track.meta.get('palm_up_hits')}, "
            f"object_hits={track.meta.get('object_hits')}",
            flush=True,
        )
    else:
        from dexagent.retarget.tracks import load_tracks

        track = load_tracks(tracks_out)
        print(f"loaded {tracks_out} ({len(track.frames)} frames)", flush=True)

    from dexagent.retarget.tracks import with_speed

    realtime_dur = track.duration
    track = with_speed(track, a.speed)
    print(
        f"speed={a.speed:g}× → track {realtime_dur:.2f}s realtime plays over {track.duration:.2f}s",
        flush=True,
    )

    frames = []
    camera = a.camera or None
    sim_fps = max(1, round(50 / a.every))
    t0 = time.time()
    play_track.last = None
    for k, _ in enumerate(play_track(sim, track, objects=(object_name,))):
        if camera and k % a.every == 0:
            frames.append(sim.render(camera)["rgb"])
    print(f"replayed in {time.time() - t0:.1f} s (sim time {sim.time:.2f} s)", flush=True)
    if play_track.last is not None:
        _, outcome, metrics = play_track.last
        print(f"outcome {outcome} {metrics}", flush=True)

    if frames:
        import imageio.v3 as iio
        from dexagent.retarget.compare import build_compare

        out = Path(a.out) / scene_dir.name
        out.mkdir(parents=True, exist_ok=True)
        path = out / "retarget_from_video.mp4"
        iio.imwrite(path, np.stack(frames), fps=sim_fps)
        print(f"film {path}", flush=True)
        compare_path = out / "retarget_compare.mp4"
        build_compare(
            clip, track, frames, out_path=compare_path, fps=float(sim_fps), speed=a.speed
        )
        print(f"compare {compare_path}", flush=True)
    if hasattr(sim, "close"):
        sim.close()


if __name__ == "__main__":
    main()
