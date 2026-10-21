"""Run a live-page scene file headless: build it, play a routine, film it.

    python -m dexagent.run tasks/hand_demos/cube_pinch --camera closeup

The scene file is executed under the same contract as the live page
(`robosimstudio_real2sim_agent.scenefile.run_scene_file`). Frames come from
`--renderer` (default `tiled`, Newton's collider ray tracer; `cycles` is the
OptiX path tracer used for snapshots).
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--routine", help="name in the file's `routines` (default: the first)")
    ap.add_argument("--camera", help="camera to film (default: none)")
    ap.add_argument("--every", type=int, default=2, help="film every N control steps")
    ap.add_argument("--renderer", default="tiled")
    ap.add_argument("--out", default=str(ROOT / "artifacts/rss"))
    ap.add_argument("--max-steps", type=int, default=5000)
    a = ap.parse_args(argv)
    os.environ["ROBOSIMSTUDIO_RENDERER"] = "dexagent.cycles_renderer.CyclesRenderer" if a.renderer == "cycles" else a.renderer
    scene_path = Path(a.scene).resolve()
    if scene_path.is_dir():
        scene_path = scene_path / "scene.py"
    sys.path.insert(0, str(scene_path.parent))
    sys.path.insert(0, str(ROOT))
    from robosimstudio_real2sim_agent.scenefile import run_scene_file
    t0 = time.time()
    loaded = run_scene_file(scene_path)
    sim, ns = loaded.sim, loaded.ns
    print(f"built {scene_path.name} in {time.time() - t0:.1f} s", flush=True)
    routines = ns.get("routines") or ({"routine": ns["routine"]} if "routine" in ns else {})
    name = a.routine or next(iter(routines))
    gen = routines[name](sim)
    out = Path(a.out) / scene_path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    t0 = time.time()
    for k, _ in enumerate(gen):
        if a.camera and k % a.every == 0:
            frames.append(sim.render(a.camera)["rgb"])
        if k >= a.max_steps:
            print("stopped at --max-steps", flush=True)
            break
    print(f"routine {name!r}: {sim.time:.2f} s simulated in {time.time() - t0:.1f} s", flush=True)
    if frames:
        import imageio.v3 as iio
        import numpy as np
        slug = name.replace(" ", "_")
        iio.imwrite(out / f"{slug}.mp4", np.stack(frames), fps=round(50 / a.every))
        pick = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        rows = [np.concatenate([frames[i] for i in pick[r * 4:r * 4 + 4]], axis=1) for r in range(2)]
        sheet = np.concatenate(rows, axis=0)
        iio.imwrite(out / f"{slug}_sheet.jpg", sheet[::2, ::2])
        print(f"film {out / (slug + '.mp4')}", flush=True)
    sim.close() if hasattr(sim, "close") else None


if __name__ == "__main__":
    main()
