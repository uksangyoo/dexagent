"""Human demo keyframes for the coding agent: sample a task's CLIP into one contact sheet.

Resolves `CLIP` / `CLIPS` from `scene.py` (without importing the sim), evenly samples a fixed
frame budget across the clip duration, and tiles them into `demo_sheet.jpg`.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_FRAMES = 8
MAX_SIDE = 1024


class DemoError(RuntimeError):
    pass


def clip_paths(task_dir: Path) -> list[Path]:
    """Return existing demo video paths declared by the task's scene.py (CLIP / CLIPS)."""
    task_dir = Path(task_dir).resolve()
    scene = task_dir / "scene.py"
    if not scene.exists():
        raise DemoError(f"no scene.py in {task_dir}")
    return _clips_from_scene(scene)


def _clips_from_scene(scene: Path, _seen: set | None = None) -> list[Path]:
    _seen = _seen or set()
    scene = scene.resolve()
    if scene in _seen:
        return []
    _seen.add(scene)
    text = scene.read_text()
    out: list[Path] = []

    # Composed tasks often load another scene via runpy.run_path(.../scene.py).
    for m in re.finditer(r'parents\[(\d+)\]\s*/\s*"([^"]+/scene\.py)"', text):
        depth, rel = int(m.group(1)), m.group(2)
        other = (scene.parents[depth] / rel).resolve()
        if other.exists():
            out.extend(_clips_from_scene(other, _seen))

    # Literal reference/...mov|mp4 paths
    for rel in re.findall(r'["\'](reference/[^"\']+\.(?:mov|mp4))["\']', text):
        p = (ROOT / rel).resolve()
        if p not in out:
            out.append(p)

    # f"reference/hand_demos/IMG_{n}.mov" for n in ("a", "b")
    for m in re.finditer(
        r'f["\']reference/hand_demos/IMG_\{n\}\.(mov|mp4)["\']\s*for n in\s*\(([^)]*)\)',
        text,
    ):
        ext, body = m.group(1), m.group(2)
        for name in re.findall(r'["\']([^"\']+)["\']', body):
            p = (ROOT / f"reference/hand_demos/IMG_{name}.{ext}").resolve()
            if p not in out:
                out.append(p)

    missing = [p for p in out if not p.is_file()]
    if missing and not any(p.is_file() for p in out):
        raise DemoError(f"demo clip(s) missing: {missing}")
    return [p for p in out if p.is_file()]


def extract_demo_sheet(clip: Path, out_jpg: Path, n: int = N_FRAMES, max_side: int = MAX_SIDE) -> Path:
    """Evenly sample `n` frames from `clip` and write a single tiled JPEG to `out_jpg`."""
    import numpy as np
    import imageio.v3 as iio

    clip = Path(clip).resolve()
    if not clip.is_file():
        raise DemoError(f"demo clip not found: {clip}")

    meta = iio.immeta(clip) or {}
    duration = float(meta.get("duration") or 0.0)
    fps = float(meta.get("fps") or 30.0)
    if duration <= 0:
        # Fall back: count frames (bounded) via iterator.
        count = 0
        for _ in iio.imiter(clip):
            count += 1
            if count > 100_000:
                break
        if count <= 0:
            raise DemoError(f"could not read frames from {clip}")
        duration = count / max(fps, 1e-6)

    times = np.linspace(0.0, max(duration - 1.0 / max(fps, 1e-6), 0.0), int(n))
    # Sequential decode: pick frames nearest to target times (avoids random seek issues).
    targets = set(int(round(t * fps)) for t in times)
    targets = {max(0, i) for i in targets}
    want = sorted(targets)
    picked: dict[int, np.ndarray] = {}
    for i, frame in enumerate(iio.imiter(clip)):
        if i in targets:
            picked[i] = np.asarray(frame)
            if len(picked) >= len(want):
                break
        if want and i > want[-1]:
            break
    if not picked:
        raise DemoError(f"no frames decoded from {clip}")

    # Map each desired time to the nearest decoded index.
    have = sorted(picked)
    frames = []
    for t in times:
        idx = int(round(t * fps))
        nearest = min(have, key=lambda j: abs(j - idx))
        frames.append(picked[nearest])

    sheet = _tile(frames, cols=4)
    sheet = _downscale(sheet, max_side)
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_jpg, sheet, extension=".jpg", quality=85)
    return out_jpg


def prepare_demo_sheet(task_dir: Path, workdir: Path, n: int = N_FRAMES) -> tuple[Path, Path]:
    """Resolve the first demo clip for `task_dir` and write `workdir/demo_sheet.jpg`.

    Returns `(demo_sheet, clip_path)`.
    """
    clips = clip_paths(task_dir)
    if not clips:
        raise DemoError(
            f"no CLIP/CLIPS demo video found for {task_dir}; "
            "declare CLIP in scene.py or pass --no-demo"
        )
    clip = clips[0]
    sheet = extract_demo_sheet(clip, Path(workdir) / "demo_sheet.jpg", n=n)
    return sheet, clip


def _tile(frames: list, cols: int = 4):
    import numpy as np
    frames = [np.asarray(f) for f in frames]
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    cols = max(1, min(cols, len(frames)))
    rows = []
    for r in range(0, len(frames), cols):
        chunk = frames[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def _downscale(img, max_side: int):
    import numpy as np
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    # Integer stride downsample (matches sheet_data_url).
    step = max(1, int(round(1 / scale)))
    return np.asarray(img)[::step, ::step]
