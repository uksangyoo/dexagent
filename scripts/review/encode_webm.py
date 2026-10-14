"""Add broadly supported VP9 previews to the recorded H.264 video assets."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "review_site/dist/assets"


def encode(path):
    target = path.with_suffix(".webm")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-loglevel", "error",
                    "-i", str(path), "-an", "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0",
                    "-row-mt", "1", "-cpu-used", "4", "-threads", "2", str(target)], check=True)
    return {"file": target.name, "bytes": target.stat().st_size}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(encode, sorted(ASSETS.glob("*.mp4"))):
            print(json.dumps(result), flush=True)
