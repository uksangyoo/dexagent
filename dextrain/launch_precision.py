"""Launch three guarded precision curricula from frozen local source snapshots."""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .resource_guard import gpu_inventory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=3, type=int, required=True)
    parser.add_argument("--warm-batch", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.5)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=10000)
    args = parser.parse_args()
    if len(set(args.gpus)) != 3 or not 0 < args.hours <= 7.5:
        parser.error("Select three distinct GPUs and at most 7.5 training hours")
    warm = [(args.warm_batch / f"seed_{s}/checkpoint.pt").resolve() for s in range(3)]
    for path in warm:
        if not path.is_file():
            parser.error(f"Missing warm-start checkpoint: {path}")
    inventory = gpu_inventory()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    for gpu in args.gpus:
        if (
            gpu not in inventory
            or inventory[gpu]["uuid"] in apps
            or inventory[gpu]["utilization"] > 5
        ):
            parser.error(f"GPU {gpu} is unavailable; no jobs were launched")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    batch = ROOT / "runs" / f"precision_motor_{stamp}"
    source = batch / "source"
    source.mkdir(parents=True)
    shutil.copytree(
        ROOT / "dextrain", source / "dextrain", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "runs/assets", source / "runs/assets")
    scene = source / "runs/assets/scene.xml"
    external = {Path(n.get("file")) for n in ET.parse(scene).findall(".//mesh[@file]")}
    manifest = {
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
        "external_mesh_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(external)
        },
        "note": "Python and scene are frozen; external installed libraries and mesh assets remain shared",
    }
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for seed, gpu in enumerate(args.gpus):
        out = batch / f"seed_{seed}"
        out.mkdir()
        command = [
            sys.executable,
            "-u",
            "-m",
            "dextrain.resource_guard",
            "--gpu",
            str(gpu),
            "--output",
            str(out),
            "--workdir",
            str(source),
            "--seconds",
            "28800",
            "--",
            sys.executable,
            "-u",
            "-m",
            "dextrain.precision_campaign",
            "--output",
            str(out),
            "--seed",
            str(seed),
            "--warm-start",
            str(warm[seed]),
            "--hours",
            str(args.hours),
            "--num-envs",
            str(args.num_envs),
            "--updates",
            str(args.updates),
        ]
        with (out / "supervisor.log").open("w") as log:
            job = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        jobs.append(
            {
                "seed": seed,
                "gpu_index": gpu,
                "gpu_uuid": inventory[gpu]["uuid"],
                "supervisor_pid": job.pid,
                "output": str(out),
                "command": command,
            }
        )
        (batch / "jobs.json").write_text(json.dumps({"jobs": jobs}, indent=2) + "\n")
    (ROOT / "runs/latest_precision_campaign.json").write_text(
        json.dumps({"batch": str(batch)}) + "\n"
    )
    print(json.dumps({"batch": str(batch), "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
