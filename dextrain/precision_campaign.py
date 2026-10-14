"""Bounded precision training followed automatically by held-out evaluation."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def verify_inputs():
    source = Path(__file__).resolve().parents[1]
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, digest in manifest["external_mesh_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Shared mesh changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.5)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    command = [
        sys.executable,
        "-u",
        "-m",
        "dextrain.train",
        "--task",
        "precision",
        "--num-envs",
        str(args.num_envs),
        "--hours",
        str(args.hours),
        "--updates",
        str(args.updates),
        "--seed",
        str(args.seed),
        "--warm-start",
        str(args.warm_start),
        "--out",
        str(args.output / "train"),
    ]
    subprocess.run(command, check=True)
    verify_inputs()
    from .evaluate_precision import evaluate

    summary = evaluate(args.output / "train/checkpoint.pt", args.output / "evaluation")
    (args.output / "campaign_finished.json").write_text(
        json.dumps(
            {
                "training_finished": True,
                "evaluation_finished": True,
                "checkpoint_update": summary["checkpoint_update"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
