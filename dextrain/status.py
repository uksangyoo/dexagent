"""Read progress and resource evidence for the latest precision or legacy batch."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .resource_guard import GIB, available_memory

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def last_record(path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 32768))
        for line in reversed(f.read().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=Path)
    args = parser.parse_args()
    latest = ROOT / "runs/latest_precision_campaign.json"
    batch = args.batch or (
        Path(read_json(latest)["batch"]) if latest.exists() else ROOT / "runs/current"
    )
    print(
        json.dumps(
            {
                "batch": str(batch.resolve()),
                "host_available_gib": available_memory() / GIB,
                "output_disk_free_gib": shutil.disk_usage(batch).free / GIB,
            }
        )
    )
    for job in read_json(batch / "jobs.json")["jobs"]:
        guarded = "supervisor_pid" in job
        out = Path(job["output"] if guarded else job["out"])
        train = out / "train" if guarded else out
        pid = job["supervisor_pid"] if guarded else job["pid"]
        result = read_json(out / "guard_result.json")
        # Avoid reporting a reused PID as an active experiment.
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
            alive = result is None and str(out) in command and "dextrain." in command
        except FileNotFoundError:
            alive = False
        evaluation = read_json(out / "evaluation/summary.json")
        rates = {}
        if evaluation:
            for mode, angles in evaluation["regimes"].items():
                total = sum(group["episodes"] for group in angles.values())
                rates[mode] = sum(group["successes"] for group in angles.values()) / total
        print(
            json.dumps(
                {
                    "gpu": job["gpu_index"],
                    "supervisor_pid" if guarded else "pid": pid,
                    "alive": alive,
                    "checkpoint": (train / "checkpoint.pt").exists(),
                    "latest": last_record(train / "metrics.jsonl"),
                    "resources": last_record(out / "resources.jsonl"),
                    "guard_result": result,
                    "evaluation_success_rate": rates,
                    "log": str(out / "process.log" if guarded else out / "train.log"),
                }
            )
        )
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv"],
        check=False,
    )


if __name__ == "__main__":
    main()
