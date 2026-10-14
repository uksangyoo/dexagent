"""Launch three independent motor-training seeds on idle, explicitly named GPUs."""

import argparse
import csv
import datetime
import io
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def smi(query):
    return list(
        csv.reader(
            io.StringIO(
                subprocess.check_output(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True
                )
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs=3, type=int, default=[2, 3, 7])
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--updates", type=int, default=10000)
    args = p.parse_args()
    if len(set(args.gpus)) != 3:
        p.error("Select three different GPU indices")
    gpu_rows = smi("--query-gpu=index,uuid,name,utilization.gpu,memory.used")
    inventory = {int(r[0]): r for r in gpu_rows}
    apps = smi("--query-compute-apps=gpu_uuid,pid")
    busy = {r[0].strip() for r in apps if len(r) > 1}
    for gpu in args.gpus:
        if gpu not in inventory:
            p.error(f"GPU {gpu} does not exist")
        row = inventory[gpu]
        if row[1].strip() in busy or int(row[3]) > 5:
            p.error(f"GPU {gpu} has active compute; choose an idle GPU")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    batch = ROOT / "runs" / f"nut_motor_{stamp}"
    batch.mkdir(parents=True)
    shutil.copytree(
        ROOT / "dextrain", batch / "source/dextrain", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "runs/assets", batch / "source/runs/assets")
    jobs = []
    for seed, gpu in enumerate(args.gpus):
        uuid = inventory[gpu][1].strip()
        out = batch / f"seed_{seed}"
        out.mkdir()
        cmd = [
            str(ROOT / ".venv-newton/bin/python"),
            "-u",
            "-m",
            "dextrain.resource_guard",
            "--gpu",
            str(gpu),
            "--output",
            str(out),
            "--workdir",
            str(batch / "source"),
            "--seconds",
            str(args.hours * 3600 + 120),
            "--",
            str(ROOT / ".venv-newton/bin/python"),
            "-u",
            "-m",
            "dextrain.train",
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(seed),
            "--hours",
            str(args.hours),
            "--updates",
            str(args.updates),
            "--out",
            str(out / "train"),
        ]
        with (out / "supervisor.log").open("w") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        jobs.append(
            {
                "gpu_index": gpu,
                "gpu_uuid": uuid,
                "seed": seed,
                "supervisor_pid": proc.pid,
                "output": str(out),
                "command": cmd,
            }
        )
        (batch / "jobs.json").write_text(
            json.dumps(
                {
                    "jobs": jobs,
                    "note": "Motor controller prerequisite training; no RTX or agent benchmark yet",
                    "wall_time_cap_hours_per_job": args.hours,
                },
                indent=2,
            )
            + "\n"
        )
    link = ROOT / "runs/current"
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(batch.name, target_is_directory=True)
    print(json.dumps({"batch": str(batch), "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
