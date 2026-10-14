"""Publish sanitized, measured experiment snapshots into the review site."""

import datetime
import json
import shutil
from pathlib import Path

from dextrain.resource_guard import GIB, available_memory

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "review_site/dist"


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer may be midway through its final line.
    return rows


def progress():
    batch = Path(json.loads((ROOT / "runs/latest_precision_campaign.json").read_text())["batch"])
    jobs = []
    for job in json.loads((batch / "jobs.json").read_text())["jobs"]:
        out = Path(job["output"])
        rows = read_rows(out / "train/metrics.jsonl")
        resources = read_rows(out / "resources.jsonl")
        completed = out / "guard_result.json"
        guard = json.loads(completed.read_text()) if completed.exists() else None
        try:
            cmd = Path(f"/proc/{job['supervisor_pid']}/cmdline").read_bytes().decode()
            alive = guard is None and str(out) in cmd and "dextrain.resource_guard" in cmd
        except FileNotFoundError:
            alive = False
        sample = resources[-1] if resources else {}
        history = rows[::max(1, len(rows)//240)]
        if rows and (not history or history[-1] != rows[-1]): history.append(rows[-1])
        jobs.append({
            "seed": job["seed"], "gpu": job["gpu_index"], "running": alive,
            "latest": rows[-1] if rows else {},
            "history": [{k: row[k] for k in ["env_steps", "reward", "mean_angle_error_deg", "successful_episodes", "terminated_episodes"] if k in row} for row in history],
            "training_successes": sum(r.get("successful_episodes", 0) for r in rows),
            "training_terminations": sum(r.get("terminated_episodes", 0) for r in rows),
            "rss_gib": sample.get("process_rss_bytes", 0)/GIB,
            "gpu_gib": sample.get("gpu_used_bytes", 0)/GIB,
            "output_mib": sample.get("output_bytes", 0)/1024**2,
            "unstable_fraction_max": max((r["unstable_fraction"] for r in rows), default=0),
            "checkpoint": (out / "train/checkpoint.pt").exists(),
            "guard_result": guard,
            "evaluation": json.loads((out / "evaluation/summary.json").read_text()) if (out / "evaluation/summary.json").exists() else None,
        })
        if jobs[-1]["evaluation"]:
            jobs[-1]["evaluation"].pop("checkpoint", None)
    return {"captured_at": datetime.datetime.now(datetime.UTC).isoformat(), "batch": batch.name,
            "jobs": jobs, "host_available_gib": available_memory()/GIB,
            "disk_free_gib": shutil.disk_usage(ROOT).free/GIB,
            "limits": {"rss_gib":8,"gpu_gib":12,"output_mib":2048,"disk_reserve_gib":20,"host_reserve_gib":64,"wall_hours":8},
            "scope": "Training snapshot. Precision held-out evaluation starts after each training job completes."}


def main():
    data = DIST / "data"
    data.mkdir(parents=True, exist_ok=True)
    aggregate = ROOT / "runs/motor_eval_20261012T015042Z/aggregate.json"
    shutil.copyfile(aggregate, data / "motor_evaluation.json")
    evaluation = json.loads(aggregate.read_text())
    state = progress()
    (data / "progress.json").write_text(json.dumps(state) + "\n")
    ids = ["learned_spin", "precision", "zero", "no_contacts", "contact_cycle", "contact_sequence", "free_grasp"]
    clips = [json.loads((data / f"{name}.json").read_text()) for name in ids]
    capture = Path(json.loads((ROOT / "runs/latest_review_capture.json").read_text())["batch"]) / "rollouts"
    for clip in clips:
        clip["captured_at"] = datetime.datetime.fromtimestamp((capture / f"{clip['id']}.json").stat().st_mtime, datetime.UTC).isoformat()
        if clip.get("checkpoint_update") and clip["frames"][0]["time_s"] == 0:
            clip["frames"][0]["forces_available"] = False
        for camera in clip["views"].values():
            mp4 = camera.get("mp4", camera["video"])
            webm = str(Path(mp4).with_suffix(".webm"))
            if (DIST / webm).exists():
                camera.update(video=webm, mp4=mp4)
        (data / f"{clip['id']}.json").write_text(json.dumps(clip) + "\n")
    for name in ids:
        shutil.copyfile(capture / f"{name}.npz", data / f"{name}_states.npz")
    payload = {"progress": state, "evaluation": evaluation, "clips": clips}
    (data / "snapshot.js").write_text("window.REVIEW_DATA = " + json.dumps(payload) + ";\n")
    (DIST / "runtime.js").write_text("window.REVIEW_LIVE = false;\n")
    print(json.dumps({"clips": len(clips), "snapshot": state["captured_at"], "batch": state["batch"]}))


if __name__ == "__main__":
    main()
