"""One budgeted iteration of a policy agent.

    python -m dexagent.iterate --workdir bench_runs/agents/cube_pinch/fingertips [--film CAMERA]

The workdir holds `policy.py` and `agent.json` ({"task": "tasks/...", "level": ...,
"max_iters": N}). Each call snapshots policy.py into iter_<k>/, scores it on
three fresh cases (severity 0, 0.5, 1; seeds unique to task, level and k, and
disjoint from the 1000-1009 test set) and appends a line to progress.jsonl.
The iteration counter only goes up, so a case can never be re-run to
cherry-pick. `--film` films the first case listed with `--film-case`
(default: the severity-1 case).
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from .bench import bench, ROOT

SEVERITIES = (0.0, 0.5, 1.0)


def seeds(task, level, k):
    h = int(hashlib.sha1(f"{task}|{level}|{k}".encode()).hexdigest()[:8], 16)
    return [20000 + (h + 7919 * j) % 900000 for j in range(len(SEVERITIES))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--film", help="camera name to film")
    ap.add_argument("--film-case", type=float, default=1.0, help="severity of the filmed case (0, 0.5 or 1)")
    a = ap.parse_args(argv)
    w = Path(a.workdir).resolve()
    cfg = json.loads((w / "agent.json").read_text())
    task, level, cap = cfg["task"], cfg["level"], int(cfg.get("max_iters", 8))
    done = sorted(int(p.name.split("_")[1]) for p in w.glob("iter_*") if p.is_dir() and p.name.split("_")[1].isdigit())
    k = (done[-1] + 1) if done else 1
    # Budget = completed iterations; one killed by a timeout while queued for the GPU does not count.
    completed = len((w / "progress.jsonl").read_text().splitlines()) if (w / "progress.jsonl").exists() else 0
    if completed >= cap:
        print(f"BUDGET EXHAUSTED: {cap} iterations used. Pick your final iteration in FINAL and stop.")
        sys.exit(3)
    it = w / f"iter_{k}"
    it.mkdir()
    shutil.copy(w / "policy.py", it / "policy.py")
    for extra in w.glob("*.py"):
        if extra.name != "policy.py":
            shutil.copy(extra, it / extra.name)
    ss = seeds(Path(task).name, cfg.get("seed_tag", level), k)
    cases = [f"{s}:{v:g}" for s, v in zip(ss, SEVERITIES)]
    order = sorted(range(3), key=lambda i: SEVERITIES[i] != a.film_case)  # filmed case first
    cases = [cases[i] for i in order]
    t0 = time.time()
    s = bench(ROOT / task, str(it / "policy.py"), level, cases, it, film=a.film,
              label=f"agent/{Path(task).name}/{level}/iter{k}")
    row = {"iter": k, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_s": round(time.time() - t0, 1),
           "successes": s["successes"], "n": s["n"],
           "by_severity": {c["case"].split(":")[1]: c["success"] for c in s["cases"]}}
    with open(w / "progress.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"ITERATION {k} ({completed + 1} of {cap} budgeted): {s['successes']}/{s['n']}")
    for c in s["cases"]:
        err = (c.get("error") or "").strip()
        print(f"  severity {c['case'].split(':')[1]:>3} (seed {c['case'].split(':')[0]}): "
              f"{'PASS' if c['success'] else 'fail'}  metrics={json.dumps(c.get('metrics'))}")
        if c.get("perturbation"):
            print(f"      perturbation: {json.dumps(c['perturbation'])}")
        if err:
            print("      ERROR:\n" + "\n".join("        " + l for l in err.splitlines()[-12:]))
        if c.get("sheet"):
            print(f"      contact sheet (12 frames): {c['sheet']}")
    print(f"  per-case details incl. your api call log: {it}/case_*.json")


if __name__ == "__main__":
    main()
