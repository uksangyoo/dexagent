"""Evaluate every agent's FINAL policy (and the expert) on the fixed test set."""
import json, sys, fcntl, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dexagent.bench import bench
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TEST = "1000:0,1001:0.5,1002:0.5,1005:1,1006:1,1007:1".split(",")
jobs = []
for w in sorted((ROOT / "bench_runs/agents").glob("*/*")):
    cfg = json.loads((w / "agent.json").read_text())
    iters = sorted(int(p.name.split("_")[1]) for p in w.glob("iter_*") if p.is_dir() and p.name.split("_")[1].isdigit())
    if not iters:
        continue
    if not (w / "FINAL").exists():
        continue  # still running
    if (w / "FINAL").stat().st_mtime < float(os.environ.get("FINAL_AFTER", "0")):
        continue  # left to an earlier evaluator
    k = int((w / "FINAL").read_text().strip())
    if k not in iters:  # no valid choice: best dev, latest on ties
        prog = [json.loads(l) for l in (w / "progress.jsonl").read_text().splitlines()]
        k = max(prog, key=lambda r: (r["successes"], r["iter"]))["iter"]
    out = w / "test"
    if (out / "summary.json").exists():
        continue
    jobs.append((ROOT / cfg["task"], str(w / f"iter_{k}" / "policy.py"), cfg["level"], out, f"test/{w.parent.name}/{w.name}/iter{k}"))
print(len(jobs), "evaluations")
def go(j):
    j[3].mkdir(parents=True, exist_ok=True)
    lk = open(j[3] / ".lock", "w")
    try:  # one evaluator per cell; several final_eval instances may run at once
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (j[3] / "summary.json").exists():
        return
    s = bench(j[0], j[1], j[2], TEST, j[3], label=j[4])
    print(j[4], f"{s['successes']}/{s['n']}", flush=True)
with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(go, jobs))
