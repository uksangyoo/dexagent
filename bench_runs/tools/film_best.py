"""Film the best held-out policy per task on one test case it passed.

    .venv/bin/python bench_runs/tools/film_best.py [camera]   -> bench_runs/best/<task>/film_*.mp4, sheet_*.jpg

Best = highest held-out success over all agent levels (ties: fewer dev iterations
to first 3/3, then level order). Run analyze.py first (reads results.json).
"""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dexagent.bench import bench

CAM = sys.argv[1] if len(sys.argv) > 1 else "reference"
res = json.loads((ROOT / "bench_runs/results.json").read_text())
jobs = []
def hardest_pass(d):
    cases = [json.loads(p.read_text()) for p in sorted(d.glob("case_*.json"))]
    passed = [c for c in cases if c["success"]]
    return max(passed, key=lambda c: float(c["case"].split(":")[1])) if passed else None


task_dirs = {p.parent.name: p.parent for p in ROOT.glob("tasks/*/*/scene.py")}
for task, b in sorted(res["best"].items()):
    out = ROOT / "bench_runs/best" / task / CAM
    if any(out.glob("film_*.mp4")):
        continue
    w = ROOT / b["policy"].rsplit("/iter_", 1)[0]
    pick = hardest_pass(w / "test")
    if pick is not None:
        cfg = json.loads((w / "agent.json").read_text())
        jobs.append((ROOT / cfg["task"], str(ROOT / b["policy"]), cfg["level"], [pick["case"]], out, f"best/{task}/{cfg['level']}"))
        continue
    pick = hardest_pass(ROOT / "bench_runs/expert" / task / "test")  # no agent passed: show the scripted routine
    if pick is not None:
        jobs.append((task_dirs[task], "expert", "expert", [pick["case"]], out, f"best/{task}/expert"))
print(len(jobs), "films", flush=True)


def go(j):
    s = bench(j[0], j[1], j[2], j[3], j[4], film=CAM, label=j[5])
    print(j[5], j[3], f"{s['successes']}/{s['n']}", flush=True)


with ThreadPoolExecutor(6) as ex:
    list(ex.map(go, jobs))
