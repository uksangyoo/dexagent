"""RQ4 (sim-side proxy): do agent-written skills hold up away from the conditions they were tuned on?

There is no real robot here. This measures the part of the sim2real gap we can probe in simulation:
how a FINAL policy degrades as conditions move away from its dev distribution. Conditions per skill:

  nominal    severity 0 (the scene as authored)
  in_dist    severity 0.5-1 (the range the agent saw while improving)
  extrap     severity 1.5 and 2 (object pose/size/mass beyond the declared range)
  noise      in_dist cases plus white joint-target noise (--noise rad)
  latency    in_dist cases plus a command delay (--latency control steps of 20 ms)

The gap is (in_dist pass rate) - (condition pass rate). Compare across levels (RQ2) and between agent
skills and the scripted expert routine (`--expert`, a reference with the same perturbations).

    PYTHONPATH=. .venv/bin/python -m dexagent.experiments.rq4_robust \
        --library skill_library --tasks cube_pinch,cube_palm --levels layered,contacts --n 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dexagent.experiments import common as C
from dexagent.score import RUNS, score
from dexagent.skills import library

SEEDS = (3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007)  # disjoint from dev (>= 20000) and held-out (1000s)


def conditions(n, noise, latency):
    seeds = SEEDS[:n]
    def cases(sev):
        return [f"{s}:{sev[i % len(sev)]:g}" for i, s in enumerate(seeds)]
    return {
        "nominal": (cases((0,)), None),
        "in_dist": (cases((0.5, 1.0)), None),
        "extrap": (cases((1.5, 2.0)), None),
        "noise": (cases((0.5, 1.0)), {"noise": noise}),
        "latency": (cases((0.5, 1.0)), {"latency": latency}),
    }


def evaluate(td, policy, level, root, conds, label):
    out = {}
    for name, (cases, robust) in conds.items():
        s = score(td, policy, level, cases, root / name, label=f"{label}/{name}", robust=robust)
        out[name] = round(s["successes"] / s["n"], 3)
    out["gap_extrap"] = round(out["in_dist"] - out["extrap"], 3)
    out["gap_noise"] = round(out["in_dist"] - out["noise"], 3)
    out["gap_latency"] = round(out["in_dist"] - out["latency"], 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--tasks", help="comma-separated tasks (default: every solved library entry)")
    ap.add_argument("--levels", help="comma-separated levels (default: all in library)")
    ap.add_argument("--n", type=int, default=6, help="cases per condition (<= 8)")
    ap.add_argument("--noise", type=float, default=0.02, help="joint-target noise std [rad]")
    ap.add_argument("--latency", type=int, default=3, help="command delay in control steps")
    ap.add_argument("--expert", action="store_true", help="also evaluate each task's expert routine")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import time
    root = Path(a.out) if a.out else RUNS / f"rq4_robust-{time.strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    conds = conditions(min(a.n, len(SEEDS)), a.noise, a.latency)
    tasks = set(a.tasks.split(",")) if a.tasks else None
    levels = set(a.levels.split(",")) if a.levels else None
    rows = []
    for e in library.entries(a.library):
        if not e.get("solved") or (tasks and e["task"] not in tasks) or (levels and e["level"] not in levels):
            continue
        td = C.task_dir(e["task"])
        r = evaluate(td, str(Path(e["path"]) / "policy.py"), e["level"], root / e["task"] / e["level"], conds,
                     f"rq4/{e['task']}/{e['level']}")
        rows.append({"task": e["task"], "level": e["level"], "who": "agent", **r})
    if a.expert:
        for t in sorted(tasks or {e["task"] for e in library.entries(a.library)}):
            td = C.task_dir(t)
            rows.append({"task": t, "level": "expert", "who": "expert",
                         **evaluate(td, "expert", "expert", root / t / "expert", conds, f"rq4/{t}/expert")})
    cols = ("task", "level", "who", "nominal", "in_dist", "extrap", "noise", "latency",
            "gap_extrap", "gap_noise", "gap_latency")
    print("\n" + C.table(rows, cols))
    C.write_summary(root, {"experiment": "rq4_robust", "noise": a.noise, "latency": a.latency, "rows": rows})


if __name__ == "__main__":
    main()
