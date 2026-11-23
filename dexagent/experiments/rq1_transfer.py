"""RQ1: does learning one skill help learn future skills?

A curriculum is an ordered list of tasks learned at one level. Two arms see the same tasks, model and
budget; they differ only in what the agent may read:

  scratch   no skill library: every task is learned from nothing
  library   a growing library: task i may read the solved skills of tasks 0..i-1 (--skills-from all)

Compare, per task and in total, iterations / tokens / cost / wall time to first solve and the held-out
pass rate. "Future skills learned more efficiently" = lower cost-to-solve in the library arm;
"future skills can be harder" = pass rate on later (harder) tasks in the order.

    PYTHONPATH=. .venv/bin/python -m dexagent.experiments.rq1_transfer \
        --order cube_pinch,cube_palm,pen_finger_roll --level layered --model <slug>

Run `--order` in the order you want to test; rerun with a permuted order to separate "earlier skill
helps" from "task is easy".
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dexagent.agent.loop import run_cell
from dexagent.experiments import common as C
from dexagent.skills import library

KEYS = ("solved_iter", "tokens_to_solve", "cost_to_solve_usd", "wall_to_solve_s", "dev_iters", "llm_cost_usd")


def compare(scratch: list[dict], lib: list[dict]) -> list[dict]:
    rows = []
    for s, l in zip(scratch, lib):
        row = {"task": Path(s["task"]).name, "n_skills_visible": l.get("n_skills_visible")}
        for k in ("solved_iter", "tokens_to_solve", "cost_to_solve_usd", "wall_to_solve_s"):
            row[f"scratch_{k}"], row[f"library_{k}"] = s.get(k), l.get(k)
        for arm, c in (("scratch", s), ("library", l)):
            row[f"{arm}_heldout"] = f"{c.get('test_successes')}/{c.get('test_n')}"
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True, help="comma-separated tasks, learned in this order")
    ap.add_argument("--level", default="layered")
    ap.add_argument("--arms", default="scratch,library")
    ap.add_argument("--solve-threshold", type=float, default=0.5)
    C.add_agent_args(ap)
    a = ap.parse_args(argv)
    cfg = C.agent_cfg(a, solve_threshold=a.solve_threshold)
    root = C.run_root(a, "rq1_transfer")
    tasks = [C.task_dir(t.strip()) for t in a.order.split(",")]
    tasks = [t for t in tasks if C.usable(t, [a.level])]
    results = {}
    for arm in a.arms.split(","):
        lib_root = root / arm / "skill_library"
        cells = []
        for td in tasks:
            cell_cfg = dict(cfg, library=str(lib_root))  # both arms record skills; only "library" reads them
            if arm == "library":
                cell_cfg["skills"] = library.resolve(lib_root, "all", exclude_task=td.name)
            cells.append(run_cell(td, a.level, root / arm, cell_cfg))
        results[arm] = cells
    if len(results) == 2:
        rows = compare(results["scratch"], results["library"])
        cols = ("task", "n_skills_visible", "scratch_solved_iter", "library_solved_iter",
                "scratch_cost_to_solve_usd", "library_cost_to_solve_usd",
                "scratch_tokens_to_solve", "library_tokens_to_solve",
                "scratch_heldout", "library_heldout")
        print("\n" + C.table(rows, cols))
    else:
        rows = None
    C.write_summary(root, {"experiment": "rq1_transfer", "order": [t.name for t in tasks], "level": a.level,
                           "config": cfg, "comparison": rows, "arms": results})


if __name__ == "__main__":
    main()
