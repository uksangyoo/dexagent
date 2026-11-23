"""RQ2: what effect does the action-representation level have?

Sweeps task x level x repeat with the same agent, model and iteration budget, and reports per level:
solve rate (held-out pass rate >= 0.5), held-out pass rate, iterations / tokens / cost / wall time to
the first fully-passing dev iteration, and policy size.

    PYTHONPATH=. .venv/bin/python -m dexagent.experiments.rq2_levels \
        --tasks cube_pinch,cube_palm --levels joints,fingertips,contacts --repeats 3 --model <slug>

Repeat r writes results/<run>/rep<r>/<task>/<level>/ (fresh workdirs, no cross-talk).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from dexagent.agent.loop import run_cell
from dexagent.experiments import common as C
from dexagent.levels import LEVELS


def aggregate(cells: list[dict], threshold: float = 0.5) -> list[dict]:
    """One row per level over all (task, repeat) cells."""
    rows = []
    for lv in LEVELS:
        cs = [c for c in cells if c["level"] == lv]
        if not cs:
            continue
        rate = lambda c: (c["test_successes"] or 0) / c["test_n"] if c.get("test_n") else 0.0  # noqa: E731
        rows.append({
            "level": lv, "cells": len(cs),
            "solve_rate": C.mean([float(C.solved(c, threshold)) for c in cs]),
            "heldout_pass": C.mean([rate(c) for c in cs]),
            "iters_to_solve": C.mean([c.get("solved_iter") for c in cs]),
            "tokens_to_solve": C.mean([c.get("tokens_to_solve") for c in cs]),
            "cost_to_solve_usd": C.mean([c.get("cost_to_solve_usd") for c in cs]),
            "wall_to_solve_s": C.mean([c.get("wall_to_solve_s") for c in cs]),
            "policy_lines": C.mean([c.get("policy_lines") for c in cs]),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(C.ALL_TASKS), help="comma-separated task names")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--repeats", type=int, default=1)
    C.add_agent_args(ap)
    a = ap.parse_args(argv)
    cfg = C.agent_cfg(a)
    root = C.run_root(a, "rq2_levels")
    jobs = []
    for r in range(a.repeats):
        for t in a.tasks.split(","):
            td = C.task_dir(t.strip())
            for lv in C.usable(td, [x.strip() for x in a.levels.split(",")]):
                jobs.append((td, lv, root / f"rep{r}"))
    print(f"[rq2] {len(jobs)} cells -> {root}", flush=True)
    cells = []
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        futs = [ex.submit(run_cell, td, lv, rr, cfg) for td, lv, rr in jobs]
        for f in as_completed(futs):
            cells.append(f.result())
    rows = aggregate(cells)
    cols = ("level", "cells", "solve_rate", "heldout_pass", "iters_to_solve", "tokens_to_solve",
            "cost_to_solve_usd", "wall_to_solve_s", "policy_lines")
    print("\n" + C.table(rows, cols))
    C.write_summary(root, {"experiment": "rq2_levels", "config": cfg, "by_level": rows, "cells": cells})


if __name__ == "__main__":
    main()
