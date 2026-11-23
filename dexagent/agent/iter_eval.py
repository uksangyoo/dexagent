"""Held-out evaluation and summary tables for iterate_llm cells."""
from __future__ import annotations

import json
from pathlib import Path

from dexagent.agent.iter_tools import used_api_names
from dexagent.bench import ROOT, bench

# Same fixed test set as bench_runs/tools/final_eval.py (never shown to the editing model).
TEST = ["1000:0", "1001:0.5", "1002:0.5", "1005:1", "1006:1", "1007:1"]


def load_progress(workdir: Path) -> list[dict]:
    p = workdir / "progress.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pick_final(workdir: Path, preferred: int | None = None) -> int | None:
    """Best dev iteration (most successes, then latest). Respects preferred if valid."""
    prog = load_progress(workdir)
    if not prog:
        return None
    iters = {int(r["iter"]) for r in prog}
    if preferred is not None and int(preferred) in iters:
        return int(preferred)
    best = max(prog, key=lambda r: (r["successes"], r["iter"]))
    return int(best["iter"])


def iters_to(prog: list[dict], need: int) -> int | None:
    for r in prog:
        if r["successes"] >= need:
            return int(r["iter"])
    return None


def consecutive_full(prog: list[dict], n: int = 3) -> bool:
    """True if the last two progress rows are both full successes (n/n)."""
    if len(prog) < 2:
        return False
    return all(r["successes"] == n and r["n"] == n for r in prog[-2:])


def run_held_out(workdir: Path, task_rel: str, level: str, final_iter: int, label: str = "") -> dict:
    policy = workdir / f"iter_{final_iter}" / "policy.py"
    out = workdir / "test"
    out.mkdir(parents=True, exist_ok=True)
    s = bench(ROOT / task_rel, str(policy), level, TEST, out,
              label=label or f"llm_iter/{workdir.parent.name}/{workdir.name}/iter{final_iter}")
    return s


def cell_summary(workdir: Path, meta: dict, test: dict | None) -> dict:
    prog = load_progress(workdir)
    final = int((workdir / "FINAL").read_text().strip()) if (workdir / "FINAL").exists() else None
    policy_src = ""
    if final is not None and (workdir / f"iter_{final}" / "policy.py").exists():
        policy_src = (workdir / f"iter_{final}" / "policy.py").read_text()
    elif (workdir / "policy.py").exists():
        policy_src = (workdir / "policy.py").read_text()
    by_sev = {}
    if test:
        for c in test.get("cases") or []:
            sev = str(c["case"]).split(":")[1]
            by_sev.setdefault(sev, [0, 0])
            by_sev[sev][1] += 1
            by_sev[sev][0] += int(bool(c.get("success")))
    return {
        "task": meta.get("task"),
        "level": meta.get("level"),
        "model": meta.get("model"),
        "workdir": str(workdir),
        "dev_iters": len(prog),
        "dev_curve": [r["successes"] / r["n"] for r in prog],
        "iters_to_2": iters_to(prog, 2),
        "iters_to_3": iters_to(prog, 3),
        "final_iter": final,
        "finish_reason": meta.get("finish_reason"),
        "write_rejects": meta.get("write_rejects", 0),
        "llm_calls": meta.get("llm_calls", 0),
        "tokens_in": meta.get("tokens_in", 0),
        "tokens_out": meta.get("tokens_out", 0),
        "llm_cost_usd": meta.get("llm_cost_usd"),
        "llm_wall_s": meta.get("llm_wall_s", 0),
        "bench_wall_s": sum(r.get("wall_s") or 0 for r in prog),
        "policy_lines": policy_src.count("\n") + 1 if policy_src else 0,
        "api_used": used_api_names(policy_src) if policy_src else [],
        "test_successes": None if not test else test.get("successes"),
        "test_n": None if not test else test.get("n"),
        "test_by_severity": by_sev or None,
    }


def print_table(rows: list[dict]) -> None:
    cols = ("level", "dev_iters", "iters_to_2", "iters_to_3", "test", "tokens_in", "tokens_out",
            "llm_cost_usd", "llm_wall_s", "write_rejects")
    display = []
    for r in rows:
        d = dict(r)
        if r.get("test_n"):
            d["test"] = f"{r['test_successes']}/{r['test_n']}"
        else:
            d["test"] = "-"
        if d.get("llm_cost_usd") is not None:
            d["llm_cost_usd"] = f"{d['llm_cost_usd']:.4f}"
        display.append(d)
    w = {c: max(len(c), *(len(str(d.get(c, "-"))) for d in display)) for c in cols}
    line = lambda d: "  ".join(str(d.get(c, "-")).ljust(w[c]) for c in cols)  # noqa: E731
    print(line({c: c for c in cols}))
    for d in display:
        print(line(d))
