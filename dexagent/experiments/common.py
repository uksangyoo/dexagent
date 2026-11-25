"""Shared helpers for the experiment scripts: CLI flags, agent config, tables, summaries."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dexagent.score import ROOT, RUNS

ALL_TASKS = (
    "hand_demos/cube_pinch", "hand_demos/cube_palm", "hand_demos/pen_gravity_pivot",
    "hand_demos/pen_pickup_pivot", "hand_demos/pen_finger_roll", "hand_demos/pen_writing_grasp",
    "hand_demos/pen_table_slide", "hand_demos/scissors",
    "inhand/cap_twist", "inhand/card_pickup", "inhand/baoding", "inhand/cube_reorient",
)


def task_dir(name: str) -> Path:
    """'cube_pinch', 'hand_demos/cube_pinch' or 'tasks/hand_demos/cube_pinch' -> absolute scene folder."""
    for cand in (ROOT / name, ROOT / "tasks" / name, *(ROOT / "tasks").glob(f"*/{name}")):
        if (cand / "scene.py").exists():
            return cand.resolve()
    sys.exit(f"unknown task {name!r}")


def add_agent_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", help="OpenRouter model slug (or use --stub)")
    ap.add_argument("--stub", help="scripted tool calls JSON instead of a model (offline validation)")
    ap.add_argument("--max-iters", type=int, default=4, help="scoring iterations per cell")
    ap.add_argument("--film", default="closeup,wrist")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--parallel", type=int, default=1, help="cells to drive at once")
    ap.add_argument("--out", help="run root (default results/<experiment>-<timestamp>)")
    ap.add_argument("--privileged", action=argparse.BooleanOptionalAction, default=True,
                    help="allow privileged object/contact reads and full iterate diagnostics (default: on)")
    ap.add_argument("--demo", action=argparse.BooleanOptionalAction, default=True,
                    help="attach auto-sampled keyframes from the task CLIP on the first turn (default: on)")
    ap.add_argument("--max-cost-usd", type=float, default=2.0,
                    help="stop each cell once cumulative LLM spend reaches this USD (default: 2; 0 disables)")


def agent_cfg(a, **extra) -> dict:
    from dexagent.score import require_cuda
    require_cuda()
    if not a.stub:
        if not a.model:
            sys.exit("--model is required (or --stub)")
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("OPENROUTER_API_KEY is not set")
    return {"model": a.model, "max_iters": a.max_iters, "film": a.film, "temperature": a.temperature,
            "max_tokens": a.max_tokens, "timeout_s": a.timeout_s, "privileged": a.privileged,
            "demo": a.demo, "max_cost_usd": a.max_cost_usd,
            "stub": str(Path(a.stub).resolve()) if a.stub else None, **extra}


def run_root(a, name: str) -> Path:
    root = Path(a.out) if a.out else RUNS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def usable(task: Path, levels, warn=True) -> list[str]:
    """Drop levels whose synergy basis is missing for this task."""
    from dexagent.levels import NEEDS_SYNERGY
    keep = []
    for lv in levels:
        if lv in NEEDS_SYNERGY and not (ROOT / "dexagent/assets/synergies" / f"{task.name}.npz").exists():
            if warn:
                print(f"[skip] {task.name}/{lv}: no synergy basis", flush=True)
            continue
        keep.append(lv)
    return keep


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def solved(cell: dict, threshold: float = 0.5) -> bool:
    n = cell.get("test_n") or 0
    return bool(n and (cell.get("test_successes") or 0) / n >= threshold)


def table(rows: list[dict], cols: tuple[str, ...]) -> str:
    cell = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    w = {c: max(len(c), *(len(cell(r.get(c))) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols)]
    lines += ["  ".join(cell(r.get(c)).ljust(w[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def write_summary(root: Path, summary: dict) -> None:
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n-> {root}/summary.json", flush=True)
