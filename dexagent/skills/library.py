"""A skill is a solved policy: the FINAL policy.py of a coding-agent cell plus its notes and scores.

Layout (the library root is any directory, e.g. skill_library/):

    <root>/<task>/<level>/policy.py      the FINAL policy (defines run(api))
                          helper *.py    any helper modules the agent wrote beside it
                          SKILL.md       task description + the agent's notes
                          metrics.json   held-out score, cost to solve, solved flag

`save` is called by dexagent.agent.loop when `--library` is given. `resolve` + `expose` give a later
run read-only access to chosen skills (`--skills-from`). `load_policy` is used by composition.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

METRIC_KEYS = ("task", "level", "model", "test_successes", "test_n", "dev_iters", "iters_to_3", "solved_iter",
               "tokens_to_solve", "cost_to_solve_usd", "wall_to_solve_s", "llm_cost_usd", "tokens_in",
               "tokens_out", "n_skills_visible", "policy_lines")


def entry_id(task: str, level: str) -> str:
    return f"{task}/{level}"


def save(root, workdir, summary: dict, solve_threshold: float = 0.5) -> Path:
    """Copy a finished cell's FINAL policy into the library. Returns the entry directory."""
    w = Path(workdir)
    final = int((w / "FINAL").read_text().strip())
    src = w / f"iter_{final}"
    task, level = Path(summary["task"]).name, summary["level"]
    dest = Path(root) / task / level
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in src.glob("*.py"):
        shutil.copy(f, dest / f.name)
    n = summary.get("test_n") or 0
    solved = bool(n and (summary.get("test_successes") or 0) / n >= solve_threshold)
    metrics = {k: summary.get(k) for k in METRIC_KEYS}
    metrics["solved"] = solved
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    notes = (w / "NOTES.md").read_text() if (w / "NOTES.md").exists() else ""
    (dest / "SKILL.md").write_text(f"# Skill: {task} at level {level}\n\n## Task\n\n{(w / 'TASK.md').read_text()}\n"
                                   f"\n## Agent notes\n\n{notes}")
    return dest


def entries(root) -> list[dict]:
    """All library entries (solved or not), sorted by id."""
    out = []
    for m in sorted(Path(root).glob("*/*/metrics.json")):
        d = json.loads(m.read_text())
        task, level = m.parent.parent.name, m.parent.name
        out.append({**d, "id": entry_id(task, level), "task": task, "level": level, "path": str(m.parent)})
    return out


def resolve(root, spec: str, exclude_task: str | None = None) -> list[dict]:
    """Solved entries named by `spec`: 'all', or comma-separated 'task' / 'task/level' items."""
    items = [x.strip() for x in spec.split(",") if x.strip()]
    pool = [e for e in entries(root) if e.get("solved") and e["task"] != exclude_task]
    if "all" in items:
        return pool
    return [e for e in pool if e["id"] in items or e["task"] in items]


def expose(skills: list[dict], dest) -> None:
    """Copy entries into dest/<task>__<level>/ so the agent can read them with read_file."""
    dest = Path(dest)
    for e in skills:
        shutil.copytree(e["path"], dest / e["id"].replace("/", "__"))


def load_policy(root, task: str, level: str | None = None) -> Path:
    """Path of a solved skill's policy.py; picks the best-scoring level when `level` is None."""
    cands = [e for e in entries(root) if e["task"] == task and e.get("solved") and (level in (None, e["level"]))]
    if not cands:
        raise KeyError(f"no solved skill for task={task!r} level={level!r} in {root}")
    best = max(cands, key=lambda e: (e["test_successes"] or 0) / max(1, e["test_n"] or 1))
    return Path(best["path"]) / "policy.py"
