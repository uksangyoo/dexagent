"""The coding-agent planner: an LLM writes policy.py in a forced action representation and
self-improves it from simulated rollouts (dexagent.score iterate).

    export OPENROUTER_API_KEY=...  CUDA_VISIBLE_DEVICES=1
    PYTHONPATH=. .venv/bin/python -m dexagent.agent.loop tasks/hand_demos/cube_pinch \
        --model anthropic/claude-haiku-5.5 --levels contacts --max-iters 4

Fresh workdirs are created under results/<run-id>/<task>/<level>/ so prior policies never leak
in, except through `--skills-from` (RQ1: read-only access to a skill library, see dexagent.skills).
Each cell: edit loop -> FINAL -> held-out test (dexagent.score.HELD_OUT) -> optional library save.
Privileged object state is allowed inside the authored policy; the forced `level` is the only
action language.

`--stub script.json` replaces the model with scripted tool calls (see dexagent.agent.stub) for
offline validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from dexagent.levels import LEVELS as ALL_LEVELS, NEEDS_SYNERGY  # noqa: E402
STUB_POLICY = "import numpy as np\n\n\ndef run(api):\n    yield from api.hold(0.5)\n"


def make_workdir(run_root: Path, task_dir: Path, level: str, max_iters: int, skills: list | None = None,
                 cfg_skills_root: str | None = None) -> Path:
    w = run_root / task_dir.name / level
    w.mkdir(parents=True, exist_ok=False)
    shutil.copy(task_dir / "TASK.md", w / "TASK.md")
    scene = (task_dir / "scene.py").read_text()
    cams = ["reference", "overview", "closeup"] + re.findall(r'CameraSpec\(\s*name="([^"]+)"', scene)
    (w / "agent.json").write_text(json.dumps({
        "task": str(task_dir.relative_to(ROOT)),
        "level": level,
        "max_iters": max_iters,
        "cameras": sorted(set(cams)),
        **({"skills_root": cfg_skills_root, "compose": True} if cfg_skills_root else {}),
    }, indent=1))
    (w / "policy.py").write_text(STUB_POLICY)
    (w / "NOTES.md").write_text(f"# {task_dir.name} at level {level}\n\n")
    if skills:
        from dexagent.skills import library
        library.expose(skills, w / "skills_lib")
    return w


def make_client(cfg: dict, log_path: Path):
    if cfg.get("stub"):
        from dexagent.agent.stub import ScriptedClient
        return ScriptedClient(json.loads(Path(cfg["stub"]).read_text()))
    from dexagent.agent.openrouter import OpenRouter
    return OpenRouter(cfg["model"], temperature=cfg.get("temperature", 0.2),
                      max_tokens=int(cfg.get("max_tokens", 8192)), log_path=str(log_path))


def run_cell(task_dir: Path, level: str, run_root: Path, cfg: dict) -> dict:
    """Drive one (task, level) cell to FINAL + held-out test. Returns a cell_summary dict."""
    from dexagent.agent import eval as E
    from dexagent.agent import prompt as P
    from dexagent.agent import tools as T

    max_iters = int(cfg["max_iters"])
    skills = cfg.get("skills")  # list of library entries visible to the agent (RQ1)
    compose = bool(cfg.get("compose"))  # RQ3: policy may call api.run_skill on library skills
    w = make_workdir(run_root, task_dir, level, max_iters, skills, cfg.get("library") if compose else None)
    film = cfg.get("film") or "closeup,overview"
    wb = T.Workbench(w, level, film=film, timeout_s=int(cfg.get("timeout_s", 3600)),
                     extra_api=("run_skill",) if compose else ())
    client = make_client(cfg, w / "llm_iter.jsonl")

    meta = {"task": str(task_dir.relative_to(ROOT)), "level": level, "model": cfg.get("model") or "stub",
            "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "llm_cost_usd": 0.0, "llm_wall_s": 0.0,
            "write_rejects": 0, "finish_reason": None, "skills": skills}
    messages = [
        {"role": "system", "content": P.system_prompt(level, max_iters, task_dir.name, compose)},
        {"role": "user", "content": P.first_user_message(
            (w / "TASK.md").read_text(), (w / "policy.py").read_text(), level, skills)},
    ]
    tools = T.TOOL_SPECS
    max_llm_turns = int(cfg.get("max_llm_turns", max_iters * 8 + 10))
    keep_sheets = int(cfg.get("keep_sheets", 2))

    def budget_left() -> int:
        done = len(E.load_progress(w))
        return max(0, max_iters - done)

    try:
        for _ in range(max_llm_turns):
            if wb.finished:
                break
            if budget_left() == 0 and not wb.finished:
                # No scoring left; nudge the model to finish, or finish ourselves after one more turn.
                pass
            t0 = time.time()
            reply = client.chat(P.prune_images(messages, keep_sheets), tools)
            meta["llm_calls"] += 1
            meta["llm_wall_s"] = round(meta["llm_wall_s"] + time.time() - t0, 2)
            u = reply.get("usage") or {}
            meta["tokens_in"] += int(u.get("prompt_tokens") or 0)
            meta["tokens_out"] += int(u.get("completion_tokens") or 0)
            cost = u.get("cost")
            if cost is None:
                cost = (u.get("cost_details") or {}).get("upstream_inference_cost")
            if cost is not None:
                meta["llm_cost_usd"] = round(meta["llm_cost_usd"] + float(cost), 6)

            msg = reply["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({"role": "user", "content": "Reply with tool calls only. Call finish if you are done."})
                continue
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})

            for tc in calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    result, sheet = {"error": f"arguments are not valid JSON: {e}"}, None
                else:
                    # If the budget is gone, refuse further run_iterate without finishing.
                    if name == "run_iterate" and budget_left() == 0:
                        result, sheet = {"budget_exhausted": True,
                                         "error": "scoring budget exhausted; call finish"}, None
                    else:
                        result, sheet = wb.dispatch(name, args)
                meta["write_rejects"] = wb.stats["write_rejects"]
                # Tool message (text). Sheet goes on the next user turn after run_iterate.
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, default=str)[:60_000]})
                if name == "run_iterate" and result.get("iter"):
                    E.log_usage(w, result, meta)
                if name == "run_iterate" and sheet is not None and not result.get("error"):
                    url = T.sheet_data_url(sheet)
                    messages.append(P.user_with_sheet(
                        P.result_user_message(result, budget_left()), url))
                if name == "run_iterate" and E.consecutive_full(E.load_progress(w)):
                    # Two consecutive 3/3: stop early like the human NOTES convention.
                    wb.finish(reason="two consecutive 3/3")
                    break
                if wb.finished:
                    break
            if wb.finished:
                break
            # If budget is gone and model didn't finish, stop after this turn.
            if budget_left() == 0 and not any(tc["function"]["name"] == "finish" for tc in calls):
                # Give one more model turn unless we already asked; force finish next.
                if meta["llm_calls"] > max_iters * 6:
                    wb.finish(reason="scoring budget exhausted")
                    break
        else:
            if not wb.finished:
                wb.finish(reason="llm turn budget exhausted")
    except Exception as e:  # noqa: BLE001
        meta["finish_reason"] = f"error: {type(e).__name__}: {e}"
        if not wb.finished:
            wb.finish(reason=meta["finish_reason"])

    meta["finish_reason"] = wb.finish_reason or meta.get("finish_reason")
    meta["write_rejects"] = wb.stats["write_rejects"]

    final = E.pick_final(w, wb.final_iter)
    test = None
    if final is not None:
        (w / "FINAL").write_text(str(final) + "\n")
        try:
            test = E.run_held_out(w, str(task_dir.relative_to(ROOT)), level, final,
                                  label=f"{run_root.name}/{task_dir.name}/{level}/iter{final}",
                                  skills_root=cfg.get("library") if compose else None)
        except Exception as e:  # noqa: BLE001
            test = {"successes": 0, "n": 0, "error": f"{type(e).__name__}: {e}", "cases": []}
    else:
        meta["finish_reason"] = (meta.get("finish_reason") or "") + " (no scored iteration)"

    summary = E.cell_summary(w, meta, test)
    (w / "cell_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if cfg.get("library") and final is not None and not compose:
        from dexagent.skills import library
        saved = library.save(cfg["library"], w, summary, solve_threshold=float(cfg.get("solve_threshold", 0.5)))
        summary["library_entry"] = str(saved)
    if summary.get("test_n"):
        test_s = f"{summary['test_successes']}/{summary['test_n']}"
    else:
        test_s = "-"
    print(f"[done] {task_dir.name}/{level}: final={final} "
          f"dev={summary['dev_iters']} test={test_s}", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="scene folder, e.g. tasks/hand_demos/cube_pinch")
    ap.add_argument("--model", help="OpenRouter model slug")
    ap.add_argument("--levels", default=",".join(ALL_LEVELS),
                    help="comma-separated subset of: " + ", ".join(ALL_LEVELS))
    ap.add_argument("--max-iters", type=int, default=4, help="scoring iterations per level (passed to agent.json)")
    ap.add_argument("--film", default="closeup,overview",
                    help="cameras for iterate's filmed case (side-by-side contact sheet)")
    ap.add_argument("--parallel", type=int, default=1, help="how many levels to drive at once")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout-s", type=int, default=3600, help="wall-clock limit per iterate subprocess")
    ap.add_argument("--stub", help="scripted turns JSON instead of a model")
    ap.add_argument("--out", help="run root (default results/<task>-<timestamp>)")
    ap.add_argument("--skills-from", default="",
                    help="library entries the agent may read, e.g. 'cube_pinch/layered,cube_palm/layered' or 'all' "
                         "(RQ1; library root from --library)")
    ap.add_argument("--library", help="skill library root; solved cells are saved here (default: none)")
    ap.add_argument("--compose", action="store_true",
                    help="RQ3: task is a long-horizon task; the policy may call api.run_skill on --skills-from entries")
    ap.add_argument("--solve-threshold", type=float, default=0.5, help="held-out pass rate that counts as solved")
    a = ap.parse_args(argv)

    task_dir = Path(a.task).resolve()
    for f in ("scene.py", "check.py", "TASK.md"):
        if not (task_dir / f).exists():
            sys.exit(f"{task_dir / f} not found")
    levels = [x.strip() for x in a.levels.split(",") if x.strip()]
    bad = [x for x in levels if x not in ALL_LEVELS]
    if bad:
        sys.exit(f"unknown level(s) {bad}; choose from {ALL_LEVELS}")
    if not a.stub:
        if not a.model:
            sys.exit("--model is required (or --stub)")
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("OPENROUTER_API_KEY is not set")

    sys.path.insert(0, str(ROOT))
    from dexagent.agent import eval as E
    from dexagent.score import RUNS

    # Drop levels that need a missing synergy asset.
    kept = []
    for lv in levels:
        if lv in NEEDS_SYNERGY and not (ROOT / "dexagent/assets/synergies" / f"{task_dir.name}.npz").exists():
            print(f"[skip] {lv}: no synergies asset for {task_dir.name}", flush=True)
            continue
        kept.append(lv)
    levels = kept
    if not levels:
        sys.exit("no levels left to run")

    skills = []
    if a.skills_from:
        if not a.library:
            sys.exit("--skills-from needs --library")
        from dexagent.skills import library
        skills = library.resolve(a.library, a.skills_from, exclude_task=task_dir.name)
        print(f"[skills] {len(skills)} visible: {[s['id'] for s in skills]}", flush=True)

    run_root = Path(a.out) if a.out else RUNS / f"{task_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = {"model": a.model, "max_iters": a.max_iters, "film": a.film, "temperature": a.temperature,
           "max_tokens": a.max_tokens, "timeout_s": a.timeout_s,
           "library": str(Path(a.library).resolve()) if a.library else None,
           "solve_threshold": a.solve_threshold,
           "skills": skills, "compose": a.compose,
           "stub": str(Path(a.stub).resolve()) if a.stub else None}
    (run_root / "config.json").write_text(json.dumps({**cfg, "task": str(task_dir.relative_to(ROOT)),
                                                      "levels": levels}, indent=1))
    print(f"[run] root={run_root} model={a.model or 'stub'} levels={levels}", flush=True)

    rows = []
    if a.parallel <= 1 or len(levels) == 1:
        for lv in levels:
            print(f"[cell] {task_dir.name}/{lv}", flush=True)
            rows.append(run_cell(task_dir, lv, run_root, cfg))
    else:
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futs = {ex.submit(run_cell, task_dir, lv, run_root, cfg): lv for lv in levels}
            for fut in as_completed(futs):
                rows.append(fut.result())
        # Stable order by level name for the table.
        order = {lv: i for i, lv in enumerate(ALL_LEVELS)}
        rows.sort(key=lambda r: order.get(r["level"], 99))

    summary = {"task": task_dir.name, "model": a.model or "stub", "run_root": str(run_root),
               "max_iters": a.max_iters, "film": a.film, "cells": rows}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print()
    E.print_table(rows)
    print(f"\n-> {run_root}/summary.json")


if __name__ == "__main__":
    main()
