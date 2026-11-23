"""Run an OpenRouter model on one scene, once per hand-command representation, and compare.

    export OPENROUTER_API_KEY=...
    export CUDA_VISIBLE_DEVICES=1
    PYTHONPATH=. .venv/bin/python -m dexagent.agent.run_agent tasks/hand_demos/cube_pinch \
        --model anthropic/claude-sonnet-4.5 --levels joints,fingertips,synergies,contacts,layered,force \
        --case 1:0 --camera closeup --max-calls 30

The objective is the scene's TASK.md, verbatim, unless --objective is given. Every level gets the same model,
objective, observations, case, call budget and sim-time limit; only the tool set differs. Each level runs through
`dexagent.bench` (so the scorer is the scene's check.py, on true simulator state). Output, under --out:

    <level>/case_<seed>_<sev>.json   bench record, with `llm_usage` (agent_calls, tool_calls, param_dims, tokens, ...)
    <level>/transcript_*.json        every turn: the model's text, tool calls and their results
    <level>/llm_*.jsonl              raw OpenRouter exchanges (images elided)
    <level>/film_*.mp4               optional; --film closeup,overview writes a side-by-side video
    comparison.json                  one row per level and case; the table is printed at the end

`--stub script.json` replaces the model with scripted turns (see dexagent.agent.stub) to test the loop offline.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = Path(__file__).resolve().with_name("llm_policy.py")
ALL_LEVELS = ("joints", "fingertips", "synergies", "contacts", "layered", "force")
NEEDS_SYNERGY = ("synergies", "layered", "force")


_POSE_NOTE = (
    "\n\n**Observation note (overrides any claim above that a pose is readable).** "
    "The agent does NOT receive object poses, in-palm poses, velocities or object joint angles. "
    "It sees only a camera image, proprioception and contact forces. Infer object location from the image."
)


def objective_text(task_dir, override):
    if override:
        text = override
    else:
        md = task_dir / "TASK.md"
        if not md.exists():
            sys.exit(f"{md} not found; pass --objective")
        text = md.read_text()
    # Soften TASK.md lines that claim privileged state the agent never gets.
    softened = []
    for line in text.splitlines():
        low = line.lower()
        if "pose is readable" in low or "readable at any time" in low:
            softened.append(line + " (not provided to the agent in this harness — use the camera image instead)")
        else:
            softened.append(line)
    return "\n".join(softened) + _POSE_NOTE


def row_for(level, case, llm):
    llm = llm or {}
    return {"level": level, "case": case["case"], "success": bool(case["success"]),
            "agent_calls": llm.get("agent_calls"), "tool_calls": llm.get("tool_calls"), "param_dims": llm.get("param_dims"),
            "rejected": llm.get("rejected_calls"), "ik_failures": llm.get("ik_failures"), "tool_errors": llm.get("tool_errors"),
            "coerced_args": llm.get("coerced_args"), "tokens_in": llm.get("tokens_in"), "tokens_out": llm.get("tokens_out"),
            "llm_wall_s": llm.get("llm_wall_s"), "sim_s": case.get("sim_s"),
            "stopped": llm.get("stopped"), "metrics": case.get("metrics"),
            "error": ((case.get("error") or "").strip().splitlines() or [None])[-1]}


def print_table(rows):
    cols = ("level", "case", "success", "agent_calls", "tool_calls", "param_dims", "rejected", "ik_failures", "sim_s", "stopped")
    w = {c: max(len(c), *(len(str(r.get(c))) for r in rows)) for c in cols}
    line = lambda r: "  ".join(str(r.get(c) if r.get(c) is not None else "-").ljust(w[c]) for c in cols)  # noqa: E731
    print(line({c: c for c in cols}))
    for r in rows:
        print(line(dict(r, success="PASS" if r["success"] else "fail")))
    levels = list(dict.fromkeys(r["level"] for r in rows))
    print("\nper level (successes / cases, mean agent calls):")
    for lv in levels:
        rs = [r for r in rows if r["level"] == lv]
        calls = [r["agent_calls"] for r in rs if r["agent_calls"] is not None]
        print(f"  {lv:<11} {sum(r['success'] for r in rs)}/{len(rs)}   calls {sum(calls) / len(calls):.1f}" if calls
              else f"  {lv:<11} {sum(r['success'] for r in rs)}/{len(rs)}   calls -")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="scene folder, e.g. tasks/hand_demos/cube_pinch")
    ap.add_argument("--model", help="OpenRouter model slug")
    ap.add_argument("--levels", default=",".join(ALL_LEVELS), help="comma-separated subset of: " + ", ".join(ALL_LEVELS))
    ap.add_argument("--objective", help="objective text; default is the scene's TASK.md")
    ap.add_argument("--case", default="1:0", help="seed:severity, comma-separated for several (default 1:0)")
    ap.add_argument("--camera", default="closeup,overview",
                    help="comma-separated cameras the model sees each turn (scene cameras, e.g. closeup,overview,reference)")
    ap.add_argument("--max-calls", type=int, default=30, help="agent-call budget per run")
    ap.add_argument("--max-tools-per-call", type=int, default=6)
    ap.add_argument("--max-sim-s", type=float, default=60.0)
    ap.add_argument("--keep-images", type=int, default=2,
                    help="how many latest observation turns keep their images in context (each turn has every --camera)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="completion token budget per OpenRouter call (raise for reasoning models)")
    ap.add_argument("--film", help="comma-separated cameras to film on the first case, concatenated left-to-right "
                                   "into one mp4 (e.g. closeup,overview). Omit to skip filming.")
    ap.add_argument("--timeout-s", type=int, default=3600, help="wall-clock limit per case process")
    ap.add_argument("--stub", help="scripted turns JSON instead of a model (offline loop test)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    task_dir = Path(a.task).resolve()
    for f in ("scene.py", "check.py"):
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
    from dexagent import levels as L
    from dexagent.agent import tools as T
    from dexagent.bench import RUNS, bench

    # Every offered tool argument must exist in its api method; fail loudly on schema drift.
    classes = {"joints": L.Joints, "fingertips": L.Fingertips, "synergies": L.Synergies, "contacts": L.Contacts,
               "layered": L.Layered, "force": L.Force}
    drift = [p for lv in levels for p in T.check_against_api(lv, classes[lv])]
    if drift:
        sys.exit("tool schema drifted from levels.py:\n  " + "\n  ".join(drift))

    out = Path(a.out) if a.out else RUNS / "llm" / f"{task_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    objective = objective_text(task_dir, a.objective)
    cases = [c.strip() for c in a.case.split(",")]
    (out / "objective.txt").write_text(objective)

    rows = []
    for lv in levels:
        if lv in NEEDS_SYNERGY and not (ROOT / "dexagent/assets/synergies" / f"{task_dir.name}.npz").exists():
            print(f"[skip] {lv}: no dexagent/assets/synergies/{task_dir.name}.npz for this scene", flush=True)
            continue
        lvl_out = out / lv
        lvl_out.mkdir(parents=True, exist_ok=True)
        cameras = [c.strip() for c in a.camera.split(",") if c.strip()]
        cfg = {"model": a.model, "objective": objective, "cameras": cameras, "camera": a.camera,
               "max_calls": a.max_calls, "max_tools_per_call": a.max_tools_per_call, "keep_images": a.keep_images,
               "temperature": a.temperature, "max_tokens": a.max_tokens,
               "log_dir": str(lvl_out), "stub": str(Path(a.stub).resolve()) if a.stub else None}
        (lvl_out / "config.json").write_text(json.dumps(cfg, indent=1))
        # The bench runs each case in a subprocess that inherits this environment.
        os.environ["DEXAGENT_LLM_CONFIG"] = str(lvl_out / "config.json")
        print(f"[run] level={lv} model={a.model or 'stub'} cases={cases}", flush=True)
        s = bench(task_dir, str(POLICY), lv, cases, lvl_out, film=a.film, label=f"llm/{task_dir.name}/{lv}",
                  max_sim_s=a.max_sim_s, timeout_s=a.timeout_s)
        for c in s["cases"]:
            rows.append(row_for(lv, c, c.get("llm_usage")))
        print(f"[done] level={lv} {s['successes']}/{s['n']}", flush=True)

    (out / "comparison.json").write_text(json.dumps({"task": task_dir.name, "model": a.model or "stub", "objective_file": "objective.txt",
                                                     "max_calls": a.max_calls, "cases": cases, "rows": rows}, indent=1, default=str))
    print()
    if rows:
        print_table(rows)
    print(f"\n-> {out}/comparison.json")


if __name__ == "__main__":
    main()
