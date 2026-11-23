"""RQ3: can skills learned in isolation be strung into a long-horizon policy (code as policy)?

Three arms on one composed task (default tasks/composed/pinch_then_palm), all at one level:

  chain     no LLM: `api.run_skill(a); hold; api.run_skill(b)` over the library skills
  agent     the coding agent writes the composed policy with api.run_skill available and the skills readable
            (glue code, checks, recovery), and self-improves from rollouts
  scratch   the coding agent learns the long-horizon task directly with no library

The library must already hold solved skills for the component tasks at --level (build it with
`python -m dexagent.agent.loop <task> --library skill_library ...` or rq1_transfer).

    PYTHONPATH=. .venv/bin/python -m dexagent.experiments.rq3_compose \
        --library skill_library --skills cube_pinch,cube_palm --level layered --model <slug>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dexagent.agent.loop import run_cell
from dexagent.experiments import common as C
from dexagent.score import HELD_OUT, score
from dexagent.skills import library
from dexagent.skills.compose import chain_source


def run_chain(td, level, root, lib_root, skills) -> dict:
    out = root / "chain" / td.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy.py").write_text(chain_source(skills))
    s = score(td, str(out / "policy.py"), level, HELD_OUT, out / "test", label=f"chain/{td.name}/{level}",
              skills_root=lib_root)
    return {"arm": "chain", "task": td.name, "level": level, "test_successes": s["successes"], "test_n": s["n"],
            "cases": s["cases"], "llm_cost_usd": 0.0, "tokens_to_solve": 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="composed/pinch_then_palm")
    ap.add_argument("--skills", required=True, help="comma-separated component tasks, in execution order")
    ap.add_argument("--library", required=True, help="skill library root")
    ap.add_argument("--level", default="layered")
    ap.add_argument("--arms", default="chain,agent,scratch")
    C.add_agent_args(ap)
    a = ap.parse_args(argv)
    td = C.task_dir(a.task)
    skills = [s.strip() for s in a.skills.split(",")]
    lib_root = str(Path(a.library).resolve())
    have = {e["task"] for e in library.entries(lib_root) if e.get("solved") and e["level"] == a.level}
    missing = [s for s in skills if s not in have]
    arms = a.arms.split(",")
    if missing and ("chain" in arms or "agent" in arms):
        raise SystemExit(f"library has no solved {a.level} skill for {missing}")
    root = C.run_root(a, "rq3_compose")
    results = {}
    if "chain" in arms:
        results["chain"] = run_chain(td, a.level, root, lib_root, skills)
    if "agent" in arms or "scratch" in arms:
        cfg = C.agent_cfg(a)
    if "agent" in arms:
        entries = library.resolve(lib_root, ",".join(skills))
        entries = [e for e in entries if e["level"] == a.level]
        results["agent"] = run_cell(td, a.level, root / "agent",
                                    dict(cfg, library=lib_root, compose=True, skills=entries))
    if "scratch" in arms:
        results["scratch"] = run_cell(td, a.level, root / "scratch", dict(cfg))
    rows = [{"arm": k, "heldout": f"{v.get('test_successes')}/{v.get('test_n')}",
             "solved_iter": v.get("solved_iter"), "tokens_to_solve": v.get("tokens_to_solve"),
             "cost_usd": v.get("llm_cost_usd")} for k, v in results.items()]
    print("\n" + C.table(rows, ("arm", "heldout", "solved_iter", "tokens_to_solve", "cost_usd")))
    C.write_summary(root, {"experiment": "rq3_compose", "task": td.name, "skills": skills, "level": a.level,
                           "arms": results})


if __name__ == "__main__":
    main()
