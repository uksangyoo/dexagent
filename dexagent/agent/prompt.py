"""Prompts and message helpers for the coding-agent (dexagent.agent.loop)."""
from __future__ import annotations

import json

from dexagent.agent.tools import api_doc
from dexagent.levels import HINTS

CONTRACT_PRIV = """\
Policy contract
- Define `def run(api):` as a generator. Every motion must be `yield from api.<method>(...)`.
- Control runs at 50 Hz (each yield is 20 ms). The sim time limit is 60 s per case.
- After `run` returns, the bench holds the last targets for 1 s and scores with the task's check.py.
- You may use numpy and scipy. Do not write files, open network connections, or spawn processes.
- Physical contract: only robot joint targets are written. Object poses, velocities and forces are
  read for planning and scoring and must never be written. Privileged reads (`api.object_pose`,
  `api.object_in_palm`, `api.contact_forces`, ...) are allowed and expected.

How to work
- Write a complete policy for the forced representation, then call `run_iterate` to score it.
- Read the returned metrics, full api_log, tip/object/contact traces, errors and the contact sheet.
  Append short notes, then revise.
- Each `run_iterate` consumes one iteration of a fixed budget. Do not burn iterations on tiny no-op edits.
- Prefer simple, closed-loop use of measured object pose over open-loop guesses.
- Call `finish` when you have two consecutive 3/3 results, when the budget is gone, or when further
  progress is unlikely. Omit `final_iter` to let the harness pick the best iteration.
"""

CONTRACT_NO_PRIV = """\
Policy contract
- Define `def run(api):` as a generator. Every motion must be `yield from api.<method>(...)`.
- Control runs at 50 Hz (each yield is 20 ms). The sim time limit is 60 s per case.
- After `run` returns, the bench holds the last targets for 1 s and scores with the task's check.py.
- You may use numpy and scipy. Do not write files, open network connections, or spawn processes.
- Physical contract: only robot joint targets are written. Privileged sim observations are OFF:
  do not call `api.object_pose`, `api.object_in_palm`, `api.object_velocity`, `api.object_joint`,
  or `api.contact_forces`. Use proprioception (`api.measured_palm`, `api.tips`, `api.tips_world`),
  contact sheets, and check.py metrics instead.

How to work
- Write a complete policy for the forced representation, then call `run_iterate` to score it.
- Read the returned metrics, api_log tails, errors and the contact sheet. Append short notes, then revise.
- Each `run_iterate` consumes one iteration of a fixed budget. Do not burn iterations on tiny no-op edits.
- Prefer closed-loop proprioception over open-loop guesses; do not invent object state from sim APIs.
- Call `finish` when you have two consecutive 3/3 results, when the budget is gone, or when further
  progress is unlikely. Omit `final_iter` to let the harness pick the best iteration.
"""


COMPOSE_PRIV = """\
Skill composition (long-horizon task)
- `yield from api.run_skill("<task>")` runs the solved skill library policy for that task, at your level, on
  the live api; it returns when that skill's `run` returns. Skills were learned in isolation on their own
  scenes, so check object state between skills (`api.object_pose`) and add glue motion if needed.
- Skills available are listed below and can be read in skills_lib/ (policy.py, SKILL.md).
- You may also write new motion yourself with this level's api and mix it freely with skills.
"""

COMPOSE_NO_PRIV = """\
Skill composition (long-horizon task)
- `yield from api.run_skill("<task>")` runs the solved skill library policy for that task, at your level, on
  the live api; it returns when that skill's `run` returns. Skills were learned in isolation on their own
  scenes, so add proprioceptive glue motion between skills if needed (no privileged object reads).
- Skills available are listed below and can be read in skills_lib/ (policy.py, SKILL.md).
- You may also write new motion yourself with this level's api and mix it freely with skills.
"""


def system_prompt(level: str, max_iters: int, task_name: str, compose: bool = False,
                  privileged: bool = True) -> str:
    compose_block = COMPOSE_PRIV if privileged else COMPOSE_NO_PRIV
    contract = CONTRACT_PRIV if privileged else CONTRACT_NO_PRIV
    return "\n\n".join([
        *([compose_block] if compose else []),
        f"You are a coding agent writing a robot hand policy for the dexagent task '{task_name}'.",
        "You edit `policy.py` and run a scoring harness. You do not control the robot live.",
        api_doc(level, privileged=privileged),
        contract,
        f"Budget: at most {max_iters} scoring iterations (`run_iterate` calls that succeed).",
        "Answer with tool calls only.",
    ])


DEMO_LABEL = "[human demo keyframes]"
CONTACT_LABEL = "[contact sheet]"


def first_user_message(task_md: str, policy: str, level: str, skills: list | None = None) -> str:
    style = (
        "Style: import numpy as np; use api.note(...) liberally; every motion is "
        "`yield from api.<method>(...)`. Use only the API for this level.\n"
    )
    style += HINTS.get(level, "") + "\n"
    if skills:
        lines = [f"- skills_lib/{sk['id'].replace('/', '__')}/: {sk['task']} at level '{sk['level']}' "
                 f"(held-out {sk.get('test_successes')}/{sk.get('test_n')})" for sk in skills]
        style += ("\nSkill library (read-only; open with read_file). These are solved policies for OTHER tasks, "
                  "possibly written for another level's API, so reuse ideas and helper code, not api calls "
                  "that your level does not have:\n" + "\n".join(lines) + "\n")
    return (
        f"TASK.md:\n{task_md.strip()}\n\n"
        f"{style}\n"
        f"Starting policy.py:\n```python\n{policy.rstrip()}\n```\n\n"
        "Write a real attempt (not just hold), append a short note, then run_iterate."
    )


def user_with_demo(text: str, demo_data_url: str | None) -> dict:
    """First user turn, optionally with a human-demo keyframe strip attached."""
    if not demo_data_url:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "text", "text": DEMO_LABEL},
        {"type": "text", "text": (
            "Human phone demo of the target skill (evenly sampled keyframes). "
            "Match the grasp and motion style. Do not invent joint trajectories from pixels alone; "
            "still use the forced level API and privileged reads when available."
        )},
        {"type": "image_url", "image_url": {"url": demo_data_url}},
    ]}


def _has_demo_image(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(p.get("type") == "text" and DEMO_LABEL in (p.get("text") or "") for p in content)


def result_user_message(result: dict, budget_left: int) -> str:
    """Text part of a follow-up observation after run_iterate (image attached separately)."""
    payload = {k: result[k] for k in result if k != "sheet"}
    hint = (
        "A contact sheet image from the filmed case is attached when available. "
        "Revise the policy, or call finish."
    )
    if result.get("privileged"):
        hint = (
            "Privileged diagnostics are on: per-case api_log is complete and tip/object/contact "
            "traces are included when present. A contact sheet from the filmed case is attached "
            "when available. Revise the policy, or call finish."
        )
    return (
        f"Iterate result ({budget_left} scoring iteration(s) left in the budget):\n"
        f"{json.dumps(payload, indent=1, default=str)}\n\n"
        f"{hint}"
    )


def prune_images(messages: list, keep_turns: int = 2) -> list:
    """Drop iterate contact-sheet images older than `keep_turns`; never drop the human demo."""
    idx = [i for i, m in enumerate(messages)
           if m.get("role") == "user" and isinstance(m.get("content"), list)
           and any(p.get("type") == "image_url" for p in m["content"])
           and not _has_demo_image(m["content"])]
    drop = set(idx[:-keep_turns]) if keep_turns > 0 else set(idx)
    out = []
    for i, m in enumerate(messages):
        if i in drop:
            m = dict(m, content=[{"type": "text", "text": "[earlier contact sheet omitted]"}
                                 if p.get("type") == "image_url" else p for p in m["content"]])
        out.append(m)
    return out


def user_with_sheet(text: str, sheet_data_url: str | None) -> dict:
    if not sheet_data_url:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "text", "text": CONTACT_LABEL},
        {"type": "image_url", "image_url": {"url": sheet_data_url}},
    ]}
