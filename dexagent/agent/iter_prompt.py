"""Prompts and message helpers for the coding-agent (iterate_llm) loop."""
from __future__ import annotations

import json
from pathlib import Path

from dexagent.agent.iter_tools import api_doc

CONTRACT = """\
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
- Read the returned metrics, api_log tails, errors and the contact sheet. Append short notes, then revise.
- Each `run_iterate` consumes one iteration of a fixed budget. Do not burn iterations on tiny no-op edits.
- Prefer simple, closed-loop use of measured object pose over open-loop guesses.
- Call `finish` when you have two consecutive 3/3 results, when the budget is gone, or when further
  progress is unlikely. Omit `final_iter` to let the harness pick the best iteration.
"""


def system_prompt(level: str, max_iters: int, task_name: str) -> str:
    return "\n\n".join([
        f"You are a coding agent writing a robot hand policy for the dexagent task '{task_name}'.",
        "You edit `policy.py` and run a scoring harness. You do not control the robot live.",
        api_doc(level),
        CONTRACT,
        f"Budget: at most {max_iters} scoring iterations (`run_iterate` calls that succeed).",
        "Answer with tool calls only.",
    ])


def first_user_message(task_md: str, policy: str, level: str) -> str:
    style = (
        "Style: import numpy as np; use api.note(...) liberally; every motion is "
        "`yield from api.<method>(...)`. Use only the API for this level.\n"
    )
    if level == "contacts":
        style += "On contacts, prefer approach/grasp/carry/release over inventing joint angles.\n"
    elif level == "joints":
        style += "On joints, set palm pose and 20 finger joint targets (or a per-finger dict).\n"
    elif level == "fingertips":
        style += "On fingertips, command palm-frame pad centres via tips={finger: xyz}.\n"
    elif level == "synergies":
        style += "On synergies, command s (length api.k) in units of one standard deviation.\n"
    return (
        f"TASK.md:\n{task_md.strip()}\n\n"
        f"{style}\n"
        f"Starting policy.py:\n```python\n{policy.rstrip()}\n```\n\n"
        "Write a real attempt (not just hold), append a short note, then run_iterate."
    )


def result_user_message(result: dict, budget_left: int) -> str:
    """Text part of a follow-up observation after run_iterate (image attached separately)."""
    payload = {k: result[k] for k in result if k != "sheet"}
    return (
        f"Iterate result ({budget_left} scoring iteration(s) left in the budget):\n"
        f"{json.dumps(payload, indent=1, default=str)}\n\n"
        "A contact sheet image from the filmed case is attached when available. "
        "Revise the policy, or call finish."
    )


def prune_images(messages: list, keep_turns: int = 2) -> list:
    """Drop image parts from all but the last `keep_turns` user messages that contain images."""
    idx = [i for i, m in enumerate(messages)
           if m.get("role") == "user" and isinstance(m.get("content"), list)
           and any(p.get("type") == "image_url" for p in m["content"])]
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
        {"type": "text", "text": "[contact sheet]"},
        {"type": "image_url", "image_url": {"url": sheet_data_url}},
    ]}
