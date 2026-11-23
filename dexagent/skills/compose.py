"""Skill composition (RQ3): run library skills as subroutines of a long-horizon policy.

`attach(api, library_root)` gives a policy `api.run_skill(task, level=None)`: a generator that runs the
solved library policy for `task` (at the api's own level unless `level` is given) on the live api and
returns that skill's return value. A composed policy is then ordinary code:

    def run(api):
        yield from api.run_skill("cube_pinch")
        yield from api.hold(0.3)
        yield from api.run_skill("cube_palm")

Skills see the same api object, so they work whenever the composed task uses the same level and the
same object names as the skill's own task. Skill code is the agent-written policy, run unmodified.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dexagent.skills import library


def _load(path: Path):
    sys.path.insert(0, str(path.parent))  # the skill's own helper modules
    try:
        spec = importlib.util.spec_from_file_location(f"skill_{path.parent.parent.name}_{path.parent.name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def attach(api, library_root) -> None:
    def run_skill(task, level=None):
        level = level or api.level
        path = library.load_policy(library_root, task, level)
        api._log.append(f"{api.t:6.2f} skill {task} ({path.parent.name})")
        result = yield from _load(path).run(api)
        api._log.append(f"{api.t:6.2f} skill {task} done")
        return result

    api.run_skill = run_skill


CHAIN_TEMPLATE = "def run(api):\n{body}"


def chain_source(tasks, pause_s: float = 0.3) -> str:
    """policy.py source that runs `tasks` in order with a short hold between (the no-LLM baseline)."""
    lines = []
    for i, t in enumerate(tasks):
        if i:
            lines.append(f"    yield from api.hold({pause_s})")
        lines.append(f"    yield from api.run_skill({t!r})")
    return CHAIN_TEMPLATE.format(body="\n".join(lines) + "\n")
