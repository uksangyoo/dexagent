"""Cheap CPU-only tests: cases, robustness hook, level surface, skill library, composition, scoring seeds."""
import types

import numpy as np
import pytest

from dexagent.agent import prompt, tools
from dexagent.experiments import common as C
from dexagent.experiments.rq2_levels import aggregate
from dexagent.levels import HINTS, LEVELS, NEEDS_SYNERGY
from dexagent.score import HELD_OUT, SEVERITIES, dev_seeds
from dexagent.sim.cases import Case, Robust, parse
from dexagent.skills import compose, library


def test_case_is_deterministic_and_severity_scaled():
    a, b = parse("7:1"), parse("7:0")
    assert a.u(0.02) != 0.0 and b.u(0.02) == 0.0
    assert parse("7:1").u(0.02) == parse("7:1").u(0.02)
    # draws are consumed regardless of severity, so the sequence is aligned
    c, d = Case(3, 1.0), Case(3, 0.5)
    assert c.u(1.0) == pytest.approx(2 * d.u(1.0))


def test_robust_latency_and_noise():
    r = Robust({"latency": 2}, seed=0)
    idx = [0, 1]
    outs = [r.apply(np.array([float(k), 0.0]), idx)[0] for k in range(5)]
    assert outs == [0.0, 0.0, 0.0, 1.0, 2.0]
    n = Robust({"noise": 0.1}, seed=0).apply(np.zeros(3), [0, 1])
    assert n[2] == 0.0 and np.any(n[:2] != 0.0)


def test_every_level_has_surface_and_hint():
    assert set(HINTS) == set(LEVELS)
    for lv in LEVELS:
        names = tools.allowed_api_names(lv)
        assert {"hold", "object_pose", "note"} <= names
        assert "run_skill" not in names and "run_skill" in tools.allowed_api_names(lv, ("run_skill",))
        no_priv = tools.allowed_api_names(lv, privileged=False)
        assert tools.PRIVILEGED_API.isdisjoint(no_priv)
        assert {"hold", "tips", "note"} <= no_priv
    assert set(NEEDS_SYNERGY) <= set(LEVELS)


def test_scan_policy_rules():
    ok = "def run(api):\n    yield from api.hold(0.1)\n"
    assert tools.scan_policy(ok, "joints") == []
    assert tools.scan_policy("def run(api):\n    yield from api.nope()\n", "joints")
    comp = "def run(api):\n    yield from api.run_skill('cube_pinch')\n"
    assert tools.scan_policy(comp, "layered")
    assert tools.scan_policy(comp, "layered", ("run_skill",)) == []


def test_privileged_scan_and_api_doc():
    priv = "def run(api):\n    api.object_pose('cube')\n    api.contact_forces('cube')\n    yield from api.hold(0.1)\n"
    proprio = "def run(api):\n    api.tips()\n    yield from api.hold(0.1)\n"
    assert tools.scan_policy(priv, "joints", privileged=True) == []
    assert tools.scan_policy(proprio, "joints", privileged=False) == []
    problems = tools.scan_policy(priv, "joints", privileged=False)
    assert any("privileged" in p for p in problems)
    assert "- api.object_pose" in tools.api_doc("joints", privileged=True)
    doc_off = tools.api_doc("joints", privileged=False)
    assert "- api.object_pose" not in doc_off and "Privileged observations: off" in doc_off


def test_dev_seeds_disjoint_from_held_out():
    held = {int(c.split(":")[0]) for c in HELD_OUT}
    for k in range(1, 20):
        s = dev_seeds("cube_pinch", "layered", k)
        assert len(s) == len(SEVERITIES) and not held & set(s) and min(s) >= 20000
    assert dev_seeds("a", "x", 1) != dev_seeds("b", "x", 1)


def _fake_cell(tmp_path, task="cube_pinch", level="layered", passed=6):
    w = tmp_path / "w" / task / level
    (w / "iter_2").mkdir(parents=True)
    (w / "iter_2" / "policy.py").write_text("def run(api):\n    api.marker = 1\n    yield from api.hold(0.1)\n")
    (w / "FINAL").write_text("2\n")
    (w / "TASK.md").write_text("# t\n")
    (w / "NOTES.md").write_text("notes\n")
    return w, {"task": f"tasks/hand_demos/{task}", "level": level, "test_successes": passed, "test_n": 6}


def test_library_round_trip_and_resolve(tmp_path):
    root = tmp_path / "lib"
    w, summ = _fake_cell(tmp_path)
    p = library.save(root, w, summ)
    assert (p / "policy.py").exists() and (p / "SKILL.md").exists()
    w2, summ2 = _fake_cell(tmp_path, "cube_palm", "layered", passed=1)
    library.save(root, w2, summ2)  # unsolved at threshold 0.5
    assert [e["id"] for e in library.resolve(root, "all")] == ["cube_pinch/layered"]
    assert library.resolve(root, "all", exclude_task="cube_pinch") == []
    assert library.resolve(root, "cube_pinch/layered")[0]["solved"]
    dest = tmp_path / "view"
    library.expose(library.resolve(root, "all"), dest)
    assert (dest / "cube_pinch__layered" / "policy.py").exists()
    assert library.load_policy(root, "cube_pinch", "layered").name == "policy.py"
    with pytest.raises(KeyError):
        library.load_policy(root, "cube_palm")


def test_compose_runs_library_skill_on_api(tmp_path):
    root = tmp_path / "lib"
    w, summ = _fake_cell(tmp_path)
    library.save(root, w, summ)
    api = types.SimpleNamespace(level="layered", t=0.0, _log=[], holds=[])

    def hold(s):
        api.holds.append(s)
        yield

    api.hold = hold
    compose.attach(api, root)
    list(api.run_skill("cube_pinch"))
    assert api.marker == 1 and api.holds == [0.1] and any("skill cube_pinch done" in l for l in api._log)
    src = compose.chain_source(["cube_pinch", "cube_palm"])
    assert src.count("run_skill") == 2 and "hold(0.3)" in src
    compile(src, "policy.py", "exec")


def test_prompt_lists_visible_skills():
    sk = [{"id": "cube_pinch/layered", "task": "cube_pinch", "level": "layered", "test_successes": 5, "test_n": 6}]
    msg = prompt.first_user_message("TASK", "pass", "contacts", sk)
    assert "skills_lib/cube_pinch__layered" in msg and HINTS["contacts"] in msg
    assert "run_skill" in prompt.system_prompt("layered", 4, "t", compose=True)


def test_prompt_privileged_modes():
    on = prompt.system_prompt("joints", 4, "t", privileged=True)
    off = prompt.system_prompt("joints", 4, "t", privileged=False)
    assert "are allowed and expected" in on
    assert "Privileged sim observations are OFF" in off
    assert "- api.object_pose" not in tools.api_doc("joints", privileged=False)
    compose_off = prompt.system_prompt("layered", 4, "t", compose=True, privileged=False)
    assert "no privileged object reads" in compose_off


def test_rq2_aggregate():
    cells = [{"level": "joints", "test_successes": 6, "test_n": 6, "solved_iter": 2, "tokens_to_solve": 100},
             {"level": "joints", "test_successes": 0, "test_n": 6, "solved_iter": None, "tokens_to_solve": None}]
    row = aggregate(cells)[0]
    assert row["level"] == "joints" and row["solve_rate"] == 0.5 and row["iters_to_solve"] == 2
    assert "level" in C.table([row], ("level", "solve_rate"))


def test_task_dir_resolves_short_names():
    assert C.task_dir("cube_pinch").name == "cube_pinch"
    assert C.task_dir("composed/pinch_then_palm").name == "pinch_then_palm"
    assert len(C.ALL_TASKS) == 12 and all(C.task_dir(t).exists() for t in C.ALL_TASKS)
