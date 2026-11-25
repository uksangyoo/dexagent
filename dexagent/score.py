"""Score a policy on a task over sampled initial conditions, headless.

Two entry points, one module:

    # score a policy file (or the scene's own expert routine) on explicit cases
    python -m dexagent.score cases tasks/inhand/cube_reorient --policy expert --cases 1:0,2:0.5,3:1
    python -m dexagent.score cases tasks/inhand/cube_reorient --policy my_policy.py --level fingertips \
        --cases 11:0,12:0.5,13:1 --out results/x --film closeup --label agentA/iter3

    # one budgeted agent iteration in a workdir (policy.py + agent.json), see `iterate()`
    python -m dexagent.score iterate --workdir results/<run>/<task>/<level> [--film CAMERA]

Each case runs in its own process (scenes draw their perturbations at import),
in parallel, behind a machine-wide pool of GPU slots so concurrent benches do
not oversubscribe the GPU. Requires an explicit `CUDA_VISIBLE_DEVICES` (no default).
The task folder supplies:

  scene.py   the scene, perturbed from `dexagent.sim.cases.current()`, and the expert `routines`
  check.py   `Monitor(sim)`: `.step()` once per control step, `.result()` -> {"success": bool, "metrics": {...}}
  TASK.md    what to achieve, the success rule, the variation ranges

A policy file defines `run(api)`, a generator that yields once per 20 ms
control step (every `api` motion call is itself a generator: `yield from
api.move(...)`). `api` is built by `dexagent.levels.make_api(level, ...)`.
`--policy expert` runs the scene's own first routine instead.

After the policy returns, the bench holds the last targets for SETTLE_S and
then asks the monitor. Policy exceptions are failures and their tracebacks
are reported. Results: <out>/case_<seed>_<sev>.json, <out>/summary.json, and a
line per case in results/ledger.jsonl.
"""
import argparse
import fcntl
import hashlib
import json
import shutil
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results"
SLOTS = int(os.environ.get("DEXAGENT_SLOTS", "14"))
SETTLE_S = 1.0
NVIDIA_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


def require_cuda():
    """Exit unless CUDA_VISIBLE_DEVICES is set. Never default a GPU index."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        sys.exit(
            "CUDA_VISIBLE_DEVICES is not set. Export it explicitly, e.g. "
            "`export CUDA_VISIBLE_DEVICES=0` (this machine) before running."
        )


def _slot():
    d = RUNS / ".slots"
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(SLOTS):
            f = open(d / f"slot{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        time.sleep(1.0)


def side_by_side(rgbs):
    """Concatenate RGB frames left-to-right, centre-cropping to the shortest height."""
    import numpy as np
    frames = [np.asarray(f) for f in rgbs]
    h = min(f.shape[0] for f in frames)
    out = []
    for f in frames:
        y0 = (f.shape[0] - h) // 2
        out.append(f[y0:y0 + h])
    return np.concatenate(out, axis=1)


TRACE_EVERY = 25  # subsample privilege traces (~0.5 s at 50 Hz)
TRACE_MAX = 120


def _sample_trace(api, np):
    """One compact privileged-state sample for the coding agent (poses, tips, contacts)."""
    P, _ = api.measured_palm()
    tips = api.tips_world()
    objs, forces = {}, {}
    for name in api.objects:
        p, _ = api.object_pose(name)
        objs[name] = np.round(np.asarray(p, float), 4).tolist()
        forces[name] = np.round(api.contact_forces(name), 2).tolist()
    return {
        "t": round(float(api.t), 2),
        "palm": np.round(np.asarray(P, float), 4).tolist(),
        "tips": np.round(np.asarray(tips, float), 4).tolist(),
        "objects": objs,
        "forces": forces,
    }


def run_one(task_dir, policy, level, case_text, out, film=None, max_sim_s=60.0, privileged=True):
    """In this process: build the perturbed scene, run the policy, score it."""
    os.environ["DEXAGENT_CASE"] = case_text
    os.environ.setdefault("ROBOSIMSTUDIO_RENDERER", "tiled")
    if Path(NVIDIA_ICD).exists():
        os.environ.setdefault("VK_ICD_FILENAMES", NVIDIA_ICD)
    task_dir = Path(task_dir).resolve()
    os.environ["DEXAGENT_TASK"] = task_dir.name
    sys.path.insert(0, str(task_dir))
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import dexagent.sim.hand as hand_mod
    from robosimstudio_real2sim_agent.scenefile import run_scene_file
    from dexagent.sim.cases import current

    # Every Controller logs its 20 finger targets, for synergy fitting and diagnosis.
    hand_log = []
    _record = hand_mod.Controller.record

    def record(self):
        hand_log.append(np.asarray(self.hand, float).copy())
        return _record(self)
    hand_mod.Controller.record = record

    privileged = bool(privileged)
    rec = {"task": task_dir.name, "policy": str(policy), "level": level, "case": case_text,
           "success": False, "metrics": {}, "error": None, "privileged": privileged}
    t0 = time.time()
    frames = []
    trace = []
    try:
        loaded = run_scene_file(task_dir / "scene.py")
        sim, ns = loaded.sim, loaded.ns
        rec["perturbation"] = {k: round(v, 5) for k, v in current().drawn.items()}
        import importlib.util
        spec = importlib.util.spec_from_file_location("task_check", task_dir / "check.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        monitor = check.Monitor(sim)
        if policy == "expert":
            routines = ns.get("routines")
            gen = next(iter(routines.values()))(sim)
            api = None
        else:
            from dexagent.levels import make_api
            api = make_api(level, sim, objects=getattr(check, "OBJECTS", ()))
            if os.environ.get("DEXAGENT_SKILLS"):
                from dexagent.skills.compose import attach
                attach(api, os.environ["DEXAGENT_SKILLS"])
            pspec = importlib.util.spec_from_file_location("policy", policy)
            pmod = importlib.util.module_from_spec(pspec)
            sys.path.insert(0, str(Path(policy).resolve().parent))
            pspec.loader.exec_module(pmod)
            gen = pmod.run(api)
        k = 0
        # --film accepts a single camera or a comma-separated list; multiple views are
        # concatenated left-to-right into one frame (same layout as the agent dual view).
        film_cams = [c.strip() for c in str(film).split(",") if c.strip()] if film else []

        def _snap():
            rgbs = [np.asarray(sim.render(c)["rgb"]) for c in film_cams]
            if len(rgbs) == 1:
                return rgbs[0]
            return side_by_side(rgbs)

        try:
            for _ in gen:
                monitor.step()
                if privileged and api is not None and k % TRACE_EVERY == 0 and len(trace) < TRACE_MAX:
                    try:
                        trace.append(_sample_trace(api, np))
                    except Exception:
                        pass
                if film_cams and k % 5 == 0:
                    frames.append(_snap())
                k += 1
                if sim.time > max_sim_s:
                    rec["events"] = [f"stopped at the {max_sim_s:.0f} s sim limit"]
                    break
        except Exception:
            rec["error"] = traceback.format_exc(limit=6)
        steps = round(SETTLE_S * 50)
        for j in range(steps):
            if api is not None:
                api._c.step()
            else:
                sim.run(1).sync()
            monitor.step()
            if film_cams and (k + j) % 5 == 0:
                frames.append(_snap())
        res = monitor.result()
        rec["success"] = bool(res.get("success")) and rec["error"] is None
        rec["metrics"] = res.get("metrics", {})
        rec["sim_s"] = round(float(sim.time), 2)
        if api is not None:
            rec["api_log"] = list(api._log) if privileged else api._log[-60:]
            rec["api_usage"] = getattr(api, "_usage", None)
            if privileged and trace:
                rec["trace"] = {"every_steps": TRACE_EVERY, "samples": trace}
    except Exception:
        rec["error"] = traceback.format_exc(limit=8)
    rec["wall_s"] = round(time.time() - t0, 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    slug = case_text.replace(":", "_")
    if hand_log:
        np.save(out / f"hand_{slug}.npy", np.array(hand_log, dtype=np.float32))
    if frames:
        import imageio.v3 as iio
        iio.imwrite(out / f"film_{slug}.mp4", np.stack(frames), fps=10)
        pick = np.linspace(0, len(frames) - 1, 12).round().astype(int)
        rows = [np.concatenate([frames[i] for i in pick[r * 4:r * 4 + 4]], axis=1) for r in range(3)]
        iio.imwrite(out / f"sheet_{slug}.jpg", np.concatenate(rows, axis=0)[::3, ::3])
        rec["sheet"] = str(out / f"sheet_{slug}.jpg")
    (out / f"case_{slug}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def score(task_dir, policy, level, cases, out, film=None, label="", max_sim_s=60.0, timeout_s=900,
          robust=None, skills_root=None, privileged=True, film_all=False):
    """Run `cases` (list of 'seed:severity') in parallel subprocesses; return the summary dict.

    `robust` is an optional dict of execution perturbations (see dexagent.sim.cases.robust).
    With `film` set, only the first case is filmed unless `film_all` is True (held-out tests)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    require_cuda()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if robust:
        env["DEXAGENT_ROBUST"] = json.dumps(robust)
    if skills_root:
        env["DEXAGENT_SKILLS"] = str(skills_root)  # exposes api.run_skill (dexagent.skills.compose)
    py = str(ROOT / ".venv/bin/python")
    priv_flag = "--privileged" if privileged else "--no-privileged"

    def one(i_case):
        i, c = i_case
        lock = _slot()
        try:
            cmd = [py, "-m", "dexagent.score", "_one", str(task_dir), "--policy", str(policy), "--level", level,
                   "--cases", c, "--out", str(out), "--max-sim-s", str(max_sim_s), priv_flag]
            if film and (film_all or i == 0):
                cmd += ["--film", film]
            log = open(out / f"log_{c.replace(':', '_')}.txt", "w")
            try:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
        finally:
            lock.close()
        p = out / f"case_{c.replace(':', '_')}.json"
        if p.exists():
            return json.loads(p.read_text())
        tail = (out / f"log_{c.replace(':', '_')}.txt").read_text()[-3000:]
        return {"task": Path(task_dir).name, "case": c, "success": False, "error": "process died or timed out:\n" + tail}

    t0 = time.time()
    with ThreadPoolExecutor(len(cases)) as ex:
        recs = list(ex.map(one, enumerate(cases)))
    summary = {"task": Path(task_dir).resolve().name, "policy": str(policy), "level": level, "label": label,
               "n": len(recs), "successes": sum(r["success"] for r in recs),
               "wall_s": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "robust": robust or None,
               "cases": [{k: r.get(k) for k in ("case", "success", "metrics", "error", "sim_s", "perturbation", "sheet", "api_usage")}
                         for r in recs]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "ledger.jsonl", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in recs:
            f.write(json.dumps({"time": summary["time"], "task": summary["task"], "policy": str(policy), "level": level,
                                "label": label, "case": r.get("case"), "success": r["success"],
                                "metrics": r.get("metrics"), "error": (r.get("error") or "")[-300:] or None,
                                "sim_s": r.get("sim_s"), "wall_s": r.get("wall_s")}, default=str) + "\n")
    return summary


# ---------------------------------------------------------------------------------------------
# Budgeted agent iterations (what the coding agent's `run_iterate` tool calls)

SEVERITIES = (0.0, 0.5, 1.0)
# Held-out cases: never shown to the editing agent. Disjoint from the dev seeds (>= 20000).
HELD_OUT = ["1000:0", "1001:0.5", "1002:0.5", "1005:1", "1006:1", "1007:1"]


def dev_seeds(task, tag, k):
    """Three dev seeds for iteration k, unique to (task, tag) and disjoint from HELD_OUT."""
    h = int(hashlib.sha1(f"{task}|{tag}|{k}".encode()).hexdigest()[:8], 16)
    return [20000 + (h + 7919 * j) % 900000 for j in range(len(SEVERITIES))]


def iterate(workdir, film=None, film_case=1.0):
    """Snapshot workdir/policy.py into iter_<k>/, score it on three fresh dev cases, log progress.

    The workdir holds `policy.py` and `agent.json` ({"task": "tasks/...", "level": ..., "max_iters": N,
    optional "seed_tag", "privileged"}). The iteration counter only goes up, so a case can never be re-run to
    cherry-pick. Returns the progress row, or None when the budget is exhausted.
    """
    w = Path(workdir).resolve()
    cfg = json.loads((w / "agent.json").read_text())
    task, level, cap = cfg["task"], cfg["level"], int(cfg.get("max_iters", 8))
    privileged = bool(cfg.get("privileged", True))
    done = sorted(int(p.name.split("_")[1]) for p in w.glob("iter_*") if p.is_dir() and p.name.split("_")[1].isdigit())
    k = (done[-1] + 1) if done else 1
    # Budget = completed iterations; one killed by a timeout while queued for the GPU does not count.
    prog = w / "progress.jsonl"
    completed = len(prog.read_text().splitlines()) if prog.exists() else 0
    if completed >= cap:
        print(f"BUDGET EXHAUSTED: {cap} iterations used. Pick your final iteration in FINAL and stop.")
        return None
    it = w / f"iter_{k}"
    it.mkdir()
    shutil.copy(w / "policy.py", it / "policy.py")
    for extra in w.glob("*.py"):
        if extra.name != "policy.py":
            shutil.copy(extra, it / extra.name)
    ss = dev_seeds(Path(task).name, cfg.get("seed_tag", level), k)
    cases = [f"{s}:{v:g}" for s, v in zip(ss, SEVERITIES)]
    cases.sort(key=lambda c: float(c.split(":")[1]) != film_case)  # filmed case first
    t0 = time.time()
    s = score(ROOT / task, str(it / "policy.py"), level, cases, it, film=film,
              label=f"{w.parent.parent.name}/{Path(task).name}/{level}/iter{k}",
              skills_root=cfg.get("skills_root"), privileged=privileged)
    row = {"iter": k, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_s": round(time.time() - t0, 1),
           "successes": s["successes"], "n": s["n"],
           "by_severity": {c["case"].split(":")[1]: c["success"] for c in s["cases"]}}
    with open(prog, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"ITERATION {k} ({completed + 1} of {cap} budgeted): {s['successes']}/{s['n']}")
    for c in s["cases"]:
        err = (c.get("error") or "").strip()
        print(f"  severity {c['case'].split(':')[1]:>3} (seed {c['case'].split(':')[0]}): "
              f"{'PASS' if c['success'] else 'fail'}  metrics={json.dumps(c.get('metrics'))}")
        if c.get("perturbation"):
            print(f"      perturbation: {json.dumps(c['perturbation'])}")
        if err:
            print("      ERROR:\n" + "\n".join("        " + l for l in err.splitlines()[-12:]))
        if c.get("sheet"):
            print(f"      contact sheet (12 frames): {c['sheet']}")
    print(f"  per-case details incl. your api call log: {it}/case_*.json")
    return row


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--policy", default="expert")
        p.add_argument("--level", default="expert")
        p.add_argument("--cases", default="0:0", help="comma-separated seed:severity")
        p.add_argument("--out")
        p.add_argument("--film", help="camera, or comma-separated cameras concatenated left-to-right")
        p.add_argument("--max-sim-s", type=float, default=60.0)
        p.add_argument("--privileged", action=argparse.BooleanOptionalAction, default=True,
                       help="record full api_log and tip/object/contact traces (default: on)")

    c = sub.add_parser("cases", help="score a policy on explicit cases")
    c.add_argument("task")
    common(c)
    c.add_argument("--label", default="")
    c.add_argument("--film-all", action="store_true",
                   help="with --film, record a video for every case (default: first case only)")
    o = sub.add_parser("_one", help="internal: one case in this process")
    o.add_argument("task")
    common(o)
    i = sub.add_parser("iterate", help="one budgeted agent iteration in a workdir")
    i.add_argument("--workdir", required=True)
    i.add_argument("--film")
    i.add_argument("--film-case", type=float, default=1.0, help="severity of the filmed case (0, 0.5 or 1)")
    a = ap.parse_args(argv)

    if a.cmd == "_one":
        run_one(a.task, a.policy, a.level, a.cases, a.out, a.film, a.max_sim_s, privileged=a.privileged)
    elif a.cmd == "iterate":
        sys.exit(0 if iterate(a.workdir, a.film, a.film_case) else 3)
    else:
        out = a.out or str(RUNS / "adhoc" / f"{Path(a.task).resolve().name}-{time.strftime('%H%M%S')}")
        policy = a.policy if a.policy == "expert" else str(Path(a.policy).resolve())
        s = score(a.task, policy, a.level, a.cases.split(","), out, a.film, a.label, a.max_sim_s,
                  privileged=a.privileged, film_all=bool(getattr(a, "film_all", False)))
        for r in s["cases"]:
            err = (r.get("error") or "").strip().splitlines()
            print(f"  {r['case']:>8}  {'PASS' if r['success'] else 'fail'}  {json.dumps(r.get('metrics'))}"
                  + (f"  ERROR: {err[-1]}" if err else ""))
        print(f"SCORE {s['task']} {a.level} {s['successes']}/{s['n']} in {s['wall_s']} s -> {out}/summary.json")


if __name__ == "__main__":
    main()
