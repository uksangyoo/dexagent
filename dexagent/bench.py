"""Score a policy on a task over sampled initial conditions, headless.

    python -m dexagent.bench tasks/inhand/cube_reorient --policy expert --cases 1:0,2:0.5,3:1
    python -m dexagent.bench tasks/inhand/cube_reorient --policy my_policy.py --level fingertips \
        --cases 11:0,12:0.5,13:1 --out runs/x --film palm --label agentA/iter3

Each case runs in its own process (scenes draw their perturbations at import),
in parallel, behind a machine-wide pool of GPU slots so concurrent benches do
not oversubscribe GPU 1. The task folder supplies:

  scene.py   the scene, perturbed from `dexagent.cases.current()`, and the expert `routines`
  check.py   `Monitor(sim)`: `.step()` once per control step, `.result()` -> {"success": bool, "metrics": {...}}
  TASK.md    what to achieve, the success rule, the variation ranges

A policy file defines `run(api)`, a generator that yields once per 20 ms
control step (every `api` motion call is itself a generator: `yield from
api.move(...)`). `api` is built by `dexagent.levels.make_api(level, ...)`.
`--policy expert` runs the scene's own first routine instead.

After the policy returns, the bench holds the last targets for SETTLE_S and
then asks the monitor. Policy exceptions are failures and their tracebacks
are reported. Results: <out>/case_<seed>_<sev>.json, <out>/summary.json, and a
line per case in bench_runs/ledger.jsonl.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "bench_runs"
SLOTS = int(os.environ.get("DEXAGENT_SLOTS", "14"))
SETTLE_S = 1.0
NVIDIA_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


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


def run_one(task_dir, policy, level, case_text, out, film=None, max_sim_s=60.0):
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
    import dexagent.hand as hand_mod
    from robosimstudio_real2sim_agent.scenefile import run_scene_file
    from dexagent.cases import current

    # Every Controller logs its 20 finger targets, for synergy fitting and diagnosis.
    hand_log = []
    _record = hand_mod.Controller.record

    def record(self):
        hand_log.append(np.asarray(self.hand, float).copy())
        return _record(self)
    hand_mod.Controller.record = record

    rec = {"task": task_dir.name, "policy": str(policy), "level": level, "case": case_text,
           "success": False, "metrics": {}, "error": None}
    t0 = time.time()
    frames = []
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
            pspec = importlib.util.spec_from_file_location("policy", policy)
            pmod = importlib.util.module_from_spec(pspec)
            sys.path.insert(0, str(Path(policy).resolve().parent))
            pspec.loader.exec_module(pmod)
            gen = pmod.run(api)
        k = 0
        try:
            for _ in gen:
                monitor.step()
                if film and k % 5 == 0:
                    frames.append(sim.render(film)["rgb"])
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
            if film and (k + j) % 5 == 0:
                frames.append(sim.render(film)["rgb"])
        res = monitor.result()
        rec["success"] = bool(res.get("success")) and rec["error"] is None
        rec["metrics"] = res.get("metrics", {})
        rec["sim_s"] = round(float(sim.time), 2)
        if api is not None:
            rec["api_log"] = api._log[-60:]
            rec["api_usage"] = getattr(api, "_usage", None)
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


def bench(task_dir, policy, level, cases, out, film=None, label="", max_sim_s=60.0, timeout_s=900):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
               PYTHONPATH=str(ROOT))
    py = str(ROOT / ".venv/bin/python")

    def one(i_case):
        i, c = i_case
        lock = _slot()
        try:
            cmd = [py, "-m", "dexagent.bench", "_one", str(task_dir), "--policy", str(policy), "--level", level,
                   "--cases", c, "--out", str(out), "--max-sim-s", str(max_sim_s)]
            if film and i == 0:
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    one = bool(argv) and argv[0] == "_one"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--level", default="expert")
    ap.add_argument("--cases", default="0:0", help="comma-separated seed:severity")
    ap.add_argument("--out")
    ap.add_argument("--film", help="camera to film on the first case (tiled renderer)")
    ap.add_argument("--label", default="")
    ap.add_argument("--max-sim-s", type=float, default=60.0)
    a = ap.parse_args(argv[1:] if one else argv)
    if one:
        run_one(a.task, a.policy, a.level, a.cases, a.out, a.film, a.max_sim_s)
        return
    out = a.out or str(RUNS / "adhoc" / f"{Path(a.task).resolve().name}-{time.strftime('%H%M%S')}")
    policy = a.policy if a.policy == "expert" else str(Path(a.policy).resolve())
    s = bench(a.task, policy, a.level, a.cases.split(","), out, a.film, a.label, a.max_sim_s)
    for c in s["cases"]:
        err = (c.get("error") or "").strip().splitlines()
        print(f"  {c['case']:>8}  {'PASS' if c['success'] else 'fail'}  {json.dumps(c.get('metrics'))}"
              + (f"  ERROR: {err[-1]}" if err else ""))
    print(f"BENCH {s['task']} {a.level} {s['successes']}/{s['n']} in {s['wall_s']} s -> {out}/summary.json")


if __name__ == "__main__":
    main()
