"""Aggregate the study: learning curves, robustness, budget-to-solve.

    python bench_runs/tools/analyze.py      -> bench_runs/results.json, bench_runs/RESULTS.md
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "bench_runs"
LEVELS = ("joints", "fingertips", "synergies", "contacts", "layered", "force", "skills")


def load_summary(p):
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    by = defaultdict(list)
    for c in s["cases"]:
        by[c["case"].split(":")[1]].append(bool(c["success"]))
    return {"n": s["n"], "successes": s["successes"], "by_severity": {k: [sum(v), len(v)] for k, v in by.items()}}


def main():
    cells = {}
    for w in sorted((B / "agents").glob("*/*")):
        task, level = w.parent.name, w.name
        cfg = json.loads((w / "agent.json").read_text())
        cap = cfg.get("max_iters", 8)
        prog = [json.loads(l) for l in (w / "progress.jsonl").read_text().splitlines()] if (w / "progress.jsonl").exists() else []
        rate = [r["successes"] / r["n"] for r in prog]
        # Carry the last score forward after an early stop (the agent judged it done).
        curve = rate + [rate[-1]] * (cap - len(rate)) if rate else []
        t = [datetime.fromisoformat(r["time"]) for r in prog]
        minutes = [round((x - t[0]).total_seconds() / 60 + prog[0]["wall_s"] / 60, 1) for x in t] if t else []
        first2 = next((r["iter"] for r in prog if r["successes"] >= 2), None)
        first3 = next((r["iter"] for r in prog if r["successes"] == 3), None)
        dev_sev = defaultdict(lambda: [0, 0])
        usage = []
        for r in prog:
            for cf in sorted((w / f"iter_{r['iter']}").glob("case_*.json")):
                c = json.loads(cf.read_text())
                sv = c["case"].split(":")[1]
                dev_sev[sv][0] += bool(c["success"]); dev_sev[sv][1] += 1
                if c.get("api_usage"):
                    usage.append((r["iter"], c["api_usage"]))
        last_iter = prog[-1]["iter"] if prog else None
        layers_last = defaultdict(int)
        for it, u in usage:
            if it == last_iter:
                for name, v in u.items():
                    layers_last[v["layer"]] += v["calls"]
        cells[(task, level)] = {"dev_by_severity": dict(dev_sev), "layers_last_iter": dict(layers_last),
                                "workdir": str(w.relative_to(ROOT)), "task": task, "level": level, "iters_used": len(prog), "dev_rate": rate,
                                "dev_curve": curve, "minutes": minutes, "first_2of3": first2, "first_3of3": first3,
                                "final": (w / "FINAL").read_text().strip() if (w / "FINAL").exists() else None,
                                "test": load_summary(w / "test" / "summary.json"), "done": (w / "FINAL").exists()}
    tasks = sorted({k[0] for k in cells} | {p.name for p in (B / "expert").glob("*") if p.is_dir() and not p.name.endswith("_smoke")})
    expert = {t: load_summary(B / "expert" / t / "test" / "summary.json") for t in tasks}

    agg = {}
    for lv in LEVELS:
        cs = [c for (t, l), c in cells.items() if l == lv and c["dev_curve"]]
        if not cs:
            continue
        cap = max(len(c["dev_curve"]) for c in cs)
        M = np.array([c["dev_curve"] + [c["dev_curve"][-1]] * (cap - len(c["dev_curve"])) for c in cs])
        tests = [c["test"] for c in cs if c["test"]]
        sev = {}
        for s in ("0", "0.5", "1"):
            num = sum(x["by_severity"].get(s, [0, 0])[0] for x in tests)
            den = sum(x["by_severity"].get(s, [0, 0])[1] for x in tests)
            sev[s] = round(num / den, 3) if den else None
        dsev = {}
        for sv in ("0", "0.5", "1"):
            n = sum(c["dev_by_severity"].get(sv, [0, 0])[0] for c in cs)
            d = sum(c["dev_by_severity"].get(sv, [0, 0])[1] for c in cs)
            dsev[sv] = round(n / d, 3) if d else None
        agg[lv] = {"cells": len(cs), "dev_by_severity": dsev, "dev_curve_mean": M.mean(0).round(3).tolist(),
                   "dev_auc": round(float(M.mean()), 3),
                   "test_rate": round(sum(x["successes"] for x in tests) / max(1, sum(x["n"] for x in tests)), 3) if tests else None,
                   "test_by_severity": sev,
                   "tasks_solved_test_80pct": sum(1 for x in tests if x["successes"] >= 0.8 * x["n"]),
                   "median_iters_to_2of3": (float(np.median([c["first_2of3"] or cap + 1 for c in cs]))),
                   "never_2of3": sum(1 for c in cs if c["first_2of3"] is None)}
    out = {"cells": [dict(v) for v in cells.values()], "levels": agg, "expert": expert}
    (B / "results.json").write_text(json.dumps(out, indent=1))

    L = ["# Results", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}.", "",
         "## By level", "", "| Level | Cells | Dev AUC (mean success over 4 iters) | Test success | Test sev 0 / 0.5 / 1 | Tasks with test >= 80% | Median iters to 2/3 | Never 2/3 |",
         "|---|---|---|---|---|---|---|---|"]
    for lv, a in agg.items():
        s = a["test_by_severity"]
        L.append(f"| {lv} | {a['cells']} | {a['dev_auc']} | {a['test_rate']} | {s['0']} / {s['0.5']} / {s['1']} | "
                 f"{a['tasks_solved_test_80pct']} | {a['median_iters_to_2of3']} | {a['never_2of3']} |")
    L += ["", "Dev success by severity (all dev cases, all iterations), 0 / 0.5 / 1:", ""]
    for lv, a in agg.items():
        d = a["dev_by_severity"]
        L.append(f"- {lv}: {d['0']} / {d['0.5']} / {d['1']}")
    L += ["", "Mean dev success per iteration (fresh cases at severities 0, 0.5, 1 each iteration):", ""]
    for lv, a in agg.items():
        L.append(f"- {lv}: {a['dev_curve_mean']}")
    L += ["", "## By task (held-out test successes; expert out of 10, agents out of 6; dev per iteration in brackets)", "",
          "| Task | Expert | " + " | ".join(LEVELS) + " |", "|---|---|" + "---|" * len(LEVELS)]
    for t in tasks:
        e = expert.get(t)
        row = [t, f"{e['successes']}/{e['n']}" if e else "-"]
        for lv in LEVELS:
            c = cells.get((t, lv))
            if not c:
                row.append("-"); continue
            te = f"{c['test']['successes']}/{c['test']['n']}" if c["test"] else ("run" if not c["done"] else "untested")
            dev = "".join(str(round(r * 3)) for r in c["dev_rate"])
            row.append(f"{te} ({dev})")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Layered level: commands per layer in the last iteration (summed over its 3 cases)", "",
          "| Task | joints | fingertips | synergies | contacts | arm | dev |", "|---|---|---|---|---|---|---|"]
    for (t, lv), c in sorted(cells.items()):
        if lv != "layered" or not c["layers_last_iter"]:
            continue
        u = c["layers_last_iter"]
        L.append(f"| {t} | " + " | ".join(str(u.get(k, 0)) for k in ("joints", "fingertips", "synergies", "contacts", "arm"))
                 + f" | {''.join(str(round(r * 3)) for r in c['dev_rate'])} |")
    L += ["", "## Best policy per task (held-out test)", "", "| Task | Best | Test | Policy |", "|---|---|---|---|"]
    best = {}
    for t in tasks:
        cand = [(c["test"]["successes"] / c["test"]["n"], lv, c) for (tt, lv), c in cells.items() if tt == t and c["test"]]
        e = expert.get(t)
        if cand:
            r, lv, c = max(cand, key=lambda x: (x[0], -(x[2]["first_3of3"] or 99), sum(x[2]["dev_rate"])))
            best[t] = {"level": lv, "test": c["test"], "policy": f"{c['workdir']}/iter_{c['final']}/policy.py"}
            L.append(f"| {t} | {lv} | {c['test']['successes']}/{c['test']['n']} | `{best[t]['policy']}` |")
        else:
            L.append(f"| {t} | - | - | - |")
    out["best"] = best
    (B / "results.json").write_text(json.dumps(out, indent=1))
    (B / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
