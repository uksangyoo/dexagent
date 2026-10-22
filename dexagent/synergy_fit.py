"""Fit leave-one-task-out postural synergies from expert finger targets.

    python -m dexagent.synergy_fit [--k 6]

Reads bench_runs/expert/<task>/**/hand_*.npy (20 finger targets per control
step, logged by the bench) and writes dexagent/assets/synergies/<task>.npz
for every task: the mean and the top-k principal directions (scaled by their
standard deviation) of all OTHER tasks' postures, each task weighted equally.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets" / "synergies"


def load():
    data = {}
    tasks = {p.parent.name for p in ROOT.glob("tasks/*/*/scene.py")}
    for d in sorted((ROOT / "bench_runs" / "expert").iterdir()):
        if d.name not in tasks:
            continue
        files = sorted(d.rglob("hand_*.npy"))
        if files:
            q = np.concatenate([np.load(f) for f in files])
            # Drop exact repeats (holds) so long static phases do not dominate.
            q = q[np.r_[True, np.any(np.abs(np.diff(q, axis=0)) > 1e-4, axis=1)]]
            data[d.name] = q
    return data


def fit(rows, k, n_per_task=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([q[rng.choice(len(q), min(len(q), n_per_task), replace=False)] for q in rows])
    mean = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    sd = S / np.sqrt(len(X) - 1)
    var = sd ** 2 / (sd ** 2).sum()
    return mean, (Vt[:k].T * sd[:k]), sd[:k], var[:k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(argv)
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [p.parent.name for p in ROOT.glob("tasks/*/*/scene.py")]
    report = {}
    for t in tasks:
        others = [q for name, q in data.items() if name != t]
        mean, basis, sd, var = fit(others, a.k)
        np.savez(OUT / f"{t}.npz", mean=mean, basis=basis, sd=sd, var=var,
                 sources=np.array([n for n in data if n != t]))
        report[t] = {"explained": round(float(var.sum()), 3), "sources": len(others)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
