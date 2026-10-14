#!/usr/bin/env python3
"""Asset audit, paired evaluation manifest and analysis; no simulated outcomes."""

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("protocol.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_hand(asset_root):
    """Read authored parameters; these are not hardware specifications."""
    models = []
    for path in sorted(Path(asset_root).glob("*.urdf")):
        tree = ET.parse(path)
        joints = tree.findall("joint")
        active = [j for j in joints if j.get("type") != "fixed"]
        links = tree.findall("link")
        children = {j.find("child").get("link") for j in joints}
        mesh_refs = sorted({m.get("filename") for m in tree.findall(".//mesh")})
        missing = []
        for ref in mesh_refs:
            # The supplied Tesollo assets use package://meshes/... locally.
            resolved = path.parent / ref.removeprefix("package://")
            if not resolved.is_file():
                missing.append(ref)
        mjcf_path = path.with_suffix(".xml")
        mjcf = ET.parse(mjcf_path) if mjcf_path.exists() else None
        models.append({
            "file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "links": len(links),
            "active_joints": len(active),
            "mimic_joints": len(tree.findall(".//mimic")),
            "roots": sorted({l.get("name") for l in links} - children),
            "authored_mass_kg": sum(float(m.get("value")) for m in tree.findall(".//mass")),
            "joint_limits": [{"name": j.get("name"), **j.find("limit").attrib} for j in active],
            "mesh_reference_count": len(mesh_refs),
            "missing_meshes_with_local_resolution": missing,
            "mjcf_actuator_count": len(mjcf.findall("./actuator/*")) if mjcf else None,
            "mjcf_sensor_count": len(mjcf.findall("./sensor/*")) if mjcf else None,
        })
    if not models:
        raise ValueError(f"No URDF assets found in {asset_root}")
    revisions = {}
    for name, path in {"IsaacLab": ROOT / "IsaacLab", "tesollo_model": ROOT / "tesollo_model",
                       "graph-as-policy": ROOT.parent / "graph-as-policy"}.items():
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        revisions[name] = result.stdout.strip() if result.returncode == 0 else None
    return {"kind": "static_asset_audit_not_manipulation_results", "revisions": revisions, "models": models}


def manifest(protocol, profile):
    cfg = protocol["profiles"][profile]
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for task, split, train_seed, case_index in itertools.product(
        cfg["tasks"], cfg["splits"], cfg["train_seeds"], range(cfg["cases_per_cell"])
    ):
        case_id = f"{task}/{split}/{case_index:04d}"
        # Same physical cases across both methods AND controller-training seeds.
        episode_seed = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], "big")
        methods = list(cfg["methods"])
        random.Random(f"{case_id}/{train_seed}").shuffle(methods)
        for method in methods:
            yield {
                "protocol_sha256": digest, "profile": profile,
                "task": task, "split": split, "method": method,
                "train_seed": train_seed, "case_id": case_id,
                "episode_seed": episode_seed,
                "observation_regime": protocol["observation_regime"],
                "status": "pending",
            }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return tuple(row[k] for k in ("task", "split", "method", "train_seed", "case_id"))


def validate_results(rows, planned=None):
    if not rows:
        raise ValueError("No measured results. Pending manifest rows are not observations.")
    seen = set()
    experiments = set()
    expected = {key(r): r for r in planned} if planned is not None else None
    for r in rows:
        k = key(r)
        if k in seen:
            raise ValueError(f"Duplicate trial: {k}")
        seen.add(k)
        if r.get("status") != "completed" or type(r.get("success")) is not bool:
            raise ValueError(f"Trial must be completed with boolean success: {k}")
        if type(r["train_seed"]) is not int:
            raise ValueError("train_seed must be an integer")
        for field in ("protocol_sha256", "observation_regime", "experiment_id"):
            if not isinstance(r.get(field), str) or not r[field]:
                raise ValueError(f"Missing {field}: {k}")
        experiments.add(tuple(r[f] for f in ("protocol_sha256", "observation_regime", "experiment_id")))
        for metric in ("elapsed_sim_s", "llm_calls", "llm_tokens", "recovery_attempts"):
            value = r.get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {metric}: {k}")
        if expected is not None:
            if k not in expected:
                raise ValueError(f"Unplanned trial: {k}")
            for field in ("protocol_sha256", "observation_regime", "episode_seed"):
                if r.get(field) != expected[k][field]:
                    raise ValueError(f"Manifest mismatch for {field}: {k}")
    if len(experiments) != 1:
        raise ValueError("Do not pool different protocols, experiments, or observation regimes")
    return {"completed": len(seen), "planned": len(expected) if expected is not None else None,
            "missing": len(set(expected) - seen) if expected is not None else None}


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    a, b = math.floor(p), math.ceil(p)
    return values[a] + (values[b] - values[a]) * (p - a)


def paired_bootstrap(rows, method_a, method_b, draws=2000, seed=0):
    """Crossed resampling of matched training seeds and physical test cases.

    Call per task/split. Incomplete or unbalanced grids fail, rather than
    silently dropping failures or pretending repeated seeds are independent.
    """
    if len({(r["task"], r["split"]) for r in rows}) != 1:
        raise ValueError("Paired comparison requires exactly one task and split")
    maps = {}
    for method in (method_a, method_b):
        maps[method] = {(r["train_seed"], r["case_id"]): int(r["success"])
                        for r in rows if r["method"] == method}
    a, b = maps[method_a], maps[method_b]
    if not a or a.keys() != b.keys():
        raise ValueError("Comparison requires identical, nonempty paired trial sets")
    seeds = sorted({k[0] for k in a})
    cases = sorted({k[1] for k in a})
    if set(itertools.product(seeds, cases)) != a.keys():
        raise ValueError("Comparison requires a complete training-seed by case grid")
    per_seed = [statistics.mean(a[s, c] - b[s, c] for c in cases) for s in seeds]
    result = {"method_a": method_a, "method_b": method_b,
              "success_difference": statistics.mean(per_seed),
              "training_seeds": len(seeds), "test_cases": len(cases),
              "per_seed_difference": per_seed, "ci95": None}
    if len(seeds) < 2 or len(cases) < 2:
        result["warning"] = "Insufficient independent seeds/cases for a crossed bootstrap interval"
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        ss = rng.choices(seeds, k=len(seeds))
        cc = rng.choices(cases, k=len(cases))
        samples.append(statistics.mean(a[s, c] - b[s, c] for s in ss for c in cc))
    result["ci95"] = [percentile(samples, .025), percentile(samples, .975)]
    result["warning"] = "Exploratory interval; small seed counts and shared object families limit inference"
    return result


def summarize(rows, planned, compare, draws):
    coverage = validate_results(rows, planned)
    if compare and coverage["missing"]:
        raise ValueError("Finish the planned grid before computing comparative intervals")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["task"], row["split"], row["method"])].append(row)
    summaries = []
    for (task, split, method), group in sorted(groups.items()):
        seeds = collections.defaultdict(list)
        for r in group:
            seeds[r["train_seed"]].append(r["success"])
        summaries.append({"task": task, "split": split, "method": method,
                          "n": len(group), "successes": sum(r["success"] for r in group),
                          "mean_seed_success": statistics.mean(map(statistics.mean, seeds.values())),
                          "per_seed_success": {str(s): statistics.mean(v) for s, v in sorted(seeds.items())}})
    pairs = []
    if compare:
        for task, split in sorted({(r["task"], r["split"]) for r in rows}):
            group = [r for r in rows if r["task"] == task and r["split"] == split]
            pair = paired_bootstrap(group, *compare, draws=draws)
            pairs.append({"task": task, "split": split, **pair})
    return {"coverage": coverage, "cells": summaries, "paired_comparisons": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("manifest")
    plan.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    plan.add_argument("--output", type=Path, required=True)
    analysis = sub.add_parser("summarize")
    analysis.add_argument("results", type=Path)
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--compare", nargs=2)
    analysis.add_argument("--bootstrap-draws", type=int, default=2000)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        value = audit_hand(ROOT / "tesollo_model" / "dg5fs")
        write_json(args.output, value)
        print(f"Audited {len(value['models'])} models: {args.output}")
    elif args.command == "manifest":
        rows = list(manifest(json.loads(PROTOCOL.read_text()), args.profile))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"Wrote {len(rows)} pending trials: {args.output}")
    else:
        if args.bootstrap_draws < 100:
            parser.error("Use at least 100 bootstrap draws")
        write_json(args.output, summarize(read_jsonl(args.results), read_jsonl(args.manifest),
                                          args.compare, args.bootstrap_draws))
        print(f"Wrote analysis: {args.output}")


if __name__ == "__main__":
    main()
