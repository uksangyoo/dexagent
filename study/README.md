# Dexterity action abstraction evaluation tools

Read [the study](../research/action_abstraction_study.md) for the proposed representation, industrial tasks, literature, and controls. This directory contains a reproducible asset audit, a pending evaluation schedule, and measured-result analysis. It does not contain trained controllers, a composed UR5e environment, an RTX runtime, or manipulation results.

Run the dependency-free tools from `dexagent`:

```bash
python study/study.py audit --output study/artifacts/asset_audit.json
python study/study.py manifest --profile pilot --output study/artifacts/pilot_manifest.jsonl
python -m unittest discover -s study -p 'test_study.py' -v
```

The pilot manifest contains 960 **pending** trials. `--profile full` produces 36,000. Methods share physical case seeds across independently trained controller seeds. A case-generation adapter must map case IDs to frozen object instances, initial states, and perturbations; random seeds alone do not define geometry or prevent train/test leakage. Store those case artifacts and their hashes before execution. The protocol hash prevents mixing changed protocols silently.

The hand import probe requires the existing Newton environment:

```bash
/home/uyoo/miniconda3/envs/newton/bin/python study/newton_asset_probe.py \
  --output study/artifacts/newton_asset_probe.json
```

It imports the real mounted hand with resolved mesh paths, builds a CPU model, and checks forward kinematics. It does not step dynamics or render. Its Newton installation differs from the one pinned in the Isaac Lab checkout. The original Tesollo files are unchanged.

After actual evaluation, write one JSON object per line by copying the corresponding manifest row and adding:

```json
{
  "status": "completed",
  "experiment_id": "a_unique_frozen_experiment_identifier",
  "success": false,
  "failure_reason": "timeout",
  "elapsed_sim_s": 120.0,
  "llm_calls": 8,
  "llm_tokens": 6000,
  "recovery_attempts": 3
}
```

This fragment illustrates the schema and is not a recorded result. `success` must be a JSON boolean. Add asset/case hashes, simulator revision, solver/renderer settings, controller checkpoint hashes, graph/prompt/model versions, and trace location in the experiment metadata. Log infrastructure failures and retried attempts separately; all completed task attempts, including drops, jams, and timeouts, count in the evaluation.

Analyze measured episodes:

```bash
python study/study.py summarize /path/to/measured_episodes.jsonl \
  --manifest study/artifacts/pilot_manifest.jsonl \
  --compare contact_grammar flat_primitives \
  --output study/artifacts/measured_summary.json
```

The analyzer checks duplicates, protocol and observation-regime consistency, pairing, and manifest coverage. It rejects pending trials, string-valued success, mismatched case seeds, and incomplete planned comparisons. Omit `--compare` to inspect partial per-seed summaries without comparative intervals. It computes a paired crossed bootstrap over controller-training seeds and test cases per task/split. Intervals are exploratory with few seeds; cases must be independent. Object-family clustered inference, multiple-comparison corrections, and latency/force/cable-metric aggregation remain to be added for the final study.

Tests use synthetic outcomes only in memory. The artifacts currently saved here are asset checks and pending trials; there is no synthetic success table masquerading as a robotics experiment.
