# Screwdriver

Pick up a screwdriver, pivot it through loaded slip, engage the screw and drive it to its seat. Screwdriver and screw are independent unactuated rigid bodies with explicit mating thread contact. Only the 26 robot motors are commanded.

Run everything from the repository root with `CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1` and the Drake environment (`.venv-drake/bin/python`).

## Prepared-grasp study (`study/`)

The frozen refined22 comparison runs three adapters × three initial finger clearances for six seconds per trial. It starts from a prepared grasp, with no pickup credit. Methods, results, limitations and exact reproduction are in [HOM_REVISION.txt](HOM_REVISION.txt).

| Module | Purpose |
| --- | --- |
| `experiment` | One prepared-grasp trial for `--interface hom/cartesian/object` |
| `seat_continuation`, `validate_checkpoint` | Checkpoint-resumed seating stop and its independent continuity/hold check |
| `pickup`, `pickup_trial` | Horizontal-tool pickup development (incomplete) |
| `audit_scene`, `contact_validation`, `pad_validation`, `thread_validation` | Model, pressure-field and thread checks |
| `metrics`, `site_report`, `package_evidence` | Website data and the evidence archive |
| `reproduce` | Isolated checkout of the exact refined22 dynamics sources |

## Agent-authored prepared-grasp goal

`python -m tasks.screwdriver.agentic_site_report` publishes the turn-30 runs of `dexagent.agentic`. See [dexagent/agentic/README.md](../../dexagent/agentic/README.md).

## Continuous sequence (`sequence/`)

`python -m tasks.screwdriver.sequence.pickup --assembly --policy <bundle>` executes pickup, loaded pivot, cap support, transport, engagement, drive and hold. Policies come from `python -m dexagent.planning.workflow prepare|compile|feedback`. A full sequence is supported in code but has not yet passed all stages. `assembly` holds the drive/reset bridge. `audit`, `contact_replay` and `thread_bench` replay or check recordings without new rollouts. `tasks.json` lists the task stages.

Outputs go to `artifacts/hom_revision`, `artifacts/sequence_revision` and `artifacts/agentic_workflow`.
