"""Paired A–F experiment manifests, budgets, and honest result validation."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexagent.gap.feedback import report
from dexagent.grammar.schema import FailureCode
from dexagent.tasks.common import TaskKind, require_capabilities, sample_case
from dexagent.tasks.monitor import EpisodeEnded, TaskMonitor

from .representations import Representation


@dataclass(frozen=True)
class Budget:
    # Improvement rounds are separate from decisions within an episode.
    improvement_iterations: int = 20
    tool_calls: int = 60
    simulated_seconds: float = 30.0
    wall_seconds: float = 600.0


def manifest(cases_per_task=20, agent_seeds=(0, 1, 2), splits=("test_id", "test_ood")):
    rows = []
    for task in TaskKind:
        for split in splits:
            for i in range(cases_per_task):
                case = sample_case(task, split, i)
                for seed in agent_seeds:
                    for method in Representation:
                        rows.append(
                            {
                                "trial_id": f"{case.case_id}:{seed}:{method.value}",
                                "case": case.to_dict(),
                                "agent_seed": seed,
                                "method": method.value,
                                "status": "pending",
                                "success": None,
                                "budget": Budget().__dict__,
                                "observation_regime": "state_and_solved_contacts",
                            }
                        )
    return rows


def run_trial(
    row,
    adapter,
    task,
    agent,
    *,
    available_capabilities,
    initialization_valid,
    provenance,
    improvement_iteration=0,
    measure_task=None,
):
    """agent.propose(state, feedback, iteration) returns an interface payload.

    The caller supplies a fresh agent with the trial's recorded seed. No LLM
    service, prompt choice, learned controller, or success outcome is inferred.
    """
    require_capabilities(row["case"]["task"], available_capabilities)
    if adapter.kind.value != row["method"]:
        raise ValueError("Representation adapter does not match the trial label")
    if measure_task is None:
        if row["case"]["task"] != TaskKind.NUT:
            raise ValueError("This benchmark requires its calibrated task-measurement adapter")

        def measure_task(state):
            return state

    if not hasattr(adapter.env, "observe_each_step"):
        raise ValueError(
            "Benchmark execution requires synchronized per-physics-step task monitoring"
        )
    if not initialization_valid:
        raise ValueError("Invalid initialization; repair the paired case before comparing methods")
    required = {
        "agent_model",
        "agent_prompt_hash",
        "agent_seed",
        "controller_hash",
        "scene_hash",
        "observation_regime",
    }
    if not required <= set(provenance):
        raise ValueError(f"Missing experimental provenance: {sorted(required - set(provenance))}")
    if provenance["agent_seed"] != row["agent_seed"]:
        raise ValueError("Agent seed mismatch")
    if provenance["observation_regime"] != row["observation_regime"]:
        raise ValueError("Observation regime mismatch")
    provenance = {**provenance, "action_horizon_s": adapter.max_action_seconds}
    budget = Budget(**row["budget"])
    started = time.monotonic()
    sim_start = adapter.env.time_s
    reports = []
    feedback = None
    verdict = None
    recoveries = 0
    failed = False
    monitor = TaskMonitor(task, measure_task)
    for decision in range(budget.tool_calls):
        if (
            time.monotonic() - started >= budget.wall_seconds
            or adapter.env.time_s - sim_start >= budget.simulated_seconds
        ):
            break
        payload = agent.propose(adapter.env.observe().model_dump(mode="json"), feedback, decision)
        if time.monotonic() - started >= budget.wall_seconds:
            break
        # All methods receive the same feedback structure. E vs F isolates
        # role changes; separate feedback-removal ablations can be added later.
        remaining = budget.simulated_seconds - (adapter.env.time_s - sim_start)
        if remaining < adapter.env.dt:
            break
        adapter.remaining_simulated_seconds = remaining
        before = adapter.env.observe()
        try:
            with adapter.env.observe_each_step(monitor):
                monitor(before)
                result = adapter.execute(payload)
        except EpisodeEnded as ended:
            code = (
                None
                if ended.verdict.success
                else {
                    "dropped": FailureCode.DROPPED,
                    "timeout": FailureCode.TIMEOUT,
                    "tension_violation": FailureCode.FORCE_LIMIT,
                }.get(ended.verdict.reason, FailureCode.CONTROLLER_FAILURE)
            )
            result = report(
                before,
                ended.state,
                success=ended.verdict.success,
                code=code,
                message=f"Task terminal predicate: {ended.verdict.reason}",
                intended=payload,
                steps=int(np.ceil((ended.state.time_s - before.time_s) / adapter.env.dt)),
            )
        reports.append(result)
        if result.success and failed:
            recoveries += 1
            failed = False
        elif not result.success:
            failed = True
        feedback = result.model_dump(mode="json")
        verdict = monitor.verdict
        if verdict is not None and verdict.terminal:
            break
    return {
        **row,
        "trial_id": f"{row['trial_id']}:iteration_{improvement_iteration}",
        "status": "completed",
        "success": bool(verdict and verdict.success),
        "provenance": provenance,
        "reports": [r.model_dump(mode="json") for r in reports],
        "improvement_iteration": improvement_iteration,
        "tool_calls": len(reports),
        "feasible_actions": sum(r.feasible for r in reports),
        "failed_actions": sum(not r.success for r in reports),
        "recoveries": recoveries,
        "task_recovered_after_failure": bool(
            verdict and verdict.success and any(not r.success for r in reports)
        ),
        "simulated_seconds": adapter.env.time_s - sim_start,
        "wall_seconds": time.monotonic() - started,
        "task_metrics": {} if verdict is None else verdict.metrics,
        "task_measurement_samples": monitor.samples,
    }


def learning_curve(training_round, evaluate_frozen, repair, *, budget=None):
    """Hooks for GaP generation/repair, without creating another graph executor.

    training_round(i) executes training cases through GaP and returns feedback.
    evaluate_frozen(i) evaluates a snapshot on held-out cases; its results never
    enter repair(). This explicit separation prevents test-feedback leakage.
    """
    budget = budget or Budget()
    curve = []
    for iteration in range(budget.improvement_iterations + 1):
        curve.append({"improvement_iteration": iteration, "held_out": evaluate_frozen(iteration)})
        if iteration < budget.improvement_iterations:
            training_feedback = training_round(iteration)
            repair(training_feedback, iteration)
    return curve


def summarize(rows):
    if not rows or any(
        r.get("status") != "completed" or type(r.get("success")) is not bool for r in rows
    ):
        raise ValueError("Only completed trials with measured Boolean outcomes can be summarized")
    ids = [r["trial_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trial id")
    cases = {}
    matched_fields = (
        "controller_hash",
        "scene_hash",
        "agent_model",
        "agent_seed",
        "observation_regime",
        "action_horizon_s",
    )
    for row in rows:
        if not set(matched_fields) <= set(row.get("provenance", {})):
            raise ValueError("Missing paired experimental provenance")
        key = (row["case"]["case_id"], row["agent_seed"], row["improvement_iteration"])
        cases.setdefault(key, []).append(row)
    for block in cases.values():
        if len(block) != 6 or {r["method"] for r in block} != {m.value for m in Representation}:
            raise ValueError("Incomplete paired A–F block")
        for row in block[1:]:
            if row["case"] != block[0]["case"] or row["budget"] != block[0]["budget"]:
                raise ValueError(
                    "Physical case or improvement budget differs within a paired block"
                )
            for name in matched_fields:
                if row["provenance"][name] != block[0]["provenance"][name]:
                    raise ValueError(f"Unmatched {name} within a paired block")

    def aggregate(subset):
        return {
            m.value: {
                "episodes": len(group := [r for r in subset if r["method"] == m.value]),
                "success_rate": sum(r["success"] for r in group) / len(group),
                "feasibility_rate": sum(r["feasible_actions"] for r in group)
                / max(1, sum(r["tool_calls"] for r in group)),
                "failed_actions": sum(r["failed_actions"] for r in group),
                "next_action_recovery_rate": sum(r["recoveries"] for r in group)
                / max(1, sum(r["failed_actions"] for r in group)),
                "task_recovery_rate": sum(r["task_recovered_after_failure"] for r in group)
                / max(1, sum(r["failed_actions"] > 0 for r in group)),
            }
            for m in Representation
        }

    strata = {}
    for row in rows:
        key = (
            f"{row['case']['task']}/{row['case']['split']}/iteration_{row['improvement_iteration']}"
        )
        strata.setdefault(key, []).append(row)
    return {key: aggregate(group) for key, group in strata.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["manifest", "summarize"])
    p.add_argument("--cases", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--results", type=Path)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        args.output.write_text("".join(json.dumps(r) + "\n" for r in manifest(args.cases)))
    else:
        if not args.results:
            p.error("--results is required")
        args.output.write_text(
            json.dumps(
                summarize([json.loads(s) for s in args.results.read_text().splitlines()]), indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
