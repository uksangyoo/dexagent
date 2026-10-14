"""Create GaP subgraphs; all execution, routing, and checkpoints remain in GaP."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow


@dataclass(frozen=True)
class HandOperation:
    tool: str
    arguments: dict


def workflow(name, operations):
    if not operations:
        raise ValueError("A composite requires at least one operation")
    sg = Subgraph(name=name, skill="dexterous-hand")
    sg.set_on_error("controller_error")
    sg.add_node("result", type="tool", tool="hand.last_report")
    sg.set_exit_router(router_field="status", success_values=["succeeded", "failed"])
    for i, op in enumerate(operations):
        if op.tool not in {
            "hand.assign_roles",
            "hand.transition_contact",
            "hand.move_object",
            "hand.validate_action",
        }:
            raise ValueError(f"Unsupported composite operation {op.tool}")
        node = f"operation_{i}"
        sg.add_node(node, type="tool", tool=op.tool, inputs=op.arguments)
        sg.add_conditional_edges(
            node,
            router_field="status",
            mapping={
                "succeeded": f"operation_{i + 1}" if i + 1 < len(operations) else "result",
                "failed": "result",
            },
        )
    sg.add_edge("START", "operation_0")
    sg.add_edge("result", "END")
    # Complete reports stay in the ordinary GaP trace; the last report is also
    # exposed as the subgraph output for repair agents.
    sg.set_outputs(report=Ref("result.report"))
    graph = Workflow(name=name, description="Composable contact transitions and object effects")
    graph.add_subgraph(sg)
    graph.add_node("skill", type="subgraph", ref=name)
    graph.add_node("done", type="end", status="success")
    graph.add_node("failed", type="end", status="failure")
    graph.add_edge("START", "skill")
    graph.add_conditional_edges(
        "skill",
        router_field="exit",
        mapping={"succeeded": "done", "failed": "failed", "controller_error": "failed"},
    )
    graph.add_edge("done", "END")
    graph.add_edge("failed", "END")
    return graph


def preserve_successful_skill(
    destination, workflow_directory, reports, checkpoint_results, provenance
):
    """Preserve an agent-modified graph only after physical verification.

    Uses GaP artifacts rather than storing a second executable plan format.
    """
    if not reports or not all(r.success for r in reports):
        raise ValueError("Every operation must succeed")
    if not checkpoint_results or not all(c.passed for c in checkpoint_results):
        raise ValueError("Passed GaP physical checkpoints are required")
    source = Path(workflow_directory)
    if not (source / "workflow.json").is_file() or not (source / "checkpoints").is_dir():
        raise ValueError(
            "Preserve a materialized GaP workflow with its physical checkpoint sidecars"
        )
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / "workflow.json", destination / "workflow.json")
    for name in ("scripts", "checkpoints", "assets"):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name)
    (destination / "verification.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "reports": [r.model_dump(mode="json") for r in reports],
                "checkpoints": [{"name": c.name, "passed": c.passed} for c in checkpoint_results],
            },
            indent=2,
        )
        + "\n"
    )
    return destination
