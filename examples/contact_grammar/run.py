"""Run a physical contact operation through GaP, with a force checkpoint."""

import argparse
import json
from pathlib import Path

import gap

from dexagent.gap.connector import DexterousConnector
from dexagent.grammar.finger_roles import Finger
from dexagent.grammar.schema import Constraints, Termination, TransitionRequest
from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv
from dexagent.skills.composition import HandOperation, workflow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/contact_grammar_demo"))
    p.add_argument("--finger", choices=[f.value for f in Finger], default="index")
    p.add_argument("--timeout", type=float, default=4.0)
    p.add_argument(
        "--object-point",
        type=float,
        nargs=3,
        default=[-0.022, 0.015, 0.010],
        help="Requested surface point in nut coordinates; default is the index-finger fixture regression case",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = UR5eTesolloEnv()
    conn = DexterousConnector(env)
    finger = Finger(args.finger)
    point, _, _ = env.contact_target(finger, tuple(args.object_point))
    constraints = Constraints(
        min_support_contacts=0,
        max_normal_force_N=5.0,
        calibration_id="fixture_bootstrap_provisional",
    )
    request = TransitionRequest(
        operation="establish_contact",
        object="hex_nut",
        finger=finger,
        object_point=tuple(point),
        constraints=constraints,
        termination=Termination(timeout_s=args.timeout),
    )
    roles = {f.value: ("SUPPORT" if f == finger else "FREE") for f in Finger}
    graph = workflow(
        "establish",
        [
            HandOperation("hand.assign_roles", {"roles": roles}),
            HandOperation("hand.transition_contact", {"request": request.model_dump(mode="json")}),
        ],
    )
    graph.save(args.output / "workflow.json")
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    source = """from gap.runtime.verify import Checkpoint
def loaded(world):
    s=world.sim_state["dexterous_hand"]
    cs=s["fingers"][FINGER]["contacts"]
    return sum(c["normal_force_N"] or 0 for c in cs) >= .05 and all(c["tangential_speed_m_s"] <= .005 for c in cs)
CHECKPOINTS=[Checkpoint(name="measured_support",subgraph="establish",predicate=loaded)]
""".replace("FINGER", repr(finger.value))
    (checkpoints / "establish.py").write_text(source)
    result = gap.execute(
        args.output,
        connector=conn,
        skills=[],
        trace_dir=args.output / "trace",
        checkpoints="raise",
        max_node_workers=1,
    )
    out = {
        "workflow_success": result.success,
        "exit_status": result.exit_status,
        "error": str(result.error) if result.error else None,
        "checkpoints": [{"name": c.name, "passed": c.passed} for c in result.checkpoint_results],
        "reports": [r.model_dump(mode="json") for r in conn.hand.history],
    }
    (args.output / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.output / "contact_trace.json").write_text(
        json.dumps(conn.hand.contacts.last_trace, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "reports"}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
