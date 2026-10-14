import contextlib
import tempfile
import unittest
from pathlib import Path

import gap
import mujoco
from gap.runtime.verify.world import World
from helpers import action_dict, state

from dexagent.gap.connector import DexterousConnector
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.skills.composition import HandOperation, workflow


class EnvDouble:
    def __init__(self):
        self.kin_model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).resolve().parents[1] / "runs/assets/scene.xml")
        )
        self.roles = dict.fromkeys(Finger, FingerRole.FREE)
        self.fixture_pose_world = None
        self.value = state()
        self.moves = 0

    def observe(self):
        return self.value.model_copy(update={"finger_roles": dict(self.roles)}, deep=True)

    def transaction(self):
        return contextlib.nullcontext()

    def close(self):
        pass


class TestConnector(DexterousConnector):
    def world_snapshot(self):
        return World(
            env_id=0,
            bodies={},
            sim_state={"dexterous_hand": self.env.observe().model_dump(mode="json")},
        )


class GaPTests(unittest.TestCase):
    def setUp(self):
        self.env = EnvDouble()
        self.conn = TestConnector(self.env)

    def test_real_executor_runs_registered_tools(self):
        roles = dict.fromkeys([f.value for f in Finger], "FREE")
        roles["thumb"] = "SUPPORT"
        graph = workflow("roles", [HandOperation("hand.assign_roles", {"roles": roles})])
        with tempfile.TemporaryDirectory() as p:
            graph.save(Path(p) / "workflow.json")
            result = gap.execute(
                p, connector=self.conn, skills=[], trace_dir=Path(p) / "trace", max_node_workers=1
            )
        self.assertTrue(result.success, str(result.error))
        self.assertEqual(result.exit_status, "success")
        self.assertEqual(self.env.roles[Finger.THUMB], FingerRole.SUPPORT)
        self.assertFalse(self.env.observe().fingers[Finger.THUMB].contacts)

    def test_failed_contact_preflight_routes_failure_and_retains_report(self):
        graph = workflow(
            "failure",
            [
                HandOperation("hand.move_object", {"action": action_dict()}),
                HandOperation(
                    "hand.assign_roles",
                    {"roles": dict.fromkeys([f.value for f in Finger], "DRIVE")},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as p:
            graph.save(Path(p) / "workflow.json")
            result = gap.execute(
                p, connector=self.conn, skills=[], trace_dir=Path(p) / "trace", max_node_workers=1
            )
        self.assertEqual(result.exit_status, "failure", str(result.error))
        self.assertEqual(self.conn.hand.history[-1].failure_code.value, "insufficient_support")
        self.assertEqual(self.env.roles[Finger.THUMB], FingerRole.FREE)
        self.assertIn("report", result.outputs["failure"])

    def test_existing_checkpoint_enforcement_uses_hand_snapshot(self):
        graph = workflow(
            "verify",
            [
                HandOperation(
                    "hand.assign_roles", {"roles": dict.fromkeys([f.value for f in Finger], "FREE")}
                )
            ],
        )
        with tempfile.TemporaryDirectory() as p:
            graph.save(Path(p) / "workflow.json")
            cp = Path(p) / "checkpoints"
            cp.mkdir()
            (cp / "verify.py").write_text(
                'from gap.runtime.verify import Checkpoint\ndef loaded(world):\n    s=world.sim_state["dexterous_hand"]\n    return len(s["fingers"]["thumb"]["contacts"]) > 0\nCHECKPOINTS=[Checkpoint(name="loaded_thumb",subgraph="verify",predicate=loaded)]\n'
            )
            result = gap.execute(
                p,
                connector=self.conn,
                skills=[],
                trace_dir=Path(p) / "trace",
                checkpoints="raise",
                max_node_workers=1,
            )
        self.assertFalse(result.success)
        self.assertEqual(len(result.checkpoint_results), 1)
        self.assertFalse(result.checkpoint_results[0].passed)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
