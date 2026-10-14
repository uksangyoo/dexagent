import contextlib
import unittest
from types import SimpleNamespace

from helpers import state

from dexagent.experiments.abstraction_ablation import manifest, run_trial
from dexagent.experiments.representations import Representation
from dexagent.tasks.common import TaskVerdict


class TaskMonitorTests(unittest.TestCase):
    def test_drop_inside_long_action_stops_before_recovery(self):
        class Env:
            dt = 0.02
            time_s = 0.0
            listener = None
            value = state().model_copy(update={"time_s": 0.0})

            def observe(self):
                return self.value.model_copy(deep=True)

            @contextlib.contextmanager
            def observe_each_step(self, listener):
                self.listener = listener
                try:
                    yield
                finally:
                    self.listener = None

        env = Env()

        class Adapter:
            kind = Representation.JOINTS
            max_action_seconds = 5.0
            reached_recovery = False

            def execute(self, payload):
                env.time_s = 0.002
                env.value.time_s = 0.002
                env.value.object_state.pose_palm.position = (1.0, 0.0, 0.0)
                env.listener(env.observe())
                self.reached_recovery = True
                raise AssertionError("Episode must end at the drop, before the next substep")

        class Task:
            def update(self, observation):
                dropped = observation.object_state.pose_palm.position[0] > 0.5
                return TaskVerdict(
                    False, dropped, {"dropped": dropped}, "dropped" if dropped else "running"
                )

        adapter = Adapter()
        adapter.env = env
        row = manifest(cases_per_task=1, agent_seeds=(0,))[0]
        result = run_trial(
            row,
            adapter,
            Task(),
            SimpleNamespace(propose=lambda *a: {"delta": [0.0] * 20}),
            available_capabilities={"free_rigid_object", "solved_contact_forces"},
            initialization_valid=True,
            provenance={
                "agent_model": "test",
                "agent_prompt_hash": "test",
                "agent_seed": 0,
                "controller_hash": "test",
                "scene_hash": "test",
                "observation_regime": row["observation_regime"],
            },
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["task_metrics"], {"dropped": True})
        self.assertAlmostEqual(result["simulated_seconds"], 0.002)
        self.assertEqual(result["reports"][0]["failure_code"], "object_dropped")
        self.assertIsNone(env.listener)
        self.assertFalse(adapter.reached_recovery)
