"""Opt-in real physics validation; never silently allocates a GPU in CPU tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import Constraints, Termination, TransitionRequest


@unittest.skipUnless(
    os.environ.get("DEXAGENT_GPU_TESTS") == "1",
    "Set DEXAGENT_GPU_TESTS=1 and select CUDA_VISIBLE_DEVICES",
)
class NewtonContactTests(unittest.TestCase):
    def test_free_body_state_round_trip(self):
        import mujoco
        import numpy as np

        from dexagent.sim.object_state import matrix
        from dexagent.sim.ur5e_tesollo_env import ROOT, UR5eTesolloEnv
        from dexagent.tasks.common import TaskKind, sample_case
        from dexagent.tasks.nut_reorientation import build_scene

        with tempfile.TemporaryDirectory() as directory:
            scene = build_scene(
                ROOT / "runs/assets/scene.xml", sample_case(TaskKind.NUT, "test_id", 1), directory
            )
            env = UR5eTesolloEnv(scene)
            for _ in range(5):
                env.step_targets(env.last_targets)
            observed = env.observe()
            self.assertTrue(observed.object_state.free_body)
            self.assertFalse(observed.object_state.externally_supported)
            data = mujoco.MjData(env.kin_model)
            data.qpos[:] = env.native_q(env.sample)
            mujoco.mj_forward(env.kin_model, data)
            body = env.kin_model.body("nut").id
            transform = matrix(observed.object_state.pose_world)
            np.testing.assert_allclose(transform[:3, 3], data.xpos[body], atol=2e-6)
            np.testing.assert_allclose(transform[:3, :3], data.xmat[body].reshape(3, 3), atol=2e-5)
            self.assertTrue(np.isfinite(env.native_q()).all())
            env.close()

    def test_establish_maintain_release_on_passive_fixture(self):
        from dexagent.controllers.contact_controller import ContactController
        from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv

        env = UR5eTesolloEnv()
        controller = ContactController(env)
        finger = Finger.INDEX
        # Fix the task input. Nearest-surface proposals can choose different
        # facets and are a separate reachability problem, not this regression.
        point, _, _ = env.contact_target(finger, (-0.022, 0.015, 0.010))
        env.roles[finger] = FingerRole.SUPPORT
        constraints = Constraints(
            max_normal_force_N=5.0,
            min_support_contacts=0,
            calibration_id="fixture_bootstrap_provisional",
        )
        termination = Termination(timeout_s=4.0, hold_s=0.2, stable_steps=5)
        reports = []
        for operation in ("establish_contact", "maintain_contact", "break_contact"):
            request = TransitionRequest(
                operation=operation,
                object="hex_nut",
                finger=finger,
                object_point=tuple(point) if operation == "establish_contact" else None,
                constraints=constraints,
                termination=termination,
            )
            result = controller.execute(request)
            reports.append(result.model_dump(mode="json"))
            Path("runs/gpu_contact_sequence.json").write_text(json.dumps(reports, indent=2) + "\n")
            self.assertTrue(result.success, f"{operation}: {result.failure_code}: {result.message}")
            if operation != "break_contact":
                self.assertGreaterEqual(
                    sum(c.normal_force_N or 0.0 for c in env.observe().fingers[finger].contacts),
                    0.05,
                )
        self.assertFalse(env.observe().fingers[finger].contacts)
        self.assertEqual(env.roles[finger], FingerRole.FREE)
        env.close()


if __name__ == "__main__":
    unittest.main()
