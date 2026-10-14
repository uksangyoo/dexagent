import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from helpers import contact, state
from scipy.spatial.transform import Rotation
from test_gap import EnvDouble

from dexagent.controllers.contact_controller import ContactController
from dexagent.controllers.fingertip_ik import integrate_ik_target
from dexagent.controllers.guards import violation
from dexagent.experiments.abstraction_ablation import Budget, learning_curve, manifest, summarize
from dexagent.experiments.representations import RepresentationAdapter, SynergyBasis
from dexagent.gap.feedback import report
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import Constraints, FailureCode, Pose, TransitionRequest
from dexagent.sim.surface_geometry import primitive_distance, project_object_surface
from dexagent.tasks.cable_wrapping import CableMeasurement, CableWrapping, wrapped_angle
from dexagent.tasks.common import TaskKind, require_capabilities, sample_case
from dexagent.tasks.nut_reorientation import NutReorientation, build_scene


class TransferProbe(ContactController):
    def __init__(self, env, receiver_success):
        super().__init__(env)
        self.calls = []
        self.receiver_success = receiver_success

    def execute(self, request):
        self.calls.append(request.operation)
        before = self.env.observe()
        if request.operation == "establish_contact":
            if self.receiver_success:
                self.env.value.fingers[request.finger] = state(
                    contacts={request.finger: [contact()]}
                ).fingers[request.finger]
            return report(
                before,
                self.env.observe(),
                success=self.receiver_success,
                code=None if self.receiver_success else FailureCode.TIMEOUT,
            )
        if request.operation == "break_contact":
            self.env.value.fingers[request.finger] = state().fingers[request.finger]
            self.env.roles[request.finger] = FingerRole.FREE
            return report(before, self.env.observe(), success=True)
        raise AssertionError(request.operation)


class TransferTests(unittest.TestCase):
    def test_release_rejects_unbalanced_support_even_with_two_loaded_fingers(self):
        env = EnvDouble()
        env.roles = dict.fromkeys(Finger, FingerRole.SUPPORT)
        env.value = state(contacts={f: [contact()] for f in (Finger.THUMB, Finger.INDEX, Finger.MIDDLE)},
                          roles=env.roles)
        env.value.object_state.free_body = True
        env.support_certificate = lambda **kwargs: SimpleNamespace(
            feasible=False, reason="remaining contacts cannot balance gravity torque"
        )
        result = ContactController(env).execute(TransitionRequest(
            operation="break_contact", object="hex_nut", finger=Finger.THUMB,
            constraints=Constraints(max_normal_force_N=5),
        ))
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.INFEASIBLE)
        self.assertIn("support-wrench", result.message)
        self.assertEqual(env.moves, 0)

    def test_soft_interpenetration_is_not_accepted_as_valid_contact(self):
        c = contact(force=0.3)
        c.separation_m = -0.005
        s = state(contacts={Finger.INDEX: [c]})
        bad = violation(s, Constraints(max_normal_force_N=5, min_support_contacts=0))
        self.assertEqual(bad[0], FailureCode.OBJECT_PENETRATION)

    def test_ik_target_integration_preserves_servo_load_error(self):
        env = EnvDouble()
        env.last_targets = env.kin_model.key_qpos[0].copy()
        env.actuated_joint_ids = list(env.kin_model.actuator_trnid[:, 0])
        env.actuated_q = np.array([env.kin_model.jnt_qposadr[j] for j in env.actuated_joint_ids])
        env.finger_q = {f: np.arange(6 + 4 * i, 10 + 4 * i) for i, f in enumerate(Finger)}
        actual = env.last_targets.copy()
        actual[10] += 0.01
        # Zero resolved increment must HOLD the existing actuator target,
        # retaining its error against load rather than drifting to measured q.
        command = integrate_ik_target(env, actual.copy(), actual)
        np.testing.assert_allclose(command, env.last_targets)

    def test_failed_receiver_never_releases_donor(self):
        for receiver_success in (False, True):
            env = EnvDouble()
            env.time_s = 0.0
            env.roles = {
                f: (FingerRole.SUPPORT if f in (Finger.THUMB, Finger.MIDDLE) else FingerRole.FREE)
                for f in Finger
            }
            env.value = state(
                contacts={Finger.THUMB: [contact()], Finger.MIDDLE: [contact()]}, roles=env.roles
            )
            controller = TransferProbe(env, receiver_success)
            request = TransitionRequest(
                operation="transfer_support",
                object="hex_nut",
                finger="thumb",
                to_finger="index",
                object_point=(0, 0, 0),
                constraints=Constraints(max_normal_force_N=5),
            )
            result = ContactController.execute(controller, request)
            self.assertEqual(
                controller.calls,
                ["establish_contact", "break_contact"]
                if receiver_success
                else ["establish_contact"],
            )
            self.assertEqual(result.success, receiver_success)
            if not receiver_success:
                self.assertEqual(env.roles[Finger.THUMB], FingerRole.SUPPORT)
                self.assertTrue(env.observe().fingers[Finger.THUMB].contacts)


class BenchmarkTests(unittest.TestCase):
    def test_edge_normal_uses_approach_and_ignores_visual_geoms(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="object">
            <geom type="box" size=".01 .01 .01"/>
            <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
            </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for perturbation in (-0.00004, 0, 0.00004):
            point, normal = project_object_surface(
                model,
                data,
                model.body("object").id,
                (0.01 + perturbation, 0.003, 0.01),
                approach_from=(0.013, 0.003, 0.0109),
            )
            np.testing.assert_allclose(normal, (1, 0, 0))
            self.assertAlmostEqual(point[0], 0.01)
            self.assertAlmostEqual(primitive_distance(model, data, 0, point), 0)

    def test_contact_proposal_lands_on_exterior_object_surface(self):
        model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).resolve().parents[1] / "runs/assets/scene.xml")
        )
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        mujoco.mj_forward(model, data)
        body = model.body("nut").id
        outside = np.array([0.009150937, -0.034352959, 0.003862786])
        query = data.xpos[body] + data.xmat[body].reshape(3, 3) @ outside
        projected, normal = project_object_surface(model, data, body, query)
        distances = [
            primitive_distance(model, data, g, projected)
            for g in np.flatnonzero(model.geom_bodyid == body)
        ]
        self.assertAlmostEqual(min(distances), 0.0, places=8)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=8)

    def test_fixture_nut_cannot_enter_free_object_benchmark(self):
        initial = state()
        initial.object_state.free_body = False
        with self.assertRaises(ValueError):
            NutReorientation(initial, 30)
        with self.assertRaises(ValueError):
            require_capabilities(TaskKind.CABLE, {"free_rigid_object"})

    def test_free_nut_requires_loaded_fingers_and_settling(self):
        initial = state()
        task = NutReorientation(initial, 30)
        s = state(time=0.5)
        s.object_state.pose_palm = Pose(
            position=(0, 0, 0.1),
            quaternion_wxyz=tuple(
                Rotation.from_euler("z", 30, degrees=True).as_quat(scalar_first=True)
            ),
        )
        self.assertFalse(task.update(s).success)
        s.fingers[Finger.THUMB] = state(contacts={Finger.THUMB: [contact()]}).fingers[Finger.THUMB]
        s.fingers[Finger.INDEX] = state(contacts={Finger.INDEX: [contact()]}).fingers[Finger.INDEX]
        self.assertFalse(task.update(s).success)
        s.time_s = 0.8
        self.assertTrue(task.update(s).success)

    def test_free_nut_builder_changes_object_dofs_not_actuators(self):
        case = sample_case(TaskKind.NUT, "train", 0)
        with tempfile.TemporaryDirectory() as p:
            path = build_scene(
                Path(__file__).resolve().parents[1] / "runs/assets/scene.xml", case, p
            )
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(model.nq, 33)
            self.assertEqual(model.nv, 32)
            self.assertEqual(model.nu, 26)
            self.assertEqual(model.joint("object_free").type[0], mujoco.mjtJoint.mjJNT_FREE)

    def test_wrap_is_contiguous_and_tension_is_not_inferred(self):
        theta = np.linspace(0, 1.5 * np.pi, 50)
        points = np.column_stack([0.03 * np.cos(theta), 0.03 * np.sin(theta), np.zeros(50)])
        self.assertAlmostEqual(wrapped_angle(points, 0.028, 0.002), 1.5 * np.pi)
        points[20:30, :2] *= 10
        self.assertLess(wrapped_angle(points, 0.028, 0.002), np.pi)
        with self.assertRaises(ValueError):
            CableWrapping(CableMeasurement(0, 1, 0, False, False), 0.1, 3, (0.5, 2))


class ExperimentTests(unittest.TestCase):
    def test_low_level_horizon_shares_physical_budget_and_stops_on_failure(self):
        env = EnvDouble()
        env.dt = 0.02
        env.time_s = 0.0

        class ServoProbe(RepresentationAdapter):
            fail_at = None

            def _execute(self, payload):
                before = self.env.observe()
                self.env.time_s += self.env.dt
                self.env.value.time_s = self.env.time_s
                self.env.moves += 1
                failed = self.env.moves == self.fail_at
                return report(
                    before,
                    self.env.observe(),
                    success=not failed,
                    steps=1,
                    code=FailureCode.FORCE_LIMIT if failed else None,
                )

        adapter = ServoProbe("A", SimpleNamespace(env=env), Constraints(max_normal_force_N=5))
        adapter.remaining_simulated_seconds = 0.07
        result = adapter.execute({"delta": [0.0] * 20, "duration_s": 1.0})
        self.assertTrue(result.success)
        self.assertEqual(result.control_steps, 3)
        self.assertAlmostEqual(env.time_s, 0.06)
        adapter.remaining_simulated_seconds = 1.0
        adapter.fail_at = env.moves + 2
        result = adapter.execute({"delta": [1.0] * 20, "duration_s": 1.0})
        self.assertFalse(result.success)
        self.assertEqual(result.control_steps, 2)
        self.assertAlmostEqual(result.elapsed_s, 0.04)

    def test_grammar_cannot_relax_shared_slip_constraint(self):
        from helpers import action_dict
        from test_gap import TestConnector

        env = EnvDouble()
        adapter = RepresentationAdapter("F", TestConnector(env), Constraints(max_normal_force_N=5))
        payload = action_dict()
        payload["constraints"]["max_slip_m_s"] = 0.5
        result = adapter.execute({"operation": "move_object", "action": payload})
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.INVALID_ACTION)
        self.assertIn("shared physical limits", result.message)

    def test_timeout_is_not_reported_as_a_physical_constraint_violation(self):
        result = report(state(), state(), success=False, code=FailureCode.TIMEOUT)
        self.assertFalse(result.constraint_violations)
        self.assertTrue(result.repair_hints)

    def test_paired_summary_rejects_controller_confound(self):
        rows = manifest(cases_per_task=1, agent_seeds=(0,), splits=("test_id",))[:6]
        for r in rows:
            r.update(
                status="completed",
                success=False,
                improvement_iteration=0,
                tool_calls=1,
                feasible_actions=1,
                failed_actions=1,
                recoveries=0,
                task_recovered_after_failure=False,
                provenance={
                    "controller_hash": "same",
                    "scene_hash": "same",
                    "agent_model": "same",
                    "agent_seed": 0,
                    "observation_regime": "state",
                    "action_horizon_s": 5.0,
                },
            )
        output = summarize(rows)
        self.assertEqual(len(output), 1)
        rows[-1]["provenance"]["controller_hash"] = "different"
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_case_pairing_and_disjoint_generalization(self):
        rows = manifest(cases_per_task=1, agent_seeds=(0,))
        self.assertEqual(len(rows), 4 * 2 * 6)
        by_case = {}
        for r in rows:
            by_case.setdefault(r["case"]["case_id"], []).append(r)
        for block in by_case.values():
            self.assertEqual({r["method"] for r in block}, set("ABCDEF"))
            self.assertTrue(all(r["case"] == block[0]["case"] for r in block))
        train = sample_case(TaskKind.NUT, "train", 0)
        ood = sample_case(TaskKind.NUT, "test_ood", 0)
        self.assertLess(train.dimension_scale, ood.dimension_scale)
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_synergies_require_training_data_and_nonzero_rank(self):
        q = np.random.default_rng(0).normal(size=(100, 20))
        with self.assertRaises(ValueError):
            SynergyBasis.fit(q, ["case"] * 100, split="test_id")
        basis = SynergyBasis.fit(q, ["case"] * 100, split="train")
        np.testing.assert_allclose(basis.basis.T @ basis.basis, np.eye(6), atol=1e-12)
        self.assertEqual(basis.decode(np.zeros(6)).shape, (20,))
        with self.assertRaises(ValueError):
            SynergyBasis.fit(np.zeros((100, 20)), ["case"] * 100, split="train")

    def test_learning_curve_never_repairs_with_held_out_feedback(self):
        received = []
        result = learning_curve(
            lambda i: {"train": i},
            lambda i: {"test": i},
            lambda evidence, i: received.append(evidence),
            budget=Budget(improvement_iterations=2),
        )
        self.assertEqual(received, [{"train": 0}, {"train": 1}])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
