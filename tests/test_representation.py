import unittest

import numpy as np
from helpers import action_dict, contact, state
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from dexagent.controllers.guards import supporting_fingers, violation
from dexagent.grammar.action_compiler import InfeasibleAction, compile_action
from dexagent.grammar.finger_roles import ContactMode, Finger, FingerRole
from dexagent.grammar.schema import (
    Constraints,
    DexterousAction,
    FailureCode,
    Pose,
    TransitionRequest,
    Twist,
)
from dexagent.sim.contact_observer import ContactObserver
from dexagent.sim.object_state import matrix, point_velocity, relative_object_state


class SchemaTests(unittest.TestCase):
    def test_illustrative_action_round_trip(self):
        a = DexterousAction.model_validate(action_dict())
        self.assertEqual(DexterousAction.model_validate_json(a.model_dump_json()), a)

    def test_bad_actions_fail_before_execution(self):
        for change in [
            lambda a: a["target"].update(axis=[0, 0, 0]),
            lambda a: a["target"].update(angle_deg=float("nan")),
            lambda a: a.update(mode="translate"),
            lambda a: a["finger_roles"].pop("pinky"),
            lambda a: a["constraints"].update(max_normal_force_N=-1),
            lambda a: a.update(unknown=1),
            lambda a: a["constraints"].pop("max_normal_force_N"),
        ]:
            d = action_dict()
            change(d)
            with self.assertRaises(ValidationError):
                DexterousAction.model_validate(d)

    def test_nonunit_quaternion_and_same_finger_transfer_rejected(self):
        with self.assertRaises(ValidationError):
            Pose(position=(0, 0, 0), quaternion_wxyz=(0, 0, 0, 0))
        with self.assertRaises(ValidationError):
            TransitionRequest(
                operation="transfer_support",
                object="hex_nut",
                finger="thumb",
                to_finger="thumb",
                object_point=(0, 0, 0),
                constraints=Constraints(max_normal_force_N=5),
            )


class ContactMathTests(unittest.TestCase):
    def test_repeated_reads_do_not_change_an_established_rolling_label(self):
        observer = ContactObserver()
        for i in range(5):
            contacts = [contact(point=(i * 0.0001, 0, 0), time=i * 0.01, omega=1)]
            result = observer.classify(Finger.THUMB, contacts, i * 0.01)
        self.assertEqual(result[0], ContactMode.ROLLING)
        for _ in range(5):
            self.assertEqual(observer.classify(Finger.THUMB, contacts, 0.04), result)

    def test_com_offset_velocity(self):
        # COM at x=1; rotation around frame origin gives v_COM=(0,2,0).
        np.testing.assert_allclose(
            point_velocity(np.eye(4), [1, 0, 0], [0, 2, 0, 0, 0, 2], [2, 0, 0]), [0, 4, 0]
        )

    def test_object_twist_subtracts_palm_rotation(self):
        palm = np.eye(4)
        obj = np.eye(4)
        obj[:3, 3] = [1, 0, 0]
        s = relative_object_state(
            "x", palm, obj, Twist(angular=(0, 0, 2)), Twist(linear=(0, 2, 0), angular=(0, 0, 2))
        )
        np.testing.assert_allclose(s.twist_palm.linear, [0, 0, 0])
        np.testing.assert_allclose(s.twist_palm.angular, [0, 0, 0])

    def test_force_sign_invariant_to_shape_order(self):
        observer = ContactObserver()
        common = dict(
            points0=np.zeros((1, 3)),
            points1=np.zeros((1, 3)),
            body_transforms=[np.eye(4), np.eye(4)],
            body_velocities=np.zeros((2, 6)),
            body_com=np.zeros((2, 3)),
            shape_body=np.array([0, 1]),
            body_fingers={1: Finger.THUMB},
            body_names=["nut", "link_1_4"],
            object_body=0,
            object_name="hex_nut",
            palm=np.eye(4),
            sample_time_s=0.0,
            force_source="mujoco_warp_constraint",
        )
        a, _ = observer.from_arrays(
            **common,
            shapes0=[0],
            shapes1=[1],
            normals=[[1, 0, 0]],
            forces=np.array([[-2, 0, 0, 0, 0, 0]]),
        )
        b, _ = observer.from_arrays(
            **common,
            shapes0=[1],
            shapes1=[0],
            normals=[[-1, 0, 0]],
            forces=np.array([[2, 0, 0, 0, 0, 0]]),
        )
        self.assertEqual(a[Finger.THUMB][0].normal_force_N, 2.0)
        self.assertEqual(b[Finger.THUMB][0].normal_force_N, 2.0)
        self.assertEqual(a[Finger.THUMB][0].normal_world, b[Finger.THUMB][0].normal_world)
        # Object and finger material patches must stay distinct under normal
        # compression; the geometric midpoint is not an object material point.
        common["points1"] = np.array([[-0.001, 0.0, 0.0]])
        compressed, _ = observer.from_arrays(
            **common,
            shapes0=[0],
            shapes1=[1],
            normals=[[1, 0, 0]],
            forces=np.array([[-2, 0, 0, 0, 0, 0]]),
        )
        c = compressed[Finger.THUMB][0]
        np.testing.assert_allclose(c.point_object, [0, 0, 0])
        np.testing.assert_allclose(c.point_finger, [-0.001, 0, 0])
        np.testing.assert_allclose(c.point_world, [-0.0005, 0, 0])

        # A zero-force penetration must remain visible to safety, without
        # manufacturing a load-bearing contact state.
        common["points1"] = np.array([[-0.004, 0.0, 0.0]])
        unloaded, _ = observer.from_arrays(
            **common,
            shapes0=[0],
            shapes1=[1],
            normals=[[1, 0, 0]],
            forces=np.zeros((1, 6)),
        )
        self.assertEqual(len(unloaded[Finger.THUMB]), 1)
        self.assertAlmostEqual(unloaded[Finger.THUMB][0].separation_m, -0.004)
        self.assertEqual(
            observer.classify(Finger.THUMB, unloaded[Finger.THUMB], 1)[0], ContactMode.FREE
        )
        measured = state(contacts={Finger.THUMB: unloaded[Finger.THUMB]})
        self.assertEqual(
            violation(measured, Constraints(max_normal_force_N=5), require_support=False)[0],
            FailureCode.OBJECT_PENETRATION,
        )

    def test_candidate_without_load_does_not_count_as_contact(self):
        observer = ContactObserver()
        c = contact(force=0)
        self.assertEqual(observer.classify(Finger.THUMB, [c], 0)[0], ContactMode.FREE)

    def test_sliding_hysteresis_and_contact_loss(self):
        o = ContactObserver()
        self.assertEqual(o.classify(Finger.THUMB, [contact(slip=0.02)], 0)[0], ContactMode.SLIDING)
        self.assertEqual(
            o.classify(Finger.THUMB, [contact(slip=0.003)], 0.01)[0], ContactMode.SLIDING
        )
        self.assertEqual(o.classify(Finger.THUMB, [], 0.02)[0], ContactMode.FREE)

    def test_rolling_needs_both_material_migration_and_low_slip(self):
        o = ContactObserver()
        for i in range(5):
            c = contact(point=(i * 0.0001, 0, 0), time=i * 0.01, omega=1)
            result = o.classify(Finger.THUMB, [c], i * 0.01)
        self.assertEqual(result[0], ContactMode.ROLLING)
        o = ContactObserver()
        for i in range(5):
            result = o.classify(Finger.THUMB, [contact(omega=1)], i * 0.01)
        self.assertEqual(result[0], ContactMode.STICKING)

    def test_repeated_observation_cannot_confirm_rolling(self):
        o = ContactObserver()
        o.classify(Finger.THUMB, [contact(omega=1)], 0)
        for _ in range(5):
            result = o.classify(Finger.THUMB, [contact(point=(0.0101, 0, 0), omega=1)], 0.01)
        self.assertNotEqual(result[0], ContactMode.ROLLING)

    def test_unknown_force_is_not_zero_force(self):
        o = ContactObserver()
        result = o.classify(Finger.THUMB, [contact(force=None)], 0)
        self.assertLessEqual(result[1], 0.3)
        self.assertIn("force unavailable", " ".join(result[2]))


class CompilerTests(unittest.TestCase):
    def test_role_assignment_does_not_invent_support(self):
        roles = dict.fromkeys(Finger, FingerRole.SUPPORT)
        s = state(roles=roles)
        self.assertEqual(supporting_fingers(s, Constraints(max_normal_force_N=5)), [])
        with self.assertRaises(InfeasibleAction):
            compile_action(DexterousAction.model_validate(action_dict()), s)

    def test_object_frame_rotation_and_finite_contact_targets(self):
        a = DexterousAction.model_validate(action_dict())
        s = state(
            contacts={f: [contact()] for f in [Finger.THUMB, Finger.INDEX, Finger.MIDDLE]},
            roles=a.finger_roles,
        )
        initial = Rotation.from_euler("x", 90, degrees=True)
        s.object_state.pose_world = Pose(
            position=(0, 0, 0.1), quaternion_wxyz=tuple(initial.as_quat(scalar_first=True))
        )
        compiled = compile_action(a, s)
        expected = initial * Rotation.from_euler("z", 30, degrees=True)
        np.testing.assert_allclose(
            matrix(compiled.target_world)[:3, :3], expected.as_matrix(), atol=1e-8
        )
        self.assertEqual(len(compiled.point_targets(matrix(compiled.target_world))), 3)

    def test_summed_per_finger_force_limit(self):
        s = state(contacts={Finger.THUMB: [contact(force=3), contact(force=3, pair=(2, 3))]})
        self.assertEqual(
            violation(s, Constraints(max_normal_force_N=5))[0], FailureCode.FORCE_LIMIT
        )

    def test_non_target_contacts_cannot_supply_support(self):
        o = ContactObserver()
        result, _ = o.from_arrays(
            points0=np.zeros((1, 3)),
            points1=np.zeros((1, 3)),
            normals=np.array([[1, 0, 0]]),
            shapes0=[0],
            shapes1=[1],
            forces=np.array([[-1, 0, 0, 0, 0, 0]]),
            shape_body=[0, 1],
            body_transforms=[np.eye(4)] * 3,
            body_velocities=np.zeros((3, 6)),
            body_com=np.zeros((3, 3)),
            body_fingers={0: Finger.THUMB, 1: Finger.INDEX},
            body_names=["a", "b", "nut"],
            object_body=2,
            object_name="hex_nut",
            palm=np.eye(4),
            sample_time_s=0,
            force_source="mujoco_warp_constraint",
        )
        self.assertTrue(all(not cs for cs in result.values()))


if __name__ == "__main__":
    unittest.main()
