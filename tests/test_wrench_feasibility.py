import unittest
from types import SimpleNamespace

from dexagent.controllers.wrench_feasibility import certify_support


def contact(point, normal):
    return SimpleNamespace(
        point_world=point, normal_world=normal, normal_force_N=0.2, separation_m=0.0
    )


class WrenchFeasibilityTests(unittest.TestCase):
    def test_opposing_contacts_need_friction_to_support_gravity(self):
        contacts = {
            "thumb": [contact((-0.02, 0, 0), (-1, 0, 0))],
            "index": [contact((0.02, 0, 0), (1, 0, 0))],
        }
        params = dict(center_of_mass_world=(0, 0, 0), mass_kg=0.1, max_force_per_finger_N=2)
        self.assertFalse(certify_support(contacts, friction=0, **params).feasible)
        certificate = certify_support(contacts, friction=0.5, **params)
        self.assertTrue(certificate.feasible)
        self.assertLess(certificate.residual, 1e-6)
        self.assertFalse(
            certify_support(
                contacts, friction=0.5, **{**params, "max_force_per_finger_N": 0.5}
            ).feasible
        )

    def test_same_contact_count_different_wrench_capacity(self):
        contacts = {
            "index": [contact((-0.02, 0, 0), (0, 0, -1))],
            "middle": [contact((0.02, 0, 0), (0, 0, -1))],
        }
        params = dict(
            center_of_mass_world=(0, 0, 0), mass_kg=0.1, max_force_per_finger_N=2, friction=0.3
        )
        self.assertTrue(certify_support(contacts, **params).feasible)
        self.assertFalse(certify_support(contacts, disturbance_force=(2, 0, 0), **params).feasible)
