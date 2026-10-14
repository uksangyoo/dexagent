"""Physical asset checks for the motor curriculum, independent of training."""

import unittest
from pathlib import Path

import mujoco
import numpy as np


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/"runs/assets/scene.xml"))
        self.data=mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data)

    def test_nut_is_passive_and_hand_has_twenty_joints(self):
        self.assertEqual(self.model.nv,27)
        self.assertEqual(self.model.nu,26)
        self.assertFalse(np.any(self.model.actuator_trnid[:,0]==self.model.joint("nut_rotation").id))

    def test_no_severe_initial_penetration(self):
        self.assertGreater(min((c.dist for c in self.data.contact),default=0),-.001)

    def test_without_contact_nut_cannot_be_rotated_by_controls(self):
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        for t in range(200):
            self.data.ctrl[6:26]+=0.002*np.sin(t*.1)
            mujoco.mj_step(self.model,self.data)
        self.assertLess(abs(self.data.qpos[-1]),1e-5)
        self.assertTrue(np.isfinite(self.data.qpos).all())


if __name__=="__main__":unittest.main()
