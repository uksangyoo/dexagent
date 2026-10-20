"""Targeted physics/interface regression checks; run with unittest."""
import unittest
from copy import deepcopy
import mujoco
import numpy as np
from screwdriver_lab.scene import build_scene
from screwdriver_lab.simulator import Simulator
from screwdriver_lab.run import acquisition_ok,success_sample,run_trial,CASES
from screwdriver_lab.controllers import training_bank
from screwdriver_lab.export_rtx import export
import tempfile
from screwdriver_lab.policy import Policy

class PipelineTests(unittest.TestCase):
    def test_tool_is_unactuated_and_hand_has_twenty_servos(self):
        for fixed,nq,nv in [(False,27,26),(True,21,21)]:
            m=mujoco.MjModel.from_xml_string(build_scene(fixture=fixed))
            self.assertEqual((m.nu,m.nq,m.nv),(20,nq,nv))
            self.assertEqual(set(m.actuator_trnid[:,0]),set(range(20)))
            self.assertEqual(m.jnt_type[-1],mujoco.mjtJoint.mjJNT_HINGE if fixed else mujoco.mjtJoint.mjJNT_FREE)
    def test_pose_goal_cannot_pass_without_retention_and_settling(self):
        good=dict(angle_deg=90,speed_rad_s=.1,drift_mm=1,tilt_deg=1,forces=[1,1,0,0,0],support_forces=[1,1,0,0,0],depth_mm=.1,external_N=0,forbidden_contact_N=0,shaft_clearance_mm=18)
        self.assertTrue(success_sample(good))
        for key,value in [('angle_deg',94),('angle_deg',float('nan')),('speed_rad_s',.21),('drift_mm',21),('tilt_deg',21),('support_forces',[1,0,0,0,0]),('forces',[9,1,0,0,0]),('depth_mm',.6),('external_N',.1),('shaft_clearance_mm',2.9),('forbidden_contact_N',.06)]:
            with self.subTest(key=key,value=value):
                bad=deepcopy(good);bad[key]=value;self.assertFalse(success_sample(bad))
    def test_initialization_rejects_slip_shaft_or_proximal_support(self):
        good=dict(speed_rad_s=.01,drift_mm=1.5,tilt_deg=5,forces=[.1,0,.5,0,0],support_forces=[.1,0,.5,0,0],depth_mm=.1,external_N=0,forbidden_contact_N=0,shaft_clearance_mm=18)
        self.assertTrue(acquisition_ok(good))
        for key,value in [('drift_mm',35),('external_N',.1),('forbidden_contact_N',.1),('support_forces',[0,0,0,0,0]),('shaft_clearance_mm',2.9),('tilt_deg',11)]:
            with self.subTest(key=key):
                bad=deepcopy(good);bad[key]=value;self.assertFalse(acquisition_ok(bad))
    def test_nominal_reset_has_clear_shaft(self):
        s=Simulator({});o=s.observe()
        self.assertGreater(o['shaft_clearance_mm'],3)
        self.assertLess(o['depth_mm'],.2)
        self.assertLess(o['forbidden_contact_N'],.05)
    def test_acquisition_violation_is_latched_and_recorded(self):
        r,q,_=run_trial(CASES[1],'object','free',training_bank(),goal=15)
        self.assertFalse(r['valid_initialization'])
        self.assertEqual(r['outcome'],'invalid_initialization')
        self.assertIn('acquisition_slip',r['acquisition_events'])
        self.assertEqual(r['metrics']['policy_calls'],0)
        self.assertEqual(r['frames'][0]['t'],0)
        self.assertTrue(all(f['phase']=='acquire' for f in r['frames']))
        self.assertEqual(len(q),len(r['frames']))
    def test_renderer_export_preserves_geometry_and_world_pose(self):
        s=Simulator({})
        with tempfile.TemporaryDirectory() as tmp:
            out=export(s.xml,[s.data.qpos],tmp)
            arr=np.load(out/'scene.npz')
            np.testing.assert_allclose(arr['transforms'][0,:,:3],s.data.geom_xpos,atol=1e-12)
            np.testing.assert_array_equal(arr['verts_0'],s.model.mesh_vert[:s.model.mesh_vertnum[0]])
    def test_joint_target_limits_and_no_object_write(self):
        s=Simulator({});before=s.data.qpos[20:].copy();target=s.bound(np.ones(20)*100)
        self.assertTrue(np.all(target<=s.model.jnt_range[:20,1]));self.assertTrue(np.all(target>=s.model.jnt_range[:20,0]))
        np.testing.assert_array_equal(s.data.qpos[20:],before)
    def test_free_tool_falls_without_contacts(self):
        s=Simulator({});s.model.geom_contype[:]=0;s.model.geom_conaffinity[:]=0
        start=s.data.qpos[22]
        for _ in range(100):mujoco.mj_step(s.model,s.data)
        self.assertLess(s.data.qpos[22],start-.1)
    def test_passive_hinge_does_not_turn_without_hand_contact(self):
        s=Simulator({},fixture=True);s.model.geom_contype[:]=0;s.model.geom_conaffinity[:]=0
        for _ in range(100):mujoco.mj_step(s.model,s.data)
        self.assertAlmostEqual(s.data.qpos[-1],0.,places=10)
    def test_policy_requires_receiving_contact_or_exhausts_budget(self):
        p=Policy('contact');o=dict(angle_deg=20,speed_rad_s=.1,forces=[0,0,0,0,0])
        self.assertEqual(p.decide(o,1).phase,'regrasp')
        p.decide(o,1.7);p.decide(o,2.4)
        self.assertEqual(p.decide(o,3.1).phase,'abort')
    def test_policy_hold_does_not_claim_success(self):
        p=Policy('object');decision=p.decide(dict(angle_deg=90,speed_rad_s=.1,forces=[1]*5),5)
        self.assertEqual(decision.phase,'hold')
        # The independent runner, rather than this decision, owns the 0.4 s gate.
        self.assertFalse(hasattr(decision,'success'))

if __name__=='__main__':unittest.main()
