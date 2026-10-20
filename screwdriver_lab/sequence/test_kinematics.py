"""Regression for the arm velocity being counted twice during pickup."""
import unittest
import numpy as np,mujoco
from .geometry import ART
from .kinematics import finger_velocity_target,tracked_contact_velocity,palm_velocity_for_tip,palm_velocity_for_slipping_tool
from .metrics import valid_thread_motion
class MovingArmTests(unittest.TestCase):
 def test_axial_push_without_thread_rotation_cannot_pass(self):
  self.assertTrue(valid_thread_motion(504,.7))
  self.assertFalse(valid_thread_motion(0,.7))
  self.assertFalse(valid_thread_motion(45,.7))
  self.assertFalse(valid_thread_motion(-504,.7))
  self.assertFalse(valid_thread_motion(float('nan'),.7))
 def test_slipping_tool_does_not_follow_arm_rotation_rigidly(self):
  palm=np.zeros(3);center=np.array([.04,0,.02]);tip=center+np.array([0,0,-.123]);omega=np.array([.1,0,0]);target=np.zeros(3)
  command=palm_velocity_for_slipping_tool(target,omega,np.zeros(3),tip,center,palm)
  np.testing.assert_allclose(command+np.cross(omega,center-palm),target,atol=1e-14)
  rigid=palm_velocity_for_slipping_tool(target,omega,omega,tip,center,palm)
  np.testing.assert_allclose(rigid,palm_velocity_for_tip(target,omega,tip,palm),atol=1e-14)
  self.assertGreater(np.linalg.norm(command-rigid),.01)
 def test_rotating_arm_tracks_the_tip_without_translation_bias(self):
  tip=np.array([.1,-.02,.3]);palm=np.array([0,0,.5]);omega=np.array([.1,-.2,.3]);target=np.array([.003,-.004,0])
  palm_v=palm_velocity_for_tip(target,omega,tip,palm)
  np.testing.assert_allclose(palm_v+np.cross(omega,tip-palm),target,atol=1e-14)
  self.assertGreater(np.linalg.norm(palm_v-target),.01)
 def test_contact_tracking_is_invariant_to_common_translation(self):
  correction=np.array([.01,-.08,.002]);translation=np.array([.06,-.06,.03])
  stationary=tracked_contact_velocity(np.zeros(3),correction)
  carried=tracked_contact_velocity(translation,correction)
  np.testing.assert_allclose(carried-stationary,translation,atol=1e-14)
  np.testing.assert_allclose(tracked_contact_velocity(translation,np.zeros(3)),translation)
 def test_fixed_grasp_needs_no_finger_motion_during_arm_translation(self):
  m=mujoco.MjModel.from_xml_path(str(ART/'pickup-sequence-initial.xml'));d=mujoco.MjData(m);d.qpos[:]=np.load(ART/'pickup-initial.npz')['qpos'][0];mujoco.mj_forward(m,d)
  names=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3'];qi=np.array([m.joint(n).qposadr[0]for n in names]);vi=np.array([m.joint(n).dofadr[0]for n in names]);v=np.array([.1,-.2,.15,.05,-.04,.02]);eps=1e-7
  for finger,local in [(1,[.0466,-.0029,.0016]),(2,[.0357,0,.0016]),(3,[.0357,-.0029,.0016])]:
   b=m.body(f'link_{finger}_4').id;p0=d.xpos[b]+d.xmat[b].reshape(3,3)@local;J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,p0,b)
   advanced=mujoco.MjData(m);advanced.qpos[:]=d.qpos;advanced.qpos[qi]+=eps*v;mujoco.mj_forward(m,advanced);p1=advanced.xpos[b]+advanced.xmat[b].reshape(3,3)@local
   desired=(p1-p0)/eps;residual=finger_velocity_target(desired,J,vi,v)
   self.assertLess(np.linalg.norm(residual),1e-7)
   self.assertGreater(np.linalg.norm(desired),.01) # The old solver incorrectly commanded this through fingers too.
if __name__=='__main__':unittest.main()
