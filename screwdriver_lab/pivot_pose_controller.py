from screwdriver_lab.pivot_spin_controller import CapturedSlipController
class CapturedPoseController(CapturedSlipController):
 """Use the same graph and observed handoff, with contact-pose IK commands."""
 def torques(self):pass
 def tick(self,o):
  super().tick(o)
  return self.pose_command(o,update_cap=False)
