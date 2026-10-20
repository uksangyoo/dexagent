"""Contact-state handoff from a physical pickup to the loaded-slip spin skill."""
import numpy as np
from screwdriver_lab.controlled_slip import SlipPairedController
class CapturedSlipController(SlipPairedController):
 def __init__(self,s,cap_offset=0.,**options):
  super().__init__(s,**options)
  if getattr(s,"control_interface","wrench")=="pose":
   s.set_hand_impedance(8.,.18)
  axis=s.d.xmat[s.tool].reshape(3,3)[:,2];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex)
  for slot,i in enumerate([0,2]):
   if s.handleloads[slot]<.01:raise RuntimeError('Spin handoff requires both loaded side contacts')
   b=s.bids[i];R=s.d.xmat[b].reshape(3,3);point=s.pad_points[slot];delta=point-s.d.xpos[s.tool];n=delta-axis*np.dot(delta,axis);n/=np.linalg.norm(n)
   self.local[slot]=R.T@(point-s.d.xpos[b]);self.phi[slot]=float(np.arctan2(np.dot(delta,ey),np.dot(delta,ex)));self.height[slot]=float(np.dot(delta,axis));self.normal_local[slot]=R.T@(-n)
  self.cap=cap_offset;self.force_targets=np.asarray(s.handleloads).copy()
  self.posture_ref=s.d.qpos[s.qi].copy()
  self.drive_center=s.d.xpos[s.tool].copy();self.drive_center[:2]=s.d.xpos[s.screw,:2];self.drive_depth=float(s.d.qpos[s.depthq])

 def torques(self):
  if getattr(self.s,'control_interface','wrench')!='pose':super().torques()
 def tick(self,o):
  q,command=super().tick(o)
  if getattr(self.s,'control_interface','wrench')=='pose' and self.phase!='seat':
   q,_=self.pose_command(o,update_cap=False)
  return q,command
