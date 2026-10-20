"""Five command contracts above the same bounded finger servos and axial feed.

Joint and synergy baselines replay a nominal single-turn calibration. This is
explicit task calibration, not learning from the human video. Other adapters
solve current fingertip geometry; only the contact policy renews contacts.
"""
import json
import numpy as np
from screwdriver_lab.screw_scene import OUT

METHODS={
 'joints':dict(name='Joint targets',contract='20 joint positions',advantage='Direct and inspectable motor commands.',cost='The policy must supply coordinated joint trajectories; the nominal turn has a finite range.'),
 'synergy':dict(name='Motor synergies',contract='3 PCA coefficients → 20 joints',advantage='Compresses the nominal turn into three motor coordinates.',cost='The basis only represents the calibrated grasp and single-turn motion.'),
 'fingertip':dict(name='Fingertip motion',contract='3 Cartesian contact targets',advantage='Specifies cap support and tangential handle motion geometrically.',cost='Local inverse kinematics does not decide when or how to regrasp.'),
 'object':dict(name='Screw motion',contract='Clockwise screw-angle target',advantage='Corrects local motion using measured screw progress.',cost='Angle feedback alone cannot renew contacts after the finger workspace runs out.'),
 'contact':dict(name='Contact skill',contract='Press → turn → release → return → regrasp → seat',advantage='Converts a long screw-in objective into repeated short, contact-aware turns.',cost='Requires task-specific contact roles, transition rules and an engagement model.')
}
class Adapter:
 def __init__(self,method):
  if method not in METHODS:raise ValueError(method)
  self.method=method
  if method in ('joints','synergy'):
   b=np.load(OUT/'motor_bank.npz');self.angles=b['angles'];self.q=b['q'];self.mean=self.q.mean(0);_,_,v=np.linalg.svd(self.q-self.mean,full_matrices=False);self.basis=v[:3]
 def command(self,s,angle,release,phase):
  if phase=='press' or self.method not in ('joints','synergy'):return s.ik(angle,release)
  q=np.array([np.interp(angle,self.angles,self.q[:,j]) for j in range(20)])
  if self.method=='synergy':q=self.mean+((q-self.mean)@self.basis.T)@self.basis
  return q

def calibrate():
 from screwdriver_lab.screw_sim import ScrewSim
 s=ScrewSim();angles=[];qs=[]
 for i in range(75):
  events=s.step(s.ik())
  if events:raise RuntimeError(events)
 for angle in np.linspace(0,30,51):
  q=s.ik(float(angle));events=s.step(q)
  if events:raise RuntimeError(events)
  angles.append(angle);qs.append(q)
 np.savez_compressed(OUT/'motor_bank.npz',angles=angles,q=qs)
 print('Calibrated single-turn motor bank',s.observe())
if __name__=='__main__':calibrate()
