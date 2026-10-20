"""Paired controller interfaces executing the same GaP-style task graph.

The lower-level baselines share a nominal task calibration, not human motion
capture. Only their hand command adapter changes; arm control and guards match.
"""
import json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.full_scene import OUT
METHODS={
 'contact':dict(name='Contact skills',contract='Contact roles + force + turn/slide/regrasp',dimension='mode + continuous targets',description='Live force regulation, object alignment, material contact-point planning, impedance scheduling, and measured phase exits.'),
 'joints':dict(name='Joint trajectories',contract='20 hand joint position targets',dimension='20',description='Replay each nominal graph node’s calibrated motor trajectory; the graph still checks physical completion.'),
 'synergy':dict(name='Motor synergies',contract='3 PCA coordinates → 20 joints',dimension='3',description='Project the same nominal motor trajectory onto a three-dimensional calibrated hand synergy basis.'),
 'fingertip':dict(name='Fingertip poses',contract='3 fingertip poses → joint IK',dimension='18',description='Track calibrated fingertip position and orientation in the palm frame through current hand kinematics.'),
 'object':dict(name='Object motion',contract='Fingertip poses + object pose error',dimension='6 object pose + grasp calibration',description='Correct fingertip pose targets using measured tool pose error, while retaining the same graph and nominal grasp plan.')}

class Adapter:
 def __init__(self,method,s,bank=None):
  if method not in METHODS:raise ValueError(f'Unknown hand interface {method!r}; choose from {list(METHODS)}')
  self.method=method
  if method=='contact':return
  path=OUT/'abstraction-bank.npz' if bank is None else bank
  b=np.load(path);self.phases=b['phase'];self.times=b['elapsed'];self.q=b['q'];self.mean=b['mean'];self.basis=b['basis'];self.p=b['point'];self.rot=b['rotation'];self.tool_p=b['tool_p'];self.tool_R=b['tool_R']
  self.cycle=b['drive_cycle'] if 'drive_cycle' in b else None
  if method=='synergy' and 'basis_all' in b:self.basis=b['basis_all'][:int(s.case.get('synergy_rank',3))]
  self.kin=mujoco.MjData(s.m);self.palm=s.m.body('palm').id;self.tips=s.bids[:3];self.local=np.array(s.nominal_local)
  self.groups={name:np.flatnonzero(self.phases==name)for name in np.unique(self.phases)}
 def command(self,s,q,phase,elapsed):
  if self.method=='contact':return q
  ix=self.groups.get(phase)
  if ix is None:raise RuntimeError(f'Calibration has no graph node {phase}')
  if phase=='turn' and self.cycle is not None and elapsed>self.times[ix[-1]]+2.:
   elapsed=self.cycle[0]+(elapsed-self.times[ix[-1]]-2.)%(self.cycle[1]-self.cycle[0])
  k=int(ix[np.argmin(np.abs(self.times[ix]-elapsed))]);ref=self.q[k]
  if self.method=='joints':return s.bound(ref)
  if self.method=='synergy':return s.bound(self.mean+((ref-self.mean)@self.basis.T)@self.basis)
  self.kin.qpos[:]=s.d.qpos;result=s.d.qpos[s.qi].copy();goalP=self.p[k].copy();goalR=self.rot[k].copy();palmR=s.d.xmat[self.palm].reshape(3,3);palmP=s.d.xpos[self.palm]
  if self.method=='object':
   # Position correction is bounded to avoid an object-pose error becoming a
   # large instantaneous fingertip command. Pose is expressed in the palm frame.
   observed=palmR.T@(s.d.xpos[s.tool]-palmP);e=np.clip(self.tool_p[k]-observed,-.003,.003);goalP+=.5*e
   observedR=palmR.T@s.d.xmat[s.tool].reshape(3,3);rv=Rot.from_matrix(self.tool_R[k]@observedR.T).as_rotvec();rv*=min(1.,.05/max(np.linalg.norm(rv),1e-12));correction=Rot.from_rotvec(.3*rv).as_matrix();goalP=self.tool_p[k]+(goalP-self.tool_p[k])@correction.T;goalR=np.array([correction@v for v in goalR])
  for _ in range(4):
   self.kin.qpos[s.qi]=result;mujoco.mj_forward(s.m,self.kin)
   for i,bid in enumerate(self.tips):
    point=self.kin.xpos[bid]+self.kin.xmat[bid].reshape(3,3)@self.local[i];Rnow=self.kin.xmat[bid].reshape(3,3);target=palmP+palmR@goalP[i];Rgoal=palmR@goalR[i]
    Jp=np.zeros((3,s.m.nv));Jr=Jp.copy();mujoco.mj_jac(s.m,self.kin,Jp,Jr,point,bid);cols=s.vi[i*4:i*4+4];J=np.vstack([Jp[:,cols],.02*Jr[:,cols]]);err=np.r_[target-point,.02*Rot.from_matrix(Rgoal@Rnow.T).as_rotvec()]
    dq=np.linalg.solve(J.T@J+1e-7*np.eye(4),J.T@err);result[i*4:i*4+4]+=np.clip(dq,-.04,.04)
  result[12:]=ref[12:];return s.bound(result)

def calibrate(record_id):
 from screwdriver_lab.arm_sim import ArmFullSim
 s=ArmFullSim();r=json.loads((OUT/(record_id+'.json')).read_text());poses=np.load(OUT/(record_id+'.npz'))['qpos'];ts=np.array([f['t']for f in r['frames']]);d=mujoco.MjData(s.m);palm=s.m.body('palm').id;points=[];rots=[];toolp=[];toolR=[]
 commands=r['commands'];q=np.array([x['q']for x in commands]);mean=q.mean(0);_,values,V=np.linalg.svd(q-mean,full_matrices=False)
 for command in commands:
  j=min(np.searchsorted(ts,command['t']),len(poses)-1);d.qpos[:]=poses[j];d.qpos[s.qi]=command['q'];mujoco.mj_forward(s.m,d);P=d.xpos[palm];R=d.xmat[palm].reshape(3,3)
  points.append([R.T@(d.xpos[b]+d.xmat[b].reshape(3,3)@local-P)for b,local in zip(s.bids[:3],s.nominal_local)]);rots.append([R.T@d.xmat[b].reshape(3,3)for b in s.bids[:3]]);toolp.append(R.T@(d.xpos[s.tool]-P));toolR.append(R.T@d.xmat[s.tool].reshape(3,3))
 turn_origin=next(x['t']for x in commands if x['phase']=='turn');starts=[];previous=None
 for frame in r['frames']:
  mode=frame['finger_modes'][0]
  if frame['phase']=='turn' and mode=='turn' and previous!='turn':starts.append(frame['t']-turn_origin)
  previous=mode
 cycle=np.array(starts[-2:]);assert len(cycle)==2 and cycle[1]>cycle[0]
 np.savez_compressed(OUT/'abstraction-bank.npz',phase=np.array([x['phase']for x in commands]),elapsed=np.array([x['elapsed']for x in commands]),q=q,mean=mean,basis=V[:3],basis_all=V,point=points,rotation=rots,tool_p=toolp,tool_R=toolR,drive_cycle=cycle)
 info=dict(calibration_record=record_id,source='nominal simulated task rollout; not video retargeting',samples=len(q),synergy_rank=3,explained_variance=float(np.sum(values[:3]**2)/np.sum(values**2)),variance_by_rank={str(n):float(np.sum(values[:n]**2)/np.sum(values**2))for n in [3,6,10,20]},repeat_cycle_elapsed_s=cycle.tolist(),repeat_policy='After a 2 s terminal settling grace, repeat the last complete driving cycle if the turn node outlasts its nominal calibration.')
 (OUT/'abstraction-bank.json').write_text(json.dumps(info,indent=2));print(info)
if __name__=='__main__':
 import sys;calibrate(sys.argv[1])
