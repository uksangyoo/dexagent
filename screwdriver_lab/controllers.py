"""Five distinct interfaces over the same hand servo and physics backend."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from screwdriver_lab.simulator import Simulator

METHODS={
 'joints':dict(name='Joint-space replay',dimension='20 joint targets',color='#70a2f7',strength='Direct control; minimal controller assumptions.',cost='The policy must coordinate every joint and manage contact timing.',failure='Open-loop replay cannot compensate for displaced or slipping contacts.'),
 'synergy':dict(name='Posture synergies',dimension='3 PCA coefficients',color='#a294e5',strength='Compact posture commands; smooth coordinated finger motion.',cost='A basis restricts the reachable posture family.',failure='Useful regrasp directions can lie outside the nominal trajectory subspace.'),
 'fingertips':dict(name='Fingertip targets',dimension='5 × 3 Cartesian targets',color='#e7bd6a',strength='Geometric intent maps through damped Jacobian IK.',cost='The policy still supplies contact trajectories and timing.',failure='Tracking fingertip positions does not guarantee that the object follows.'),
 'object':dict(name='Object-angle servo',dimension='1 rotation goal',color='#6bc6c0',strength='The controller closes the loop on actual tool angle.',cost='A dexterous controller must realize the requested object effect.',failure='Fixed contacts run out of workspace; feedback alone cannot make a regrasp.'),
 'contact':dict(name='Contact-aware policy',dimension='Roles + angle + transition',color='#fc997d',strength='An explicit policy alternates turns, finger resets and contact checks.',cost='Reliable support transfer and grounded contact state are prerequisites.',failure='Regrasp can lose support or fail to establish its receiving contact.')}

def training_bank():
    """Nominal kinematic reference only; no evaluation cases or success labels."""
    s=Simulator({},fixture=True);d=mujoco.MjData(s.model)
    q=s.initial_q.copy();qs=[]
    lo=s.model.jnt_range[:20,0]+1e-5;hi=s.model.jnt_range[:20,1]-1e-5
    for angle in np.linspace(0,100,51):
        targets=s.material_targets(angle,compression=0)
        prior=q.copy()
        def residual(x):
            d.qpos[:20]=x;mujoco.mj_forward(s.model,d)
            pts=[d.xpos[b]+d.xmat[b].reshape(3,3)@s.patches[i][0] for i,b in enumerate(s.bids)]
            return np.r_[(np.array(pts[:3])-np.array(targets[:3])).ravel()*100,.02*(x-prior)]
        q=least_squares(residual,q,bounds=(lo,hi),max_nfev=40).x;qs.append(q.copy())
    qs=np.array(qs);mean=qs.mean(0);_,sv,vt=np.linalg.svd(qs-mean,full_matrices=False)
    return dict(angles=np.linspace(0,100,51),q=qs,mean=mean,basis=vt[:3].T,variance=float(np.sum(sv[:3]**2)/np.sum(sv**2)))

class Controller:
    def __init__(self,sim,method,bank):
        self.sim=sim;self.method=method;self.bank=bank
        self.home=sim.target.copy();self.bias=0.;self.reset_finger=-1;self.reset_start=None;self.reset_t=0.
    def command(self,decision,elapsed):
        s=self.sim;angle=decision.target_deg
        if decision.phase=='abort':return s.target.copy()
        if decision.phase=='repatch':
            s.patches=s.capture_patches();self.bias=0.;return s.target.copy()
        if decision.phase=='regrasp':
            i=decision.finger
            if self.reset_finger!=i:
                self.reset_finger=i;self.reset_start=s.target.copy();self.reset_t=elapsed
            q=s.target.copy();u=np.clip((elapsed-self.reset_t)/.5,0,1);u=u*u*(3-2*u)
            q[4*i:4*i+4]=(1-u)*self.reset_start[4*i:4*i+4]+u*self.home[4*i:4*i+4]
            return s.bound(q)
        self.reset_finger=-1
        if self.method in ('joints','synergy'):
            q=np.array([np.interp(angle,self.bank['angles'],self.bank['q'][:,i]) for i in range(20)])
            if self.method=='synergy':q=self.bank['mean']+self.bank['basis']@(self.bank['basis'].T@(q-self.bank['mean']))
            return s.bound(s.target+np.clip(q-s.target,-.04,.04))
        if self.method in ('object','contact'):
            error=angle-s.observe()['angle_deg']
            self.bias=float(np.clip(self.bias+.08*error*s.dt,-10,10));angle+=np.clip(.4*error+self.bias,-15,15)
        tg=s.material_targets(angle,compression=.002)
        return s.ik(tg,gain=.35)
