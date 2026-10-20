"""CPU physics backend. Only hand actuators receive commands during rollouts."""
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from screwdriver_lab.scene import build_scene, grasp, ROOT

class Simulator:
    def __init__(self, case, fixture=False):
        self.case=case
        self.xml=build_scene(radius=case.get('radius_mm',18)/1000, friction=case.get('friction',.8),offset_mm=case.get('offset_mm',0),yaw_deg=case.get('yaw_deg',0),fixture=fixture)
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.model.opt.timestep=case.get('timestep_s',.002)
        self.data.qpos[:20]=(grasp(self.model) if case.get('radius_mm',18)!=18 or case.get('offset_mm',0)!=0 else json.loads((ROOT/'screwdriver_lab/artifacts/grasp.json').read_text()))
        self.data.qpos[:20]+=np.random.default_rng(case.get('seed',0)).normal(0,case.get('jitter_rad',0),20)
        self.data.qpos[:20]=np.clip(self.data.qpos[:20],self.model.jnt_range[:20,0],self.model.jnt_range[:20,1])
        self.target=self.data.qpos[:20].copy();self.data.ctrl[:]=self.target
        self.oid=self.model.body('screwdriver').id
        self.bids=[self.model.body(f'link_{i}_4').id for i in range(1,6)]
        self.force=np.zeros(5);self.depth=0.;self.selfdepth=0.;self.external=0.
        self.max_force=0.;self.max_depth=0.;self.max_selfdepth=0.
        self.support_force=np.zeros(5);self.forbidden_contact_N=0.;self.shaft_clearance_m=float("inf")
        self.min_shaft_clearance_m=float("inf");self.listeners=[]
        self.hand_geoms=[g for g in range(self.model.ngeom) if self.model.body(int(self.model.geom_bodyid[g])).name.startswith("link_") or self.model.body(int(self.model.geom_bodyid[g])).name=="palm"]
        self.dt=.02;self.substeps=round(self.dt/self.model.opt.timestep)
        mujoco.mj_forward(self.model,self.data)
        self.start_pos=self.data.xpos[self.oid].copy()
        self.start_rot=self.data.xmat[self.oid].reshape(3,3).copy()
        self.patches=self.capture_patches()
        self.initial_q=self.target.copy()
        self.measure_contacts()
    def capture_patches(self):
        patches=[]
        R=self.data.xmat[self.oid].reshape(3,3);p=self.data.xpos[self.oid]
        for b in self.bids:
            best=None
            for g in np.flatnonzero(self.model.geom_bodyid==b):
                line=np.zeros(6)
                dist=mujoco.mj_geomDistance(self.model,self.data,int(g),self.model.geom('handle').id,.3,line)
                if best is None or dist<best[0]:best=(dist,line.copy())
            dist,line=best
            local=self.data.xmat[b].reshape(3,3).T@(line[:3]-self.data.xpos[b])
            obj=R.T@(line[3:]-p)
            normal=np.r_[obj[:2],0.];normal/=max(np.linalg.norm(normal),1e-9)
            patches.append((local,obj,normal))
        return patches
    def ik(self, targets, active=(0,1,2), gain=.65):
        target=self.target.copy()
        for i in active:
            b=self.bids[i];local=self.patches[i][0]
            point=self.data.xpos[b]+self.data.xmat[b].reshape(3,3)@local
            J=np.zeros((3,self.model.nv));mujoco.mj_jac(self.model,self.data,J,None,point,b)
            J=J[:,i*4:i*4+4];e=targets[i]-point
            e*=min(1.,.002/max(np.linalg.norm(e),1e-9))
            dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(3),e)
            target[i*4:i*4+4]+=np.clip(gain*dq,-.04,.04)
        return self.bound(target)
    def bound(self,q):
        q=np.clip(q,self.data.qpos[:20]-.16,self.data.qpos[:20]+.16)
        return np.clip(q,self.model.jnt_range[:20,0]+1e-5,self.model.jnt_range[:20,1]-1e-5)
    def material_targets(self, angle_deg, compression=.0007, active=(0,1,2)):
        R=Rotation.from_euler('z',angle_deg,degrees=True).as_matrix()@self.start_rot
        return [self.start_pos+R@(obj-normal*compression) for _,obj,normal in self.patches]
    def measure_contacts(self):
        forces=np.zeros(5);support=np.zeros(5);forbidden=0.;depth=0.;selfdepth=0.;external=0.
        for k,c in enumerate(self.data.contact):
            bs=[int(self.model.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
            if self.oid in bs:
                b=bs[1-bs.index(self.oid)];name=self.model.body(b).name or ''
                f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,k,f)
                if name.startswith('link_'):
                    finger=int(name.split('_')[1])-1
                    load=max(0,float(f[0]));forces[finger]+=load;depth=max(depth,-c.dist)
                    object_geom=c.geom1 if bs[0]==self.oid else c.geom2
                    if self.model.geom(object_geom).name=='handle' and name.endswith('_4'):support[finger]+=load
                    else:forbidden+=load
                else:
                    external+=max(0,float(f[0]));depth=max(depth,-c.dist)
            elif all((self.model.body(b).name or '').startswith('link_') for b in bs): selfdepth=max(selfdepth,-c.dist)
        shaft=self.model.geom("shaft").id
        self.shaft_clearance_m=min(mujoco.mj_geomDistance(self.model,self.data,g,shaft,.3,None) for g in self.hand_geoms)
        self.min_shaft_clearance_m=min(self.min_shaft_clearance_m,self.shaft_clearance_m)
        self.support_force=support;self.forbidden_contact_N=forbidden
        self.force=forces;self.depth=depth;self.selfdepth=selfdepth;self.external=external
        self.max_force=max(self.max_force,float(max(forces)));self.max_depth=max(self.max_depth,depth);self.max_selfdepth=max(self.max_selfdepth,selfdepth)
    def step(self,q,stop_on_guard=True):
        self.target=self.bound(q);self.data.ctrl[:]=self.target
        events=[]
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
            self.measure_contacts()
            if not np.isfinite(self.data.qpos).all(): events.append('nonfinite')
            if self.depth>.0005: events.append('penetration')
            if self.force.max()>8.:events.append('force_limit')
            if self.selfdepth>.0015:events.append('self_collision')
            if self.shaft_clearance_m<.003:events.append('shaft_clearance')
            if self.forbidden_contact_N>.05:events.append('forbidden_contact')
            for listener in self.listeners:listener(self)
            if events and stop_on_guard: break
        return set(events)
    def observe(self):
        R=self.data.xmat[self.oid].reshape(3,3)@self.start_rot.T
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.oid,velocity,0)
        tilt=np.degrees(np.arccos(np.clip(R[2,2],-1,1)))
        return dict(t=float(self.data.time),angle_deg=float(np.degrees(np.arctan2(R[1,0],R[0,0]))),tilt_deg=float(tilt),drift_mm=float(np.linalg.norm(self.data.xpos[self.oid]-self.start_pos)*1000),speed_rad_s=float(np.linalg.norm(velocity[:3])),forces=self.force.tolist(),support_forces=self.support_force.tolist(),forbidden_contact_N=self.forbidden_contact_N,shaft_clearance_mm=self.shaft_clearance_m*1000,depth_mm=self.depth*1000,external_N=self.external,q=self.data.qpos[:20].tolist())
    def settle(self,seconds=1.):
        errors=set()
        for _ in range(round(seconds/self.dt)):
            q=self.ik(self.material_targets(0,compression=.001))
            errors.update(self.step(q))
        return errors,self.observe()
