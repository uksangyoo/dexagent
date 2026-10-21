"""Video-derived bimanual policy graph. Actuation writes joint setpoints only.

Free rack poses, velocities, masses, gravity and external forces are never
written by this controller. A failed lift prevents the flip/transfer stages.
"""
import json,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robot import Kinematics, hand_rotation, HAND

HERE=Path(__file__).resolve().parent


def grip_coordinates(kin,side):
    m,d=kin.model,kin.data
    palm=m.body(side+"_palm").id
    out=np.zeros(20)
    for i in range(1,6):
        idx=np.array([m.joint(f"{side}_joint_{i}_{j}").qposadr[0] for j in range(1,5)])
        joints=[m.joint(f"{side}_joint_{i}_{j}").id for j in range(1,5)]
        bounds=m.jnt_range[joints]
        bid=m.body(f"{side}_link_{i}_4").id
        s=1 if side=="right" else -1
        y={1:s*.028,2:s*.028,3:s*.005,4:-s*.018,5:-s*.0425}[i]
        target=np.array([.045 if i==1 else .073,y,.113 if i==1 else .135 if i<5 else .12])
        def error(q):
            d.qpos[idx]=q;mujoco.mj_kinematics(m,d)
            point=d.xpos[bid]+d.xmat[bid].reshape(3,3)@np.array([.026,0,0])
            return np.r_[d.xmat[palm].reshape(3,3).T@(point-d.xpos[palm])-target,.001*q]
        fit=least_squares(error,np.clip(np.array([0,.5,.8,.5]),bounds[:,0]+1e-6,bounds[:,1]-1e-6),bounds=(bounds[:,0],bounds[:,1]),max_nfev=100)
        out[(i-1)*4:i*4]=fit.x
    return out


class Controller:
    def __init__(self,sim):
        self.sim=sim;self.kin=Kinematics()
        self.arms=sim.scene.handles.arms
        self.target=sim.pipeline.control.joint_target_q.numpy().copy()
        self.q={s:np.array(h.home_q) for s,h in self.arms.items()}
        self.integral={s:np.zeros(6) for s in self.arms}
        self.grips={s:grip_coordinates(self.kin,s) for s in self.arms}
        self.trace=[];self.phase="settle";self.start=time.time()
        self.spacing={s:.305 for s in self.arms}

    def step(self,positions=None,rotations=None,grip=0):
        actual=self.sim.state.joint_q.numpy()
        for s,h in self.arms.items():
            if positions is not None:
                self.q[s]=self.kin.solve(s,positions[s],rotations[s],self.q[s])
            # Joint-space integral action offsets steady gravity sag through
            # the bounded motors; never through body-level gravity cancellation.
            self.integral[s]=np.clip(self.integral[s]+.05*(self.q[s]-actual[list(h.arm_coords)]),-.12,.12)
            self.target[list(h.arm_coords)]=self.q[s]+self.integral[s]
            self.target[list(h.gripper_coords)]=self.grips[s]*grip
        self.sim.pipeline.control.joint_target_q.assign(self.target)
        self.sim.run(1).sync()
        sim=self.sim
        sim.contacts.update(sim.step_count)
        tray=sim.scene.handles.objects["tray_10"]
        pen=sim.contacts.penetrations()
        self.trace.append({"t":sim.time,"phase":self.phase,"tray":sim["tray_10"].pose.tolist(),
                           "hand_force_N":{s:sim.contacts.pairwise_force(h.shapes,tray.shapes).tolist() for s,h in self.arms.items()},
                           "max_penetration_mm":float(pen[0,2]*1000) if len(pen) else 0,
                           "joint_q":sim.state.joint_q.numpy().tolist(),"joint_targets":self.target.tolist()})

    def move(self,name,centers,height=.15,grips=(0,0),angles=(0,0),seconds=1,force_control=False):
        self.phase=name
        for t in np.linspace(0,1,max(2,round(seconds*50))):
            u=t*t*(3-2*t)
            center=(1-u)*np.array(centers[0])+u*np.array(centers[1])
            R=Rotation.from_rotvec([0,(1-u)*angles[0]+u*angles[1],0]).as_matrix()
            if force_control and self.trace:
                for s,sign in (("left",1),("right",-1)):
                    force=np.array(self.trace[-1]["hand_force_N"][s])
                    normal=max(0.,float(force@(R@np.array([0,sign,0]))))
                    self.spacing[s]=float(np.clip(self.spacing[s]+np.clip(normal-14,-8,8)*.00005,.275,.34))
            positions={s:center+R@np.array([-.005,sign*self.spacing[s],height]) for s,sign in (("left",1),("right",-1))}
            rotations={s:R@hand_rotation(s) for s in positions}
            self.step(positions,rotations,(1-u)*grips[0]+u*grips[1]);yield


def first_cycle(sim,session=None,stop_after="complete"):
    c=Controller(sim)
    # UI sliders must not overwrite independent 20-DOF hand targets.
    old={}
    if session is not None:
        for name,panel in session.arms.items():old[name]=panel.suspended;panel.suspended=True
    outcome="incomplete"
    try:
        for _ in range(50):c.step();yield
        p=np.array(sim["tray_10"].position)
        hover=p.copy();hover[2]=1.0
        yield from c.move("approach",(hover,p),grips=(0,0),seconds=1.5)
        yield from c.move("close",(p,p),grips=(0,1),seconds=1)
        yield from c.move("regulate_grip",(p,p),grips=(1,1),seconds=1.5,force_control=True)
        if stop_after=="grasp":outcome="grasp_probe";return
        high=p+np.array([0,0,.32])
        yield from c.move("lift",(p,high),grips=(1,1),seconds=1.5,force_control=True)
        lift=float(sim["tray_10"].position[2]-p[2])
        if lift<.15:outcome="failed_lift";return
        if stop_after=="lift":outcome="lift_probe";return
        yield from c.move("flip",(high,high),grips=(1,1),angles=(0,np.pi),seconds=2,force_control=True)
        up=Rotation.from_quat(sim["tray_10"].pose[3:]).as_matrix()[2,2]
        if up>-.85:outcome="failed_flip";return
        feed=np.array([1.02,0,1.05])
        yield from c.move("transfer",(high,feed),grips=(1,1),angles=(np.pi,np.pi),seconds=1.5,force_control=True)
        seated=feed.copy();seated[2]=.945
        yield from c.move("lower",(feed,seated),grips=(1,1),angles=(np.pi,np.pi),seconds=.7)
        yield from c.move("release",(seated,seated),grips=(1,0),angles=(np.pi,np.pi),seconds=.7)
        outcome="released_unverified"
    except Exception as exc:
        outcome=f"error:{type(exc).__name__}:{exc}"
        raise
    finally:
        out=HERE/"results";out.mkdir(exist_ok=True)
        report={"outcome":outcome,"simulated_s":sim.time,"wall_s":time.time()-c.start,"trace":c.trace}
        file=out/f"cycle-{time.time_ns()}.json";file.write_text(json.dumps(report))
        (out/"latest.json").write_text(json.dumps(report))
        print(f"CRATE_RESULT {outcome} {file}",flush=True)
        if session is not None:
            # A scalar gripper slider cannot preserve a 20-DOF pose; leave it
            # suspended until the user explicitly resumes manual control.
            for name in old:session.arms[name].suspended=True


def grasp_probe(sim,session=None):yield from first_cycle(sim,session,stop_after="grasp")
def lift_probe(sim,session=None):yield from first_cycle(sim,session,stop_after="lift")
