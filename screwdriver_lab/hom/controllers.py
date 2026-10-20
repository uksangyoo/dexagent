"""Three policy interfaces sharing a bounded joint motor controller.

HOM is a paper-inspired relative-contact-velocity controller, not an official
implementation. It explicitly selects rolling/spin/sliding/separation tasks.
"""
import numpy as np,mujoco
from scipy.optimize import lsq_linear
from screwdriver_lab.hom.contact_graph import specification,advance

def unit(v):return v/max(1e-12,np.linalg.norm(v))

class ManipulationPolicy:
    def __init__(self,h,initial,interface='hom',speed=45,amplitude=24,drive_load=.4,return_load=.025,normal_gain=.001,cycle='paired',hybrid=False,load_ramp_s=.15,release_reset_spin=False,hold_load=.2,hold_cap_load=.6):
        if interface=='joint':raise ValueError('Joint baseline requires a frozen nominal command trace; not implemented yet')
        self.h=h;self.interface=interface;self.speed=np.radians(speed);self.amplitude=np.radians(amplitude)
        self.drive_load=drive_load;self.return_load=return_load;self.hold_load=hold_load;self.hold_cap_load=hold_cap_load
        self.normal_gain=normal_gain;self.ready=0.
        self.hybrid=hybrid;self.normal_tasks=[];self.seating_requested=False;self.seat_load=0.;self.unexpected_load=0.
        self.release_reset_spin=release_reset_spin;self.load_ramp_s=load_ramp_s;self.cycle=cycle;self.strokes=np.zeros(2);self.driver=0;self.normal_targets=np.full(2,drive_load);self.omegas=np.zeros(2)
        self.qref=h.d.qpos[h.qi].copy();self.home=self.qref.copy();self.phase='acquire';self.phase_start=0.;self.stroke=0.;self.cycles=0;self.trace=[]
        self.local=[np.array(r['local'])for r in initial[:3]];self.filtered=np.zeros(3)
        self.tool=h.m.body('screwdriver').id;self.screw=h.m.body('screw').id;self.bids=[h.m.body(f'link_{i}_4').id for i in [1,2,3]]
        self.center=h.d.xpos[self.tool].copy();self.axis0=h.d.xmat[self.tool].reshape(3,3)[:,2].copy()
        self.initial_points=[np.array(r['target'])for r in initial[:3]]
        self.heights=[np.dot(p-self.center,self.axis0)for p in self.initial_points]
        self.phi=[np.arctan2(*(p-self.center)[1::-1])for p in self.initial_points]
        self.loads=np.zeros(3);self.slips=np.zeros(3);self.points=self.initial_points.copy();self.areas=np.zeros(3)
        self.graph=specification(cycle,interface)
        self.joint_template=None;self.last_command={}
        self.posture_preparation=None;self.preparation_failed=False
    def prepare_posture(self,target,duration=1.5):
        """Smooth robot motor references toward a contact-preserving IK plan."""
        self.posture_preparation=dict(start=float(self.h.d.time),duration=duration,initial=self.h.d.qpos[self.h.qi].copy(),target=np.asarray(target).copy())
        self.transition('prepare_posture')
    def posture_tick(self):
        h=self.h;d=h.d;p=self.posture_preparation;elapsed=d.time-p['start'];u=float(np.clip(elapsed/p['duration'],0,1));alpha=u*u*(3-2*u)
        qref=p['initial']+alpha*(p['target']-p['initial']);vref=(p['target']-p['initial'])*6*u*(1-u)/p['duration'];self.normal_tasks=[]
        axis=d.xmat[self.tool].reshape(3,3)[:,2];center=d.xpos[self.tool]
        for i,b in enumerate(self.bids):
            point=self.points[i];r=point-center;n=axis if i==1 else unit(r-axis*np.dot(r,axis));J=np.zeros((3,h.m.nv));K=J.copy();mujoco.mj_jac(h.m,d,J,None,point,b);mujoco.mj_jac(h.m,d,K,None,point,self.tool)
            self.normal_tasks.append((J,n,.6 if i==1 else self.drive_load,float(self.loads[i]),i,float(n@(K@d.qvel))))
        ids=[h.m.joint(f'joint_{i}_{j}').id for i in [1,2,3]for j in range(1,5)];q=d.qpos[h.m.jnt_qposadr[ids]];margin=float(np.minimum(q-h.m.jnt_range[ids,0],h.m.jnt_range[ids,1]-q).min())
        self.last_command=dict(phase='prepare_posture',fraction=alpha,minimum_joint_margin_rad=margin,contact_preserving_robot_target=True)
        if elapsed>=p['duration']and margin>.04 and min(self.loads)>.05:
            self.qref=d.qpos[h.qi].copy();self.home=self.qref.copy();self.center=center.copy();self.axis0=axis.copy();self.initial_points=[x.copy()for x in self.points];self.heights=[np.dot(x-center,axis)for x in self.points];self.phi=[np.arctan2(*(x-center)[1::-1])for x in self.points];self.ready=0.;self.posture_preparation=None;self.transition('acquire')
        elif elapsed>3.:self.preparation_failed=True
        return qref,vref
    def measure(self,raw_only=False):
        h=self.h;h.sync();d=h.d;axis=d.xmat[self.tool].reshape(3,3)[:,2];center=d.xpos[self.tool]
        self.loads[:]=0;self.areas[:]=0;self.seat_load=0.;self.unexpected_load=0.;weighted=np.zeros((3,3))
        for c in h.contacts():
            names=set(c['names'])
            if 'slot_floor' in names and any(n.startswith('workpiece_')for n in names):self.seat_load+=abs(float(c['force'][2]))
            is_robot=lambda n:n.startswith(('finger_tip_','arm_collision_','source_geometry_'))
            if any(is_robot(n)for n in names):
                allowed=names in [{'finger_tip_1','handle'},{'finger_tip_2','cap'},{'finger_tip_3','handle'}]
                if not allowed:self.unexpected_load=max(self.unexpected_load,float(np.linalg.norm(c['force'])))
            for i,target in [(0,'handle'),(1,'cap'),(2,'handle')]:
                if set(c['names'])!={f'finger_tip_{i+1}',target}:continue
                if not c['hydro']:raise RuntimeError('Fingertip contact fell back to a point law')
                radial=c['point']-center;normal=axis if i==1 else unit(radial-axis*np.dot(radial,axis))
                load=abs(np.dot(c['force'],normal));self.loads[i]+=load;weighted[i]+=load*c['point'];self.areas[i]+=c['area']
        if not raw_only:self.filtered+=.15*(self.loads-self.filtered)
        for i,b in enumerate(self.bids):
            R=d.xmat[b].reshape(3,3)
            if self.loads[i]>.002 and not raw_only:
                measured=R.T@(weighted[i]/self.loads[i]-d.xpos[b]);self.local[i]+=.25*(measured-self.local[i])
            point=d.xpos[b]+R@self.local[i]
            if not raw_only:self.points[i]=point
            if self.loads[i]>.002:point=weighted[i]/self.loads[i]
            J=np.zeros((3,h.m.nv));K=J.copy();mujoco.mj_jac(h.m,d,J,None,point,b);mujoco.mj_jac(h.m,d,K,None,point,self.tool)
            radial=point-center;normal=axis if i==1 else unit(radial-axis*np.dot(radial,axis));tangent=-np.cross(axis,normal)
            self.slips[i]=np.dot((J-K)@d.qvel,tangent)*1000
    def transition(self,name):
        self.trace.append(dict(t=self.h.d.time,from_node=self.phase,to=name,loads_N=self.loads.tolist()))
        self.phase=name;self.phase_start=self.h.d.time
    def hold_at_contact(self):
        """Start a motor-driven hold from current measured contact geometry.

        Reset controller references, not physical state. Releasing accumulated
        stroke-reference error prevents a seated fastener from being driven
        through another commanded return stroke.
        """
        from pydrake.all import HydroelasticContactRepresentation
        h=self.h;d=h.d
        query=h.scene.get_query_output_port().Eval(h.scene.GetMyContextFromRoot(h.context));ins=query.inspector()
        surfaces,_=query.ComputeContactSurfacesWithFallback(HydroelasticContactRepresentation.kTriangle)
        for surface in surfaces:
            names={ins.GetName(g).split('::')[-1]for g in [surface.id_M(),surface.id_N()]}
            for i,target in [(0,'handle'),(1,'cap'),(2,'handle')]:
                if names=={f'finger_tip_{i+1}',target}:
                    self.points[i]=surface.centroid()
                    self.local[i]=d.xmat[self.bids[i]].reshape(3,3).T@(self.points[i]-d.xpos[self.bids[i]])
        self.qref=d.qpos[h.qi].copy();self.home=self.qref.copy()
        self.center=d.xpos[self.tool].copy();self.axis0=d.xmat[self.tool].reshape(3,3)[:,2].copy()
        self.initial_points=[p.copy()for p in self.points]
        self.heights=[np.dot(p-self.center,self.axis0)for p in self.points]
        self.phi=[np.arctan2(*(p-self.center)[1::-1])for p in self.points]
        self.normal_targets=self.filtered[[0,2]].copy();self.omegas[:]=0
        self.strokes[:]=0;self.stroke=0.
        self.seating_requested=True;self.transition('hold')
    def tick(self,dt=.002):
        if self.posture_preparation is not None:return self.posture_tick()
        if self.phase=='hold':
            # Hold the captured robot posture, with pressure regulated through
            # the same bounded motors. A migrating pressure centroid is not a
            # material point and must not become a new tangential motion goal.
            h=self.h;d=h.d;axis=d.xmat[self.tool].reshape(3,3)[:,2];center=d.xpos[self.tool]
            load,_=advance(self,dt);self.normal_tasks=[]
            for i,b in enumerate(self.bids):
                point=self.points[i];r=point-center;n=axis if i==1 else unit(r-axis*np.dot(r,axis))
                J=np.zeros((3,h.m.nv));K=J.copy();mujoco.mj_jac(h.m,d,J,None,point,b);mujoco.mj_jac(h.m,d,K,None,point,self.tool)
                self.normal_tasks.append((J,n,self.hold_cap_load if i==1 else load,float(self.loads[i]),i,float(n@(K@d.qvel))))
            self.last_command=dict(phase='hold',mode='captured_robot_posture_with_pressure',stroke_deg=0.,strokes_deg=[0.,0.])
            return self.qref.copy(),np.zeros(26)
        h=self.h;d=h.d;t=d.time;axis=d.xmat[self.tool].reshape(3,3)[:,2];center=d.xpos[self.tool]
        load,omega=advance(self,dt)
        vrefs=np.zeros(26);targets=[];self.normal_tasks=[]
        for i,b in enumerate(self.bids):
            point=self.points[i];radial=point-center;normal=np.array([0.,0.,1.])if i==1 else unit(radial-axis*np.dot(radial,axis));tangent=-np.cross(axis,normal)
            J=np.zeros((3,h.m.nv));Jr=J.copy();mujoco.mj_jac(h.m,d,J,Jr,point,b);cols=h.vi[4*i:4*i+4];qcols=h.qi[4*i:4*i+4]
            expected=(self.hold_cap_load if self.phase=='hold' else .6) if i==1 else load
            stroke=self.stroke
            if self.cycle=='alternating' and i!=1:
                slot=0 if i==0 else 1;expected=self.normal_targets[slot];omega=self.omegas[slot];stroke=self.strokes[slot]
            normal_speed=np.clip(self.normal_gain*(self.filtered[i]-expected),-.003,.003)
            if self.hybrid:
                normal_speed=0.
                # Pressure damping is relative to the moving object surface.
                # World-frame damping otherwise unloads a contact whenever
                # the free tool translates during the drive/reset handoff.
                K=np.zeros((3,h.m.nv));mujoco.mj_jac(h.m,d,K,None,point,self.tool)
                # Use the current tactile load in the inner pressure loop;
                # filtering belongs to graph guards, not delayed motor feedback.
                self.normal_tasks.append((J,normal.copy(),float(expected),float(self.loads[i]),i,float(normal@(K@d.qvel))))
            if i==1:
                # Index stabilizes the free cap; force still comes only from
                # the finger motors and the resolved hydroelastic patch.
                cap_center=center+axis*.043
                error=np.r_[d.xpos[self.screw,:2]-cap_center[:2],0.]
                # Stabilize the cap's center, not the moving pressure centroid.
                # A centroid target tilts the tool as the contact patch rolls.
                desired=(6*(self.initial_points[i]-point) if self.phase=='hold' else 12*error)+normal*normal_speed
                A=J[:,cols];rhs=desired
                contact_command=dict(separation_m_s=float(normal_speed),slide_m_s=[0,0],roll_free=True,spin_free=True)
            else:
                returning=self.phase=='reset' or (self.cycle=='alternating' and omega<0)
                radius=h.m.geom('handle').size[0];height_error=self.heights[i]-np.dot(point-center,axis)
                desired=tangent*radius*omega+axis*4*height_error+normal*normal_speed
                # Center opposing normal motions on the screw axis.
                error=np.r_[center[:2]-d.xpos[self.screw,:2],0.]
                # Do not turn axis correction into unequal pinch pressure.
                # The normal coordinate belongs to pressure regulation.
                desired-=2*(error-normal*np.dot(error,normal))
                if self.phase=='hold':desired=6*(self.initial_points[i]-point);desired-=normal*np.dot(desired,normal);desired+=normal*normal_speed
                if self.interface in ['cartesian','joint']:
                    phi=self.phi[i]-stroke;target=self.center+radius*np.array([np.cos(phi),np.sin(phi),0.])+self.heights[i]*self.axis0
                    desired=6*(target-point);desired-=normal*np.dot(desired,normal);desired+=normal*normal_speed
                    A=J[:,cols];rhs=desired
                elif self.interface=='object':
                    # A rigid grasp twist commands both fingertip translation
                    # and orientation. It has no independent slip coordinate.
                    A=np.vstack([J[:,cols],.01*Jr[:,cols]])
                    rhs=np.r_[desired,-.01*axis*omega]
                else:
                    # Relative contact frames decouple 2 roll, 1 spin,
                    # 2 slide, and 1 separation coordinates. Here rolling
                    # and cap spin are released; normal separation regulates
                    # pressure, and circumferential sliding handles reset.
                    Jobject=np.zeros((3,h.m.nv));mujoco.mj_jac(h.m,d,Jobject,None,point,self.tool)
                    object_velocity=Jobject@d.qvel
                    # The return command is relative sliding, so measured
                    # object motion must be subtracted inside the constraint.
                    # During drive, only axial/normal object translation is
                    # followed; circumferential motion follows the twist goal.
                    returning=self.phase=='reset' or (self.cycle=='alternating' and omega<0)
                    if returning and self.cycle=='alternating':desired-=tangent*radius*self.speed
                    followed=np.zeros(3)if self.phase=='hold'else object_velocity if returning else object_velocity-tangent*np.dot(object_velocity,tangent)
                    basis=np.stack([tangent,axis,normal]);A=basis@J[:,cols];rhs=basis@(desired+followed)
                    Jobject_rotation=np.zeros((3,h.m.nv));mujoco.mj_jacBody(h.m,d,None,Jobject_rotation,self.tool)
                    # Match relative spin about the contact normal while
                    # releasing the two rolling coordinates. Four constraints
                    # fit the finger's four actuated DOFs; a rigid 6D grasp does not.
                    if not (self.release_reset_spin and returning):
                        A=np.vstack([A,.01*normal@Jr[:,cols]])
                        rhs=np.r_[rhs,0. if self.phase=='hold' else .01*np.dot(normal,Jobject_rotation@d.qvel)]
                contact_command=dict(separation_m_s=float(normal_speed),slide_m_s=[float((-2*radius*self.speed if self.cycle=='alternating'else radius*omega)if returning else 0),0],object_spin_goal_rad_s=float(omega if omega>0 else 0),roll_free=self.interface=='hom',spin_free=bool(self.interface=='hom' and self.release_reset_spin and returning))
            # Damping/posture are below the contact task weight. Bounded
            # least squares enforces joint speed and position limits.
            damping=.003 if i!=1 and self.interface=='hom' and self.release_reset_spin and returning else .001
            A=np.vstack([A,damping*np.eye(4)]);rhs=np.r_[rhs,damping*(.8 if self.release_reset_spin else .3)*(self.home[4*i:4*i+4]-d.qpos[qcols])]
            ids=[h.m.joint(f'joint_{i+1}_{j}').id for j in range(1,5)]
            lower=np.maximum(-1.5,(h.m.jnt_range[ids,0]+1e-4-d.qpos[qcols])/dt)
            upper=np.minimum(1.5,(h.m.jnt_range[ids,1]-1e-4-d.qpos[qcols])/dt)
            v=lsq_linear(A,rhs,bounds=(lower,upper),tol=1e-8).x
            vrefs[4*i:4*i+4]=v;targets.append(contact_command)
        # Anti-windup bounds accumulated impedance deflection. This changes
        # robot joint references only, never object state/contact parameters.
        self.qref[:12]+=vrefs[:12]*dt
        self.qref[:12]=np.clip(self.qref[:12],d.qpos[h.qi[:12]]-.08,d.qpos[h.qi[:12]]+.08)
        if self.hybrid:
            for i,task in enumerate(self.normal_tasks):
                J,n=task[:2]
                sl=slice(4*i,4*i+4);Ji=J[:,h.vi[sl]];error=self.qref[sl]-d.qpos[h.qi[sl]]
                self.qref[sl]-=Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),n)*np.dot(n,Ji@error)
        self.last_command=dict(phase=self.phase,stroke_deg=float(np.degrees(self.stroke)),strokes_deg=np.degrees(self.strokes).tolist(),contact=targets)
        return self.qref.copy(),vrefs

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
