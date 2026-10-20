"""Task experiments with physical gravity, bounded motors and mating threads.

No force is applied to the tool or fastener by the task controller. CPU MuJoCo
provides the current task evidence. Newton/MuJoCo-Warp is an additional
GPU 1 backend under validation. Motor commands are held for 2 ms (500 Hz).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import mujoco
from screwdriver_lab.thread_geometry import replace_thread
from screwdriver_lab.arm_scene import ArmIK
from screwdriver_lab import soft_finger

OUT=Path(__file__).parent/'artifacts/physical_task'

def configure(s, threads=True, torque=.11, friction=.6,cycle='paired',interface='wrench'):
    root=E.fromstring(s.xml)
    root.find('option').set('impratio','100')
    flag=root.find('option/flag')
    if flag is not None:flag.set('nativeccd','disable')
    for body in root.iter('body'):body.set('gravcomp','0')
    for actuator in list(root.find('actuator'))[:20]:actuator.set('forcerange',f'{-torque} {torque}')
    for i,actuator in enumerate(list(root.find('actuator'))[20:]):
        actuator.set('kp','1200'if i<3 else'400');actuator.set('kv','40'if i<3 else'5')
    # Keep the imported silicone pad geometry. Baseline is an explicit
    # compliant contact model, with no extra point torsional/rolling moment.
    for geom in root.iter('geom'):
        if geom.get('name','').startswith('finger_tip_'):
            geom.set('condim','4');geom.set('friction',f'{friction} {soft_finger.torsion_length(1.,friction)} 0')
    if threads:replace_thread(root,24)
    oldm,oldd=s.m,s.d;s.xml=E.tostring(root,encoding='unicode')
    s.m=mujoco.MjModel.from_xml_string(s.xml);s.d=mujoco.MjData(s.m)
    for j in range(oldm.njnt):
        name=oldm.joint(j).name;other=s.m.joint(name).id
        n=7 if oldm.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
        a,b=oldm.jnt_qposadr[j],s.m.jnt_qposadr[other];s.d.qpos[b:b+n]=oldd.qpos[a:a+n]
    s.d.ctrl[:]=oldd.ctrl
    s.jids=np.array([s.m.joint(f'joint_{i}_{j}').id for i in range(1,6)for j in range(1,5)])
    s.qi=s.m.jnt_qposadr[s.jids];s.vi=s.m.jnt_dofadr[s.jids]
    s.tipids=[s.m.geom(f'finger_tip_{i}').id for i in range(1,6)]
    s.bids=[s.m.body(f'link_{i}_4').id for i in range(1,6)]
    for attr,name in [('tool','screwdriver'),('cap','swivel_cap'),('screw','screw')]:setattr(s,attr,s.m.body(name).id)
    s.depthq=s.m.joint('screw_depth').qposadr[0];s.angleq=s.m.joint('screw_angle').qposadr[0]
    s.anglev=s.m.joint('screw_angle').dofadr[0];s.toq=s.m.joint('tool_free').qposadr[0]
    s.arm=ArmIK(s.m);s.arm.last=s.d.qpos[s.arm.qi].copy();mujoco.mj_forward(s.m,s.d)
    s.transport_load=1.3
    s.pivot_load=.65;s.pivot_settle_s=.12;s.pivot_capture_tilt_deg=35
    s.align_lateral_gate_mm=.9
    s.align_timeout_s=24
    s.align_bit_tip=True
    s.control_interface=interface
    s.cap_drive_load_N=2.;s.cap_damping_compensation=False;s.centering_gain=0.
    if cycle=='alternating':
        from screwdriver_lab.alternating_contact import AlternatingContactController
        s.controller_factory=AlternatingContactController
    original_observe=s.observe
    def physical_observe():
        o=original_observe()
        o['engaged']=bool(o['engaged'] and -.3<o['bit_height_mm']<.4 and o['bit_lateral_mm']<.3)
        o['thread_pitch_error_mm']=float((s.d.qpos[s.depthq]-.001*s.d.qpos[s.angleq]/(2*np.pi))*1000)
        o['finger_motor_torque_Nm']=s.d.actuator_force[:20].tolist()
        return o
    s.observe=physical_observe
    s.thread_parameters=dict(friction=.15,pitch_m=.001,radius_m=.003,half_flank_angle_deg=30)if threads else None
    s.slip_options=dict(amplitude=12,speed=8,drive_load=.6,slide_load=.03,
                        force_p=.5,force_i=.5,normal_d=2,orientation_gain=.015,posture_gain=.01)

class MotorControl:
    """Finite motor effort, including gravity/bias compensation in the budget."""
    def __init__(self,s,torque=.11):
        s.direct_task_effort=True
        self.s=s;self.limit=np.r_[np.full(20,torque),[150]*3,[28]*3]
        self.qi=np.r_[s.qi,s.arm.qi];self.vi=np.r_[s.vi,s.arm.vi]
        self.tick=0;self.peak=np.zeros(26);self.saturated=np.zeros(26);self.samples=0
        self.torque=np.zeros(26)
        self.cap_active=False;self.cap_local=s.local[1].copy();self.cap_missing=0
        self.transport_anchors=None;self.transport_local=None
        self.drive_started=None
    def command(self):
        s=self.s
        if np.any(s.d.xfrc_applied) or np.any(s.d.qfrc_applied):raise RuntimeError("Task requires robot-actuator-only control")
        if self.tick%2==0:
            pd=s.m.actuator_gainprm[:,0]*(s.d.ctrl-s.d.qpos[self.qi])+s.m.actuator_biasprm[:,2]*s.d.qvel[self.vi]
            if s.stage=="turn" or (s.stage=="seat" and getattr(s,"effort_seating",False)):
                for finger,tau in getattr(s,"task_effort",{}).items():pd[4*finger:4*finger+4]=tau
            # qfrc_bias contains gravity/Coriolis; compensation is delivered
            # through the same motors and clipped with the task torque.
            compensation=s.d.qfrc_bias[self.vi].copy()
            blend=0.
            if s.stage in ['turn','seat']:
                if self.drive_started is None:self.drive_started=float(s.d.time)
                blend=float(np.clip((s.d.time-self.drive_started)/1.,0,1))
                # Cancel modeled joint damping only for the side fingers'
                # wrench task; the compensation remains inside motor limits.
                for sl in [slice(0,4),slice(8,12)]:compensation[sl]-=getattr(s,"side_damping_compensation",1.)*s.d.qfrc_passive[self.vi[sl]]
                if getattr(s,'cap_damping_compensation',True):compensation[4:8]-=blend*s.d.qfrc_passive[self.vi[4:8]]
            requested=pd+compensation
            if getattr(s,'transport_interface','position')=='wrench' and s.stage in ['lift','reorient','cap_support','align']:
                palm=s.m.body('palm').id;Rp=s.d.xmat[palm].reshape(3,3)
                if self.transport_anchors is None:
                    self.transport_anchors=[Rp.T@(p-s.d.xpos[palm])for p in s.pad_points]
                    self.transport_local=[s.d.xmat[s.bids[i]].reshape(3,3).T@(s.pad_points[k]-s.d.xpos[s.bids[i]])for k,i in enumerate([0,2])]
                for slot,finger in enumerate([0,2]):
                    body=s.bids[finger];R=s.d.xmat[body].reshape(3,3);sl=slice(4*finger,4*finger+4);cols=self.vi[sl]
                    if s.handleloads[slot]>.01:
                        self.transport_local[slot]+=.2*(R.T@(s.pad_points[slot]-s.d.xpos[body])-self.transport_local[slot])
                    point=s.d.xpos[body]+R@self.transport_local[slot]
                    target=s.d.xpos[palm]+Rp@self.transport_anchors[slot]
                    axis=s.d.xmat[s.tool].reshape(3,3)[:,2];normal=point-s.d.xpos[s.tool]
                    normal-=axis*np.dot(normal,axis);normal/=max(1e-9,np.linalg.norm(normal))
                    J=np.zeros((3,s.m.nv));K=J.copy()
                    mujoco.mj_jac(s.m,s.d,J,None,point,body);mujoco.mj_jac(s.m,s.d,K,None,target,palm)
                    relative=(J-K)@s.d.qvel;e=target-point
                    et=e-normal*np.dot(e,normal);vt=relative-normal*np.dot(relative,normal)
                    load=s.pivot_load if s.stage=='reorient'else s.transport_load
                    fn=np.clip(load+.5*(load-s.handleloads[slot])+2*np.dot(relative,normal),.1,2.)
                    force=-normal*fn+100*et-vt
                    Ji=J[:,cols];null=np.eye(4)-Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),Ji)
                    posture=.2*(s.d.ctrl[sl]-s.d.qpos[self.qi[sl]])-.01*s.d.qvel[cols]
                    requested[sl]=Ji.T@force+null@posture+s.d.qfrc_bias[cols]-s.d.qfrc_passive[cols]
            self.cap_missing=self.cap_missing+1 if s.index_cap_load<.01 else 0
            if self.cap_missing>50:self.cap_active=False
            if s.stage in ['cap_support','align','turn','seat'] and (s.index_cap_load>.05 or self.cap_active):
                self.cap_active=True
                body=s.bids[1];R=s.d.xmat[body].reshape(3,3)
                points=[c.pos.copy()for c in s.d.contact if
                    {s.m.geom(c.geom1).name,s.m.geom(c.geom2).name}=={'cap','finger_tip_2'}]
                points=getattr(s,'cap_contact_points',points)
                if points:self.cap_local+=.2*(R.T@(np.mean(points,axis=0)-s.d.xpos[body])-self.cap_local)
                point=s.d.xpos[body]+R@self.cap_local;axis=s.d.xmat[s.tool].reshape(3,3)[:,2]
                target=s.d.xpos[s.tool]+axis*.043
                if s.stage in ['turn','seat']:
                    # During driving, stabilize the cap above the engaged
                    # screw using the index fingertip's bounded lateral force.
                    target[:2]=(1-blend)*target[:2]+blend*s.d.xpos[s.screw,:2];axis=np.array([0.,0.,1.])
                J=np.zeros((3,s.m.nv));K=J.copy();Jr=J.copy()
                mujoco.mj_jac(s.m,s.d,J,Jr,point,body);mujoco.mj_jac(s.m,s.d,K,None,target,s.tool)
                relative=(J-K)@s.d.qvel;error=target-point
                tangential=error-axis*np.dot(error,axis);vt=relative-axis*np.dot(relative,axis)
                load=.3+(getattr(s,"cap_drive_load_N",.6)-.3)*blend
                driving=s.stage in ['turn','seat']
                if driving:
                    # The lateral target is anchored over the screw, so its
                    # damping must use the screw frame. Subtracting the tool's
                    # velocity here would leave whole-tool rocking undamped.
                    headJ=np.zeros((3,s.m.nv));mujoco.mj_jacBody(s.m,s.d,headJ,None,s.screw)
                    anchored=(J-headJ)@s.d.qvel;anchored-=axis*np.dot(anchored,axis)
                    vt=(1-blend)*vt+blend*anchored
                if driving and getattr(s,"cap_axis_control",True):
                    cap_center=s.d.xpos[s.tool]+s.d.xmat[s.tool].reshape(3,3)[:,2]*.043
                    axis_error=s.d.xpos[s.screw]-cap_center;axis_error[2]=0.
                    tangential=(1-blend)*tangential+blend*axis_error
                lateral=(40+(getattr(s,"cap_axis_stiffness",100.)-40)*blend)*tangential-(.4+(getattr(s,"cap_axis_damping",3.)-.4)*blend)*vt
                lateral*=min(1.,(getattr(s,'cap_lateral_limit_N',.3) if driving else .15)/max(1e-9,np.linalg.norm(lateral)))
                force=lateral-axis*np.clip(load+2*np.dot(relative,axis),.05,max(.8,1.3*load))
                Ji=J[:,self.vi[4:8]]
                null=np.eye(4)-Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),Ji)
                posture=.1*(s.d.ctrl[4:8]-s.d.qpos[self.qi[4:8]])-.01*s.d.qvel[self.vi[4:8]]
                if driving and getattr(s,'cap_orientation_gain',0)>0:
                    normal=R@s.pad_normals_local[1]
                    posture=Jr[:,self.vi[4:8]].T@(s.cap_orientation_gain*np.cross(normal,-axis))-.01*s.d.qvel[self.vi[4:8]]
                requested[4:8]=Ji.T@force+null@posture+compensation[4:8]
            # Continuous rating and a conservative DC torque-speed envelope.
            speed=np.abs(s.d.qvel[self.vi[:20]])
            available=self.limit.copy();available[:20]=np.minimum(available[:20],.65*np.maximum(0,1-speed/(85*2*np.pi/60)))
            self.torque=np.clip(requested,-available,available)
            self.peak=np.maximum(self.peak,abs(self.torque));self.saturated+=(abs(requested)>available)
            self.samples+=1
        self.tick+=1
        return self.torque
    def report(self):
        return dict(command_hz=500,peak_motor_torque_Nm=self.peak.tolist(),
                    saturation_fraction=(self.saturated/max(1,self.samples)).tolist(),
                    gravity_compensation='bounded actuator torque; body gravcomp=0',
                    tool_or_screw_control_force=False)

class CpuDevelopmentBackend:
    """CPU reference backend; never uses either GPU."""
    def __init__(self,s,torque=.11):
        self.s=s;self.motor=MotorControl(s,torque);self.stats={'backend':'MuJoCo CPU development diagnostic'}
    def set_hand_impedance(self,*args):pass
    def step(self):
        s=self.s;soft_finger.update(s);tau=self.motor.command()
        gain=s.m.actuator_gainprm.copy();bias=s.m.actuator_biasprm.copy();target=s.d.ctrl.copy()
        # Convert the robot actuators to direct effort for this step. The
        # controller-facing PD gains remain available between steps.
        s.m.actuator_gainprm[:,0]=1;s.m.actuator_biasprm[:]=0;s.d.ctrl[:]=tau
        for _ in range(4):s.m.opt.timestep=.00025;mujoco.mj_step(s.m,s.d)
        # Observe the forces produced by the held motor command. Restoring
        # the controller's PD gains before forward() would recompute a
        # different contact solve and feed fictitious forces to the policy.
        mujoco.mj_forward(s.m,s.d)
        s.m.actuator_gainprm[:]=gain;s.m.actuator_biasprm[:]=bias;s.d.ctrl[:]=target
    def measure(self):self.s.measure()
    def step_batch(self):
        for _ in range(20):self.step();self.measure()
        return [self.s.peak,self.s.peakpen,0,0,0]

def main():
    from screwdriver_lab.inhand_policy import run
    p=argparse.ArgumentParser();p.add_argument('--name',default='physical-v1')
    p.add_argument('--timeout',type=float,default=35);p.add_argument('--diagnostic-cpu',action='store_true')
    p.add_argument('--ideal-thread-diagnostic',action='store_true');p.add_argument('--torque',type=float,default=.11)
    p.add_argument('--friction',type=float,default=.6);p.add_argument('--interface',default='wrench',choices=['wrench','velocity','pose'])
    p.add_argument('--gpu-contact',choices=['native','newton'],default='newton');p.add_argument('--checkpoint-only',action='store_true');p.add_argument('--initial-y-mm',type=float,default=0);p.add_argument('--cycle',choices=['paired','alternating'],default='paired');p.add_argument('--gpu-collision-hz',type=int,choices=[1000,4000],default=4000);a=p.parse_args();OUT.mkdir(exist_ok=True)
    source_dir=Path(__file__).parent;sources={}
    for name in ['physical_task.py','alternating_contact.py','soft_finger.py','newton_task_backend.py','thread_geometry.py','inhand_policy.py','controlled_slip.py','slip_graph.py','pivot_spin_controller.py','pivot_pose_controller.py']:
        data=(source_dir/name).read_bytes();digest=hashlib.sha256(data).hexdigest();sources[name]=digest
        snapshots=OUT/'source_snapshots';snapshots.mkdir(exist_ok=True)
        (snapshots/(digest+'_'+name)).write_bytes(data)
    backends=[]
    def backend(s):
        s.gpu_contact_source=a.gpu_contact;s.gpu_collision_hz=a.gpu_collision_hz
        if a.diagnostic_cpu:b=CpuDevelopmentBackend(s,a.torque)
        else:
            from screwdriver_lab.newton_task_backend import NewtonTaskBackend
            b=NewtonTaskBackend(s,a.torque)
        s.checkpoint_only=a.checkpoint_only
        backends.append(b)
        if a.diagnostic_cpu:
            from screwdriver_lab.drive_checkpoint import save_checkpoint
            s.save_drive_checkpoint=lambda cap:save_checkpoint(s,b,a.name,cap)
            s.checkpoint_only=a.checkpoint_only
        return b
    result,q,xml=run(a.timeout,transport_torsion=0,interface=a.interface,initial_y_mm=a.initial_y_mm,
        physics_setup=lambda s:configure(s,not a.ideal_thread_diagnostic,a.torque,a.friction,a.cycle,a.interface),backend_factory=backend)
    result.update(id=a.name,physical_source_sha256=sources,physics=dict(explicit_threads=not a.ideal_thread_diagnostic,
        finger_continuous_torque_Nm=a.torque,friction=a.friction,contact=soft_finger.description(),**backends[0].stats),motor=backends[0].motor.report())
    result['method']={'pose':'Contact-pose admittance','velocity':'Tangential-velocity control','wrench':'Hybrid contact-wrench control'}[a.interface]
    result['parameters'].update(cycle=a.cycle,normal_drive_N=.4 if a.cycle=='alternating'else .6,normal_return_N=.25 if a.cycle=='alternating'else .03,cap_drive_N=2.,pivot_load_N=.65,pivot_capture_tilt_deg=35,
        stroke_deg=24 if a.cycle=='alternating'else 12,stroke_speed_deg_s=15 if a.cycle=='alternating'else 8,
        thread_torque_Nm=None if not a.ideal_thread_diagnostic else .003,
        torsional_friction_m='load dependent soft-finger law',impratio=100,normal_contact_solref=[.004,1.])
    result['physics']['mujoco_version']=mujoco.__version__
    result['physics']['gravity_compensation_body_count']=int(np.count_nonzero(backends[0].s.m.body_gravcomp))
    result['physics']['equality_count']=int(backends[0].s.m.neq)
    (OUT/(a.name+'.json')).write_text(json.dumps(result));np.savez_compressed(OUT/(a.name+'.npz'),qpos=q)
    (OUT/(a.name+'.xml')).write_text(xml);print('RESULT',result['outcome'],result['final'],flush=True)

if __name__=='__main__':main()
