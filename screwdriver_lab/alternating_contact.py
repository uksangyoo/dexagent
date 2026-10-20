"""Overlapping drive/return strokes with regulated positive side-pad loads."""
import numpy as np
from screwdriver_lab.pivot_spin_controller import CapturedSlipController

class AlternatingContactController(CapturedSlipController):
    def __init__(self,s,**kwargs):
        super().__init__(s,**kwargs)
        self.angle=np.zeros(2);self.driver=0;self.half_cycles=0
        self.return_load=getattr(s,'return_normal_N',.25);self.drive_load=getattr(s,'drive_normal_N',.4)
        self.amplitude=24.;self.speed=15.;self.interface=getattr(s,'control_interface','wrench')
        self.force_targets=np.asarray(s.handleloads).copy()
        self.graph.schema={
            'version':3,'meta':{'name':'overlapping_loaded_slip'},
            'edges':[['START','acquire']],
            'nodes':{n:{'type':'code','ref':'AlternatingContactController.tick'}for n in ['acquire','thumb_drive','middle_drive','axis_recover','seat']},
            'conditional_edges':{
                'axis_recover':{'router_field':'exit','mapping':{'centered_thumb':'thumb_drive','centered_middle':'middle_drive'}},
                'acquire':{'router_field':'exit','mapping':{'loaded':'thumb_drive'}},
                'thumb_drive':{'router_field':'exit','mapping':{'stroke_complete':'middle_drive','axis_misaligned':'axis_recover','seated':'seat'}},
                'middle_drive':{'router_field':'exit','mapping':{'stroke_complete':'thumb_drive','axis_misaligned':'axis_recover','seated':'seat'}}}}
    def torques(self):
        if self.interface!='pose':super().torques()
    def transition(self,phase,o):
        if phase=='seat':
            self.trace.append(dict(t=o['t'],from_node=self.phase,to='seat',screw_angle_deg=o['screw_angle_deg']))
            self.phase='seat';self.start=o['t'];self.seat_q=self.s.d.qpos[self.s.qi].copy()
            if hasattr(self,'force_integral'):self.force_integral[:]=0
        else:super().transition(phase,o)
    def tick(self,o):
        s=self.s;t=o['t'];old=self.angle.copy()
        if self.phase=='seat':
            s.modes=['seat','seat'];self.command_speed=np.zeros(2)
            if getattr(s,'effort_seating',False):
                goal=(self.drive_load+self.return_load)/2
                self.force_targets+=np.clip(goal-self.force_targets,-.01,.01)
            return self.seat_q.copy(),dict(mode='seat')
        if self.phase=='acquire':
            goal=np.full(2,self.drive_load)
            if t-self.start>.5 and min(s.handleloads)>.15 and max(abs(np.asarray(s.handleloads)-goal))<.15 and s.index_cap_load>.1:
                self.transition('alternate',o)
        elif self.phase=='axis_recover':
            # Equalize opposing grip forces before the cap reaches its lateral
            # friction/actuator limit. Recovery is entirely robot actuation.
            goal=np.full(2,(self.drive_load+self.return_load)/2)
            if t-self.start>.3 and o['tilt_deg']<.6 and s.index_cap_load>.5 and min(s.handleloads)>.2:
                self.transition('alternate',o)
        else:
            if getattr(s,'axis_recovery',False) and (o['tilt_deg']>2. or s.index_cap_load<.1):
                self.transition('axis_recover',o)
                goal=np.full(2,(self.drive_load+self.return_load)/2)
            else:
                goal=np.full(2,self.return_load);goal[self.driver]=self.drive_load
            # Pause a stroke while a missing pad is being reacquired. Normal
            # force commands remain active; no clearance/regrasp is commanded.
            if self.phase!='axis_recover' and min(s.handleloads)>.03 and s.handleloads[self.driver]>self.drive_load-.1 and s.handleloads[1-self.driver]<self.return_load+.1 and s.index_cap_load>.1:
                self.angle[self.driver]=min(self.amplitude,self.angle[self.driver]+self.speed*.02)
                other=1-self.driver;self.angle[other]=max(0.,self.angle[other]-self.speed*.02)
            reached=[]
            for slot,point in enumerate(s.pad_points):
                delta=point-self.drive_center;observed=np.arctan2(delta[1],delta[0])
                wanted=self.phi[slot]-np.radians(self.angle[slot])
                reached.append(s.handleloads[slot]>.03 and abs(np.arctan2(np.sin(observed-wanted),np.cos(observed-wanted)))<np.radians(4))
            if self.phase!='axis_recover' and self.angle[self.driver]>=self.amplitude and self.angle[1-self.driver]<=0 and all(reached):
                self.trace.append(dict(t=t,from_node=f'driver_{self.driver}',to=f'driver_{1-self.driver}',screw_angle_deg=o['screw_angle_deg'],handle_loads_N=list(s.handleloads)))
                self.driver=1-self.driver;self.half_cycles+=1;self.cycles=self.half_cycles//2
        ramp=.02 if self.phase=='axis_recover'else .005
        self.force_targets+=np.clip(goal-self.force_targets,-ramp,ramp)
        self.command_speed=(self.angle-old)/.02
        self.cap=float(np.clip(self.cap+.001*(s.axial-.6)*.02,-.007,.004))
        q=s.contact_ik(0,[0,0],cap=self.cap,active=(1,));q[12:]=s.home[12:]
        if self.interface=='pose':q,_=self.pose_command(o,update_cap=False)
        s.modes=['drive'if v>1e-6 else'slide_return'if v< -1e-6 else'hold'for v in self.command_speed]
        if self.phase in ['acquire','axis_recover']:s.modes=[self.phase,self.phase]
        s.finger_roll=self.angle.tolist()
        return q,dict(driver=self.driver,angle_deg=self.angle.tolist(),force_target_N=self.force_targets.tolist())
