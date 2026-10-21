"""Typed hand commands lowered to the existing 20 + 6 motor controller."""
import numpy as np
from dexagent.sim.controllers import ManipulationPolicy
from dexagent.agentic.contracts import ABSTRACTIONS


class HydroHandAdapter:
    def __init__(self, plant, initialization, abstraction):
        self.plant=plant;self.abstraction=abstraction;self.operation=ABSTRACTIONS[abstraction]['operation']
        self.controller=ManipulationPolicy(plant,initialization,abstraction,speed=60,
            drive_load=.4,return_load=.06,cycle='alternating',hybrid=True,hold_load=.2,hold_cap_load=.6)
        self.command=None;self.trace=[]
        names=[j.name()for j in plant.control_joints]
        assert len(names)==26 and all(n.startswith('joint_')for n in names[:20])
        self.layout={'hand':names[:20],'arm':names[20:]}

    def dispatch(self, action, arguments):
        if action not in {'acquire','hold',self.operation}:raise ValueError('Command does not match abstraction contract')
        c=self.controller
        if action=='hold' and self.command!='hold':c.hold_at_contact()
        if action==self.operation:
            if c.phase=='hold':raise ValueError('A held primitive cannot restart without a new explicit acquisition')
            c.speed=np.radians(arguments.get('speed_deg_s',60));c.amplitude=np.radians(arguments.get('stroke_deg',24))
        if self.command!=action:self.trace.append(dict(t=self.plant.d.time,operation=action,arguments=arguments.copy(),interface=self.abstraction))
        self.command=action

    def step(self):
        c=self.controller
        # The acquisition micro-controller can only finish its loading stage;
        # turning remains disallowed until the high-level graph issues turn.
        speed=c.speed
        if self.command=='acquire':c.speed=0.
        qref,vref=c.tick();c.speed=speed
        assert qref.shape==vref.shape==(26,)
        self.plant.motor_step(qref,vref,normal_tasks=c.normal_tasks)
        return qref,vref,self.plant.last_effort.copy()
