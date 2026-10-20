"""Inspectable code-as-policy. A scripted feedback policy, not a live LLM run.

Swap Policy via --policy package.module:Class. It must return Decision values;
the common actuator/contact guards remain outside the policy.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Decision:
    phase: str
    target_deg: float
    reason: str
    finger: int = -1

class Policy:
    def __init__(self, method, goal_deg=90.):
        self.method=method;self.goal=goal_deg;self.phase='rotate'
        self.anchor=0.;self.next_turn=min(20.,goal_deg);self.phase_start=0.
        self.finger_index=0;self.repairs=0;self.last_progress=0.;self.next_check=2.
    def decide(self, observation, elapsed):
        angle=observation['angle_deg']
        if abs(self.goal-angle)<3 and observation['speed_rad_s']<.2:
            self.phase='hold'
            return Decision('hold',self.goal,'angle reached; verify 0.4 s continuous hold')
        if self.method!='contact':
            return Decision('rotate',min(self.goal,elapsed*18.),'advance the shared turn-and-hold objective')
        if self.phase=='regrasp':
            finger=[1,0,2][self.finger_index]
            if elapsed-self.phase_start>=.6:
                if observation['forces'][finger]<.05:
                    self.repairs+=1
                    if self.repairs>2:return Decision('abort',angle,'receiver contact not confirmed within repair budget')
                self.finger_index+=1;self.phase_start=elapsed
                if self.finger_index==3:
                    self.phase='rotate';self.anchor=angle;self.next_turn=min(angle+20,self.goal)
                    return Decision('repatch',angle,'reobserve material patches after contact transfer')
                finger=[1,0,2][self.finger_index]
            return Decision('regrasp',angle,'receiver-first finger reset; other fingers keep actuator targets',finger)
        if angle>=self.next_turn-2 and self.next_turn<self.goal:
            self.phase='regrasp';self.finger_index=0;self.phase_start=elapsed
            return Decision('regrasp',angle,'turn workspace exhausted; reset index, thumb, then middle',1)
        if elapsed>=self.next_check:
            progress=angle-self.last_progress;self.last_progress=angle;self.next_check=elapsed+2
            if progress<1. and elapsed>2:
                self.repairs+=1
                if self.repairs>2:return Decision('abort',angle,'no progress after two patch-replanning attempts')
                return Decision('repatch',angle,'low progress; refresh observed contact patches')
        return Decision('rotate',min(self.next_turn,angle+6.),'bounded object increment with contact feedback')
