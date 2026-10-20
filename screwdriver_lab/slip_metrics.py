"""Accumulate physical contact evidence at every integration step (1 kHz)."""
import numpy as np
class SlipMetrics:
 def __init__(self,dt=.001,load_threshold=.01,slip_threshold=.1):
  self.dt=dt;self.load_threshold=load_threshold;self.slip_threshold=slip_threshold;self.time=0.;self.return_time=np.zeros(2);self.loaded_time=np.zeros(2);self.loaded_slide_time=np.zeros(2);self.slide_distance=np.zeros(2);self.lost=np.zeros(2);self.longest_lost=np.zeros(2);self.both=0.;self.positive_angle=0.;self.negative_angle=0.;self.last_angle=None;self.loads=[];self.first_angle=None
 def add(self,o,modes):
  if 'establish' in modes or o['t']<1.5:return
  self.time+=self.dt;loads=np.asarray(o['handle_forces_N']);slip=np.asarray(o['handle_slip_signed_mm_s']);loaded=loads>self.load_threshold
  self.loads.append(loads);self.both+=self.dt*bool(np.all(loaded));self.lost=np.where(loaded,0,self.lost+self.dt);self.longest_lost=np.maximum(self.lost,self.longest_lost)
  for i in range(2):
   if modes[i]!='slide_return':continue
   self.return_time[i]+=self.dt;self.loaded_time[i]+=self.dt*loaded[i];self.loaded_slide_time[i]+=self.dt*(loaded[i] and slip[i]<-self.slip_threshold);self.slide_distance[i]+=self.dt*max(0,-slip[i])*loaded[i]
  angle=o['screw_angle_deg']
  if self.first_angle is None:self.first_angle=angle
  if self.last_angle is not None:
   self.positive_angle+=max(0,angle-self.last_angle);self.negative_angle+=max(0,self.last_angle-angle)
  self.last_angle=angle
 def report(self):
  ratio=lambda x,y:np.divide(x,y,out=np.zeros_like(x),where=y>0).tolist()
  return dict(sample_rate_Hz=round(1/self.dt),load_threshold_N=self.load_threshold,slip_threshold_mm_s=self.slip_threshold,drive_duration_s=self.time,return_duration_s=self.return_time.tolist(),return_loaded_fraction=ratio(self.loaded_time,self.return_time),return_loaded_sliding_fraction=ratio(self.loaded_slide_time,self.return_time),loaded_return_sliding_distance_mm=self.slide_distance.tolist(),both_pads_loaded_fraction=self.both/max(self.time,1e-9),longest_contact_dropout_s=self.longest_lost.tolist(),positive_rotation_deg=self.positive_angle,backward_rotation_deg=self.negative_angle,net_rotation_deg=(self.last_angle-self.first_angle)if self.last_angle is not None else 0,median_pad_load_N=np.median(self.loads,axis=0).tolist()if self.loads else [0,0])
