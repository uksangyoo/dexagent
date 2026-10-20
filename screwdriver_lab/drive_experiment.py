"""Controlled drive-subgraph ablation with an attached UR5e and paired reset.

This starts at an aligned grasp and is explicitly distinct from full pickup.
"""
import argparse,json
import numpy as np,mujoco
from screwdriver_lab.arm_sim import ArmFullSim
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.screw_scene import CENTER,R,ROOT

def run(mode='regrasp',hydro=False,timeout=15):
 if mode not in ['regrasp','slide','hold']:raise ValueError('mode must be regrasp, slide, or hold')
 s=ArmFullSim({'force_weighted_slip':True});g=json.loads((ROOT/'screwdriver_lab/artifacts/screw_in/precision_grasp.json').read_text());s.d.qpos[s.qi]=g['q'];s.d.qpos[s.arm.qi]=s.arm.solve(s.p0,R,exact=True);s.d.qpos[s.toq:s.toq+7]=np.r_[CENTER,[1.,0,0,0]];s.d.ctrl[:20]=g['q'];s.d.ctrl[20:]=s.d.qpos[s.arm.qi];mujoco.mj_forward(s.m,s.d);s.capture();s.peak=0;s.peakpen=0;s.measure();s.set_hand_impedance(2.,.08);s.stage='drive_benchmark';s.slip=[0.,0.];s.modes=['press','press'];s.feed=0
 if hydro:
  from screwdriver_lab.hydro_backend import HydroBackend
  s.hydro=HydroBackend(s)
 frames=[];poses=[];actions=[];phase='press';start=0.;slide_start=1.5;slide_rad=np.zeros(2);slide_bias=np.zeros((2,2));angle=0.;release=0.;phase_angle=0.;turn_angle=0.;cycle=0;held=0.;nextframe=0.;outcome='timeout'
 def capture(o):
  nonlocal nextframe,held
  if o['t']>=nextframe-1e-9:frames.append(dict(o,phase=phase));poses.append(s.d.qpos.copy());nextframe+=1/30
  held=held+(.02 if hydro else .001) if o['depth_mm']>=1.98 and o['engaged'] and .15<o['axial_force_N']<5 and o['seat_force_N']>.1 and o['screw_speed_rad_s']<.15 else 0.
 capture(s.observe())
 for k in range(int(timeout/.02)):
  o=s.observe();t=o['t'];elapsed=t-start
  if phase=='press':
   if elapsed>=1.5:phase='turn';start=t;turn_angle=o['screw_angle_deg']
  elif phase=='turn':
   angle=min(30,(t-start)*30);release=0
   if o['screw_angle_deg']-turn_angle>=22 or t-start>1.5:phase='release';start=t;phase_angle=angle
  elif phase=='release':
   release=min(.004,(t-start)*.02)
   if t-start>=.25:phase='return';start=t
  elif phase=='return':
   angle=phase_angle*max(0,1-(t-start)/.35);release=.004
   if t-start>=.35:phase='regrasp';start=t
  elif phase=='regrasp':
   angle=0;release=.004*max(0,1-(t-start)/.35)
   if t-start>=.45:cycle+=1;phase='turn';start=t;turn_angle=o['screw_angle_deg']
  if o['depth_mm']>=1.98 and o['seat_force_N']>.1:phase='seat';release=0
  if mode=='hold':angle=0.;release=0.;phase='hold'
  q=s.ik(angle,release)
  if mode=='slide' and t>=1.5:
   phase='sliding';angles=[]
   for i in range(2):
    p=((t-slide_start)/2.4+.325+i*.5)%1;drive=p<.65;angles.append(-15+30*p/.65 if drive else 15-30*(p-.65)/.35)
    target=1.0 if drive else .06;mi=0 if drive else 1;slide_bias[i,mi]=np.clip(slide_bias[i,mi]-.0003*(target-s.loads[i+1])*.02,-.001,.001);slide_rad[i]=-.001*target+slide_bias[i,mi];s.modes[i]='drive' if drive else 'slide_return'
   q=s.ik_pose(CENTER-[0,0,s.d.qpos[s.depthq]],0,angles,slide_rad,-.0018);s.finger_roll=angles
  s.feed=float(np.clip(s.feed+.00025*(.6-s.axial)*.02,-.001,.01));cart=np.array([s.feed,0,0,0.]);s.stage='turn' if phase!='press' else 'drive_benchmark';
  if mode!='slide' or t<1.5:s.finger_roll=[angle,angle];s.modes=[phase,phase]
  events=s.step_full(q,cart,capture);actions.append(dict(t=t,phase=phase,q=q.tolist(),cart=cart.tolist(),angle=angle,release=release))
  if k%50==0:print(round(t,2),phase,round(s.observe()['depth_mm'],3),np.round(s.loads,2),round(s.axial,2),flush=True)
  if events:outcome=events[0];break
  if held>=.4:outcome='seated';break
 capture(s.observe());r=dict(id='drive-'+('hydro'if hydro else'point')+'-'+mode,method=mode,case='aligned',scope='drive subgraph only; no pickup in this experiment',outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,trace=actions,metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=float(s.peakpen*1000),cycles=cycle),target_depth_mm=2)
 if hydro:r['hydro']=s.hydro.stats
 return r,np.array(poses),s.xml
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--hydro',action='store_true');p.add_argument('--timeout',type=float,default=15);p.add_argument('--mode',default='regrasp');a=p.parse_args();r,q,xml=run(mode=a.mode,hydro=a.hydro,timeout=a.timeout);(OUT/(r['id']+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(r['id']+'.npz'),qpos=q);(OUT/(r['id']+'.xml')).write_text(xml);print('RESULT',r['outcome'],r['final'])
