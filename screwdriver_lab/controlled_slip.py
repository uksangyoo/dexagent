"""Load-regulated contact sliding with index-cap support; no clearance/regrasp phase."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from screwdriver_lab.reference_control import ReferenceSim,ROOT,OUT,CENTER
from screwdriver_lab.screw_sim import ScrewSim

class SlipSim(ReferenceSim):
 def __init__(self,grasp=None,pad_time=.004,torsion=.006,index_kp=8,thread_torque=.0008,native_ccd=1,pickup_position=None,friction=.9,initial_tilt=0):
  import screwdriver_lab.reference_control as rc
  import xml.etree.ElementTree as E
  saved=rc.REF
  if grasp:rc.REF=Path(grasp)
  try:super().__init__({"thread_torque_Nm":thread_torque,"friction":friction,**({"pickup_position":pickup_position}if pickup_position is not None else {})})
  finally:rc.REF=saved
  root=E.fromstring(self.xml)
  # Seating is determined by head/workpiece contact, not a duplicate travel stop.
  self.m.jnt_range[self.m.joint('screw_depth').id,1]=.004
  root.find(".//joint[@name='screw_depth']").set('range','-.0001 .004')
  if not native_ccd:
   self.m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_NATIVECCD
   E.SubElement(root.find('option'),'flag',nativeccd='disable')
  self.m.actuator_gainprm[4:8,0]=index_kp;self.m.actuator_biasprm[4:8,1]=-index_kp;self.m.actuator_biasprm[4:8,2]=-.18
  for a in list(root.find('actuator'))[4:8]:a.set('kp',str(index_kp));a.set('kv','.18')
  for i in self.tipids:
   self.m.geom_priority[i]=1;self.m.geom_solref[i]=[pad_time,1.]
   g=root.find(f".//geom[@name='{self.m.geom(i).name}']");g.set('priority','1');g.set('solref',f'{pad_time} 1');self.m.geom_friction[i,1]=torsion;g.set('friction',' '.join(map(str,self.m.geom_friction[i])))
   if torsion==0:self.m.geom_condim[i]=3;g.set('condim','3')
  # Use steel density for the shaft and bit rather than inconsistent hand-set masses.
  for name in ['shaft','driver_bit','driver_bit_cross']:
   geom=root.find(f".//geom[@name='{name}']");geom.attrib.pop('mass',None);geom.set('density','7800')
  for j,act in enumerate(root.find('actuator')):
   act.set('kp',str(self.m.actuator_gainprm[j,0]));act.set('kv',str(-self.m.actuator_biasprm[j,2]))
  q0=self.d.qpos.copy();ctrl0=self.d.ctrl.copy();self.xml=E.tostring(root,encoding='unicode');self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.d.qpos[:]=q0;self.d.ctrl[:]=ctrl0
  from screwdriver_lab.arm_scene import ArmIK
  self.arm=ArmIK(self.m);self.arm.last=self.d.qpos[self.arm.qi].copy()
  if initial_tilt:
   from scipy.spatial.transform import Rotation as Rot
   from screwdriver_lab.reference_full import clear_arm
   tilt=Rot.from_euler('x',initial_tilt,degrees=True);R=tilt.as_matrix();pivot=CENTER+np.array([0,0,-.123]);self.p0=pivot+R@(self.p0-pivot);self.R0=R@self.R0;self.arm_orientation_target=self.R0.copy();self.d.qpos[self.toq:self.toq+3]=pivot+R@(self.d.qpos[self.toq:self.toq+3]-pivot);self.d.qpos[self.toq+3:self.toq+7]=(tilt*Rot.from_quat(self.d.qpos[self.toq+3:self.toq+7],scalar_first=True)).as_quat(scalar_first=True);clear_arm(self,self.p0,self.R0);self.d.ctrl[20:]=self.d.qpos[self.arm.qi]
  mujoco.mj_forward(self.m,self.d);self.peak=self.peakpen=0;self.measure()
 def measure(self):
  ScrewSim.measure(self)
  self.handleloads=[0.,0.];self.slip=[0.,0.];self.signed_slip=[0.,0.];self.index_cap_load=0.;self.rigid_tool_load=0.;self.pad_points=np.zeros((2,3));self.pad_normals=np.zeros((2,3));self.pad_weights=np.zeros(2)
  axis=self.d.xmat[self.tool].reshape(3,3)[:,2]
  for k,c in enumerate(self.d.contact):
   ids=[c.geom1,c.geom2];gn=[self.m.geom(g).name for g in ids];bs=[int(self.m.geom_bodyid[g])for g in ids]
   if not any(x in gn for x in ['handle','cap']):continue
   target='handle'if'handle'in gn else'cap';obj=gn.index(target);other=1-obj;name=gn[other];body=self.m.body(bs[other]).name
   f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);load=max(0.,float(f[0]))
   if body.startswith('link_')and not name.startswith('finger_tip_'):self.rigid_tool_load+=load
   if target=='cap'and name=='finger_tip_2':self.index_cap_load+=load
   if target!='handle'or name not in ['finger_tip_1','finger_tip_3']:continue
   slot=0 if name=='finger_tip_1'else 1;self.handleloads[slot]+=load
   weight=max(load,1e-8);self.pad_points[slot]+=weight*c.pos;self.pad_normals[slot]+=weight*c.frame[:3]*(1 if obj==0 else -1);self.pad_weights[slot]+=weight
   if load<1e-7:continue
   J=np.zeros((3,self.m.nv));K=J.copy();mujoco.mj_jac(self.m,self.d,J,None,c.pos,bs[other]);mujoco.mj_jac(self.m,self.d,K,None,c.pos,self.tool);v=(J-K)@self.d.qvel;n=c.frame[:3];vt=v-n*np.dot(v,n)
   radial=c.pos-self.d.xpos[self.tool];radial-=axis*np.dot(radial,axis);radial/=max(1e-9,np.linalg.norm(radial));tangent=-np.cross(axis,radial)
   self.slip[slot]+=load*np.linalg.norm(vt)*1000;self.signed_slip[slot]+=load*np.dot(v,tangent)*1000
  for slot in range(2):
   if self.pad_weights[slot]>0:self.pad_points[slot]/=self.pad_weights[slot];self.pad_normals[slot]/=max(1e-9,np.linalg.norm(self.pad_normals[slot]))
  self.slip=[float(v/max(1e-9,f))for v,f in zip(self.slip,self.handleloads)];self.signed_slip=[float(v/max(1e-9,f))for v,f in zip(self.signed_slip,self.handleloads)]
 def observe(self):
  o=super().observe();o.update(handle_slip_signed_mm_s=self.signed_slip.copy(),rigid_tool_load_N=float(self.rigid_tool_load));return o

class SlipController:
 def __init__(self,s,drive_load=.3,slide_load=.012,amplitude=24,speed=16,gain=.004,pad_weight=.012,surface_tracking=1,force_p=.6,force_i=4,normal_d=2,orientation_gain=.015,velocity_gain=0,posture_gain=0):
  self.posture_gain=posture_gain;self.velocity_gain=velocity_gain;self.command_speed=np.zeros(2);self.orientation_gain=orientation_gain;self.force_p=force_p;self.force_i=force_i;self.normal_d=normal_d;self.surface_tracking=surface_tracking;self.s=s;self.drive_load=drive_load;self.slide_load=slide_load;self.amplitude=amplitude;self.speed=speed;self.gain=gain;self.pad_weight=pad_weight;self.phase='acquire';self.start=0.;self.angle=0.;self.compression=np.zeros(2);self.cap=-.0003;self.cycles=0;self.trace=[];self.ready=0.;self.phi=[];self.height=[];self.local=[];self.normal_local=[]
  for i in [0,2]:
   line=np.zeros(6);mujoco.mj_geomDistance(s.m,s.d,s.tipids[i],s.m.geom('handle').id,.1,line);b=s.bids[i]
   if np.linalg.norm(line[:3]-s.d.xpos[b])>.07:line[:3]=s.d.xpos[b]+s.d.xmat[b].reshape(3,3)@s.local[i];line[3:]=line[:3]
   delta=line[3:]-s.d.xpos[s.tool];axis=s.d.xmat[s.tool].reshape(3,3)[:,2];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex);self.phi.append(np.arctan2(np.dot(delta,ey),np.dot(delta,ex)));self.height.append(np.dot(delta,axis));self.local.append(s.d.xmat[b].reshape(3,3).T@(line[:3]-s.d.xpos[b]));n=delta-axis*np.dot(delta,axis);n/=np.linalg.norm(n);self.normal_local.append(s.d.xmat[b].reshape(3,3).T@(-n))
 def transition(self,phase,o):
  self.trace.append(dict(t=o['t'],from_node=self.phase,to=phase,screw_angle_deg=o['screw_angle_deg'],handle_loads_N=o['handle_forces_N'],index_cap_N=o['index_cap_load_N']));self.phase=phase;self.start=o['t'];self.ready=0.
 def tick(self,o):
  s=self.s;t=o['t'];dt=t-self.start;loads=np.asarray(s.handleloads);target=self.slide_load if self.phase in ['unload','slide_return']else self.drive_load
  stable=np.all(loads>target*.5)&np.all(loads<target*1.7)
  self.ready=self.ready+.02 if stable else 0.
  if self.phase=='acquire'and t>1. and self.ready>.1 and s.index_cap_load>.1:self.transition('drive',o)
  elif self.phase=='drive':
   self.angle=min(self.amplitude,self.angle+self.speed*.02)
   if self.angle>=self.amplitude:self.transition('unload',o)
  elif self.phase=='unload'and self.ready>.12:self.transition('slide_return',o)
  elif self.phase=='slide_return':
   # Continue tangential motion only while both side pads retain light load.
   if min(loads)>.002 and max(loads)<max(.04,self.slide_load*2):self.angle=max(0.,self.angle-self.speed*.02)
   if self.angle<=0:self.cycles+=1;self.transition('acquire',o)
  return self.pose_command(o)
 def pose_command(self,o,update_cap=True):
  s=self.s;t=o['t'];loads=np.asarray(s.handleloads);target=self.force_targets.copy()if hasattr(self,'force_targets')else np.full(2,self.slide_load if self.phase in ['unload','slide_return']else self.drive_load)
  self.compression=np.clip(self.compression+self.gain*(target-loads)*.02,-.004,.005)
  if update_cap:self.cap=float(np.clip(self.cap+.001*(s.axial-.6)*.02,-.007,.004))
  q=s.contact_ik(0,[0,0],cap=self.cap,active=(1,))
  R=s.d.xmat[s.tool].reshape(3,3);axis=R[:,2];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex);center=s.d.xpos[s.tool].copy();radius=float(s.m.geom('handle').size[0]);gaps=[]
  for slot,i in enumerate([0,2]):
   b=s.bids[i];rot=s.d.xmat[b].reshape(3,3)
   if not hasattr(self,'pose_local'):self.pose_local=[v.copy()for v in self.local]
   if s.handleloads[slot]>.001:self.pose_local[slot]=rot.T@(s.pad_points[slot]-s.d.xpos[b])
   point=s.d.xpos[b]+rot@self.pose_local[slot];delta=point-center;radial=delta-axis*np.dot(delta,axis);gaps.append((np.linalg.norm(radial)-radius)*1000);angle=self.angle[slot]if np.ndim(self.angle)else self.angle
   if hasattr(self,'return_angles') and self.phase in ['unload','slide_return','seat']:angle=self.return_angles[slot]*self.angle/self.amplitude
   phi=self.phi[slot]-np.radians(angle);n=ex*np.cos(phi)+ey*np.sin(phi);targetp=center+(radius-self.compression[slot])*n+self.height[slot]*axis
   if hasattr(self,"drive_center"):
    reference=self.drive_center-np.array([0,0,s.d.qpos[s.depthq]-self.drive_depth])
    targetp=reference+(radius-self.compression[slot])*np.array([np.cos(phi),np.sin(phi),0.])+self.height[slot]*np.array([0.,0.,1.])
   J=np.zeros((3,s.m.nv));Jr=J.copy();mujoco.mj_jac(s.m,s.d,J,Jr,point,b);cols=s.vi[4*i:4*i+4];A=J[:,cols];e=targetp-point;e*=min(1.,.002/max(1e-9,np.linalg.norm(e)))
   if self.pad_weight:
    normal=s.d.xmat[b].reshape(3,3)@self.normal_local[slot];x,y,z=normal;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);A=np.vstack([A,-self.pad_weight*skew@Jr[:,cols]]);e=np.r_[e,self.pad_weight*(-n-normal)]
   dq=np.linalg.solve(A.T@A+1e-6*np.eye(4),A.T@e);q[4*i:4*i+4]+=np.clip(.8*dq,-.04,.04)
  q[12:]=s.home[12:];s.stage='turn'if t>1.5 else'press';s.modes=[self.phase]*2;s.finger_roll=[self.angle]*2
  return s.bound(q),dict(force_target_N=np.asarray(target).tolist(),compression_mm=(1000*self.compression).tolist(),pad_gaps_mm=gaps,angle_deg=self.angle)

class SlipTorqueController(SlipController):
 """1 kHz hybrid normal-force / tangential-position servo, robot joints only."""
 def torques(self):
  if getattr(self.s,"resolved_rate",False):
   from screwdriver_lab.resolved_rate import command
   command(self);return
  if not hasattr(self,"force_integral"):self.force_integral=np.zeros(2);self.filtered_loads=np.asarray(self.s.handleloads).copy()
  s=self.s;R=s.d.xmat[s.tool].reshape(3,3);axis=R[:,2];center=s.d.xpos[s.tool];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex);radius=float(s.m.geom('handle').size[0]);target=self.slide_load if self.phase in ['unload','slide_return']else self.drive_load
  self.filtered_loads+=.2*(np.asarray(s.handleloads)-self.filtered_loads);self.debug_forces=[]
  for slot,i in enumerate([0,2]):
   b=s.bids[i];rot=s.d.xmat[b].reshape(3,3);point=s.d.xpos[b]+rot@self.local[slot];radial=point-center;radial-=axis*np.dot(radial,axis);n=radial/max(1e-9,np.linalg.norm(radial));angle=self.angle[slot]if np.ndim(self.angle)else self.angle
   if hasattr(self,'return_angles') and self.phase in ['unload','slide_return','seat']:angle=self.return_angles[slot]*self.angle/self.amplitude
   phi=self.phi[slot]-np.radians(angle);desired=center+radius*(ex*np.cos(phi)+ey*np.sin(phi))+self.height[slot]*axis
   if hasattr(self,"drive_center"):
    reference=self.drive_center-np.array([0,0,s.d.qpos[s.depthq]-self.drive_depth])
    desired=reference+radius*np.array([np.cos(phi),np.sin(phi),0.])+self.height[slot]*np.array([0.,0.,1.])
   J=np.zeros((3,s.m.nv));Jr=J.copy();mujoco.mj_jac(s.m,s.d,J,Jr,point,b);cols=s.vi[4*i:4*i+4];Ji=J[:,cols];v=J@s.d.qvel;error=desired-point;et=error-n*np.dot(error,n);vt=v-n*np.dot(v,n)
   # Keep a material point on the fingertip when contact is absent.
   # mj_geomDistance can return spurious closest points for these mesh pairs.
   if not hasattr(self,'live_local'):self.live_local=[v.copy()for v in self.local]
   if s.handleloads[slot]>.001:
    observed=rot.T@(s.pad_points[slot]-s.d.xpos[b]);self.live_local[slot]+=.2*(observed-self.live_local[slot])
   line=np.zeros(6);line[:3]=s.d.xpos[b]+rot@self.live_local[slot];line[3:]=line[:3]
   rn=line[:3]-center;rn-=axis*np.dot(rn,axis);rn/=max(1e-9,np.linalg.norm(rn))
   Jn=np.zeros((3,s.m.nv));mujoco.mj_jac(s.m,s.d,Jn,None,line[:3],b);vn=np.dot(Jn@s.d.qvel,rn)
   target=float(self.force_targets[slot])if hasattr(self,"force_targets")else target
   if hasattr(self,"drive_center"):
    reference=self.drive_center-np.array([0,0,s.d.qpos[s.depthq]-self.drive_depth])
    target=max(.002,target+getattr(s,"centering_gain",0.)*np.dot(center-reference,rn))
   self.force_integral[slot]=np.clip(self.force_integral[slot]+self.force_i*(target-self.filtered_loads[slot])*.001,-.5,.5)
   if s.handleloads[slot]<.002:self.force_integral[slot]=max(0.,self.force_integral[slot])
   Fn=np.clip(target+self.force_p*(target-self.filtered_loads[slot])+self.force_integral[slot]+self.normal_d*vn,-.3,1.5)
   gap=np.linalg.norm((line[:3]-center)-axis*np.dot(line[:3]-center,axis))-radius
   if s.handleloads[slot]<.005 and gap>2e-5:
    Fn=max(Fn,min(.2,.05+100*gap))
   # The normal and tangent tasks must use the SAME live surface point.
   # A fixed material point rotates off the patch and couples tangential error
   # into an unintended normal squeeze.
   point=line[:3];Ji=Jn[:,cols];v=Jn@s.d.qvel;error=desired-point;et=error-rn*np.dot(error,rn);vt=v-rn*np.dot(v,rn)
   axial_error=np.dot(et,axis)*axis;axial_velocity=np.dot(vt,axis)*axis
   F=getattr(s,"tangent_stiffness",150.)*(et-axial_error)-getattr(s,"tangent_damping",2.)*(vt-axial_velocity)+150*axial_error-2*axial_velocity
   if self.velocity_gain:
    tangent=-np.cross(axis,rn);tangent/=max(1e-9,np.linalg.norm(tangent));desired_speed=np.radians(self.command_speed[slot])*radius
    F=self.velocity_gain*(desired_speed-np.dot(v,tangent))*tangent+(150*np.dot(error,axis)-2*np.dot(v,axis))*axis
   if self.command_speed[slot]<-1e-6 and getattr(s,'return_friction_ff',0):
    tangent=-np.cross(axis,rn);tangent/=max(1e-9,np.linalg.norm(tangent))
    # Robot-side Coulomb friction compensation during sliding only. The
    # object still receives force solely through the solved contact law.
    F-=s.return_friction_ff*s.m.geom_friction[s.tipids[i],0]*self.filtered_loads[slot]*tangent
   if self.phase=='seat' and getattr(s,'effort_seating',False):
    tangent=-np.cross(axis,rn);tangent/=max(1e-9,np.linalg.norm(tangent))
    # Final tightening is delivered by bounded finger motor torques through
    # the actual pad/handle, bit/slot and thread contacts.
    F=s.seating_tangent_N*tangent-getattr(s,'tangent_damping',2.)*(vt-axial_velocity)+150*axial_error-2*axial_velocity
   normal=rot@self.normal_local[slot];ori=np.cross(normal,-rn);N=np.eye(4)-Ji.T@np.linalg.solve(Ji@Ji.T+1e-8*np.eye(3),Ji)
   tau=Jn[:,cols].T@(-rn*Fn)+Ji.T@F+N@(Jr[:,cols].T@(self.orientation_gain*ori)+self.posture_gain*(getattr(self,'posture_ref',s.home)[4*i:4*i+4]-s.d.qpos[s.qi[4*i:4*i+4]])-.01*s.d.qvel[cols])
   if getattr(s,"direct_task_effort",False):
    if not hasattr(s,"task_effort"):s.task_effort={}
    s.task_effort[i]=tau.copy()
   # Invert the existing position actuator's force law, retaining its limits.
   qtarget=s.d.qpos[s.qi[4*i:4*i+4]]+(tau+.08*s.d.qvel[cols])/2.
   s.d.ctrl[4*i:4*i+4]=np.clip(qtarget,s.m.jnt_range[s.jids[4*i:4*i+4],0]+1e-5,s.m.jnt_range[s.jids[4*i:4*i+4],1]-1e-5)
   self.debug_forces.append(dict(Fn=float(Fn),target=target,point=point.tolist(),normal=rn.tolist(),tangent_force=F.tolist(),vn=float(vn),tau=tau.tolist()))
 def step(self,q,cart,listener):
  s=self.s;s.apply_control(q,cart);events=[]
  for _ in range(20):
   if not hasattr(self,"guard_loads"):self.guard_loads=np.asarray(s.handleloads).copy()
   self.guard_loads+=(1-np.exp(-.001/.05))*(np.asarray(s.handleloads)-self.guard_loads)
   if self.phase!='seat' or getattr(s,'effort_seating',False):self.torques()
   if getattr(s,'hydro',None) is not None:s.hydro.step();s.hydro.measure()
   else:mujoco.mj_step(s.m,s.d);mujoco.mj_forward(s.m,s.d);s.measure()
   o=s.observe();listener(o)
   if s.peak>12:events.append('force_limit')
   if s.pen>.0006:events.append('penetration')
   if s.forbidden>.3:events.append('shaft_contact')
   if s.rigid_tool_load>.2:events.append('rigid_link_contact')
   if s.stage=='turn'and(o['tilt_deg']>8 or o['bit_lateral_mm']>1.2 or o['bit_height_mm']>3):events.append('engagement_lost')
   if events:break
  return events

class SlipAlternatingController(SlipTorqueController):
 """One loaded pad turns the tool while the other slides back under positive load."""
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.angle=np.zeros(2);self.force_targets=np.full(2,self.drive_load);self.driver=0;self.phase='establish';self.last_swap=0.;self.half_cycles=0
 def tick(self,o):
  s=self.s;t=o['t'];loads=np.asarray(s.handleloads);old_angle=self.angle.copy()
  if self.phase=='establish':
   if t>1 and min(loads)>.07 and s.index_cap_load>.1:self.transition('regulate',o)
  else:
   target=np.full(2,self.slide_load);target[self.driver]=self.drive_load;self.force_targets+=np.clip(target-self.force_targets,-.008,.008)
   mu=float(s.m.geom_friction[s.tipids[0],0]);radius=float(s.m.geom('handle').size[0]);resistance=float(s.m.dof_frictionloss[s.anglev]);budget=mu*radius*(loads[self.driver]-loads[1-self.driver])
   ready=loads[self.driver]>self.drive_load*.5 and loads[1-self.driver]>.01 and budget>1.2*resistance
   if ready:
    self.phase='slip_turn';move=np.array([1.,-1.])if self.driver==0 else np.array([-1.,1.]);self.angle=np.clip(self.angle+move*self.speed*.02,-self.amplitude/2,self.amplitude/2)
    if self.angle[self.driver]>=self.amplitude/2-1e-9:
     self.half_cycles+=1;self.cycles=self.half_cycles//2;self.driver=1-self.driver;self.trace.append(dict(t=t,to='swap_loads',driver=self.driver,loads=loads.tolist(),screw_angle_deg=o['screw_angle_deg']));self.phase='regulate';self.start=t
   else:self.phase='regulate'
  self.command_speed=(self.angle-old_angle)/.02
  self.cap=float(np.clip(self.cap+.001*(s.axial-.6)*.02,-.007,.004));q=s.contact_ik(0,[0,0],cap=self.cap,active=(1,));q[12:]=s.home[12:];s.stage='turn'if t>1.5 else'press';s.modes=['drive'if i==self.driver else'slide_return'for i in range(2)]if self.phase!='establish'else['establish']*2;s.finger_roll=self.angle.tolist()
  gaps=[mujoco.mj_geomDistance(s.m,s.d,s.tipids[i],s.m.geom('handle').id,.1,None)*1000 for i in [0,2]]
  return q,dict(force_target_N=self.force_targets.tolist(),pad_gaps_mm=gaps,angle_deg=self.angle.tolist(),driver=self.driver)

class SlipPairedController(SlipTorqueController):
 """Loaded drive, light-contact sliding reset against passive thread resistance."""
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.force_targets=np.full(2,min(.05,self.drive_load))
  from screwdriver_lab.slip_graph import SlipPolicyGraph
  self.graph=SlipPolicyGraph()

 def transition(self,phase,o):
  super().transition(phase,o)
  if phase in ['unload','seat']:
   if phase=='seat':self.angle=self.amplitude;self.seat_q=self.s.d.ctrl[:20].copy()
   s=self.s;axis=s.d.xmat[s.tool].reshape(3,3)[:,2];ex=np.array([1.,0.,0.]);ex-=axis*np.dot(ex,axis);ex/=np.linalg.norm(ex);ey=np.cross(axis,ex);angles=[]
   for slot,i in enumerate([0,2]):
    point=s.pad_points[slot] if s.pad_weights[slot]>0 else s.d.xpos[s.bids[i]]+s.d.xmat[s.bids[i]].reshape(3,3)@self.local[slot]
    delta=point-s.d.xpos[s.tool];phi=np.arctan2(np.dot(delta,ey),np.dot(delta,ex));angles.append(np.degrees(np.arctan2(np.sin(self.phi[slot]-phi),np.cos(self.phi[slot]-phi))))
   self.return_angles=np.array(angles)
   if hasattr(self,'force_integral'):self.force_integral[:]=0

 def tick(self,o):
  s=self.s;t=o['t'];old=self.angle;drag,resistance=self.graph.advance(self,o)
  self.command_speed[:]=(self.angle-old)/.02
  if self.phase=='slide_return'and hasattr(self,'return_angles'):self.command_speed*=self.return_angles/self.amplitude
  goal=self.slide_load if self.phase in ['unload','slide_return']else self.drive_load
  self.force_targets+=np.clip(goal-self.force_targets,-.01,.01)
  self.cap=float(np.clip(self.cap+.001*(s.axial-(1.0 if self.phase=='seat'else .6))*.02,-.007,.004));q=s.contact_ik(0,[0,0],cap=self.cap,active=(1,))
  if self.phase=='seat':q[:4]=self.seat_q[:4];q[8:12]=self.seat_q[8:12]
  q[12:]=s.home[12:];s.stage='turn'if t>1.5 else'press';s.modes=[self.phase]*2;s.finger_roll=[self.angle]*2
  return q,dict(force_target_N=self.slide_load if self.phase in ['unload','slide_return']else self.drive_load,estimated_return_drag_Nm=drag,thread_resistance_Nm=resistance,angle_deg=self.angle)

class SlipAdmittanceController(SlipPairedController):
 def torques(self):pass
 def tick(self,o):
  super().tick(o)
  return self.pose_command(o,update_cap=False)

def run(name,timeout=15,servo='position',grasp=None,pad_time=.004,torsion=.006,backend='mujoco',index_kp=8,thread_torque=.0008,native_ccd=1,hydro_kh=5e10,friction=.9,initial_tilt=0,**options):
 s=SlipSim(grasp,pad_time,torsion,index_kp,thread_torque,native_ccd,friction=friction,initial_tilt=initial_tilt)
 if backend=='hydro':
  from screwdriver_lab.slip_hydro import SlipHydroBackend
  s.hydro=SlipHydroBackend(s,kh=hydro_kh)
 c=(SlipAdmittanceController if servo=='admittance'else SlipPairedController if servo=='paired'else SlipAlternatingController if servo=='alternating'else SlipTorqueController if servo=='hybrid'else SlipController)(s,**options);frames=[];qs=[];commands=[];next_frame=0.;outcome='timeout';held=0.
 from screwdriver_lab.slip_metrics import SlipMetrics
 evidence=SlipMetrics()
 source_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),ROOT/'screwdriver_lab/reference_control.py',ROOT/'screwdriver_lab/slip_hydro.py',ROOT/'screwdriver_lab/slip_metrics.py',ROOT/'screwdriver_lab/slip_graph.py',ROOT/'screwdriver_lab/slip_compare.py']}
 source_dir=OUT/'source_snapshots';source_dir.mkdir(exist_ok=True)
 for filename,digest in source_sha.items():
  p=ROOT/'screwdriver_lab'/filename;(source_dir/(digest+'_'+filename)).write_bytes(p.read_bytes())
 def capture(o):
  nonlocal next_frame,held
  evidence.add(o,s.modes)
  seated=o['depth_mm']>=1.98 and o['engaged'] and o['screw_speed_rad_s']<.15 and .15<o['axial_force_N']<5 and o['seat_force_N']>.1
  held=held+.001 if seated else 0.
  if o['t']+1e-9>=next_frame:frames.append(dict(o,phase=c.phase));qs.append(s.d.qpos.copy());next_frame+=1/30
 capture(s.observe())
 for k in range(int(timeout/.02)):
  o=s.observe()
  if servo in ['paired','admittance']and o['depth_mm']>=1.98 and o['seat_force_N']>.1 and c.phase!='seat':c.transition('seat',o)
  q,cmd=c.tick(o);s.feed=float(s.d.qpos[s.depthq]);events=(c.step if servo in ['hybrid','alternating','paired','admittance']else s.step_full)(q,np.array([s.feed,0,0,0.]),capture);commands.append(dict(t=o['t'],phase=c.phase,**cmd))
  if k%50==0:print(round(o['t'],2),c.phase,'angle',np.round(c.angle,1),'depth',round(o['depth_mm'],3),'loads',np.round(s.handleloads,3),'slip',np.round(s.signed_slip,2),'bias',np.round(c.compression*1000,3),flush=True)
  if events:outcome=events[0];break
  if held>=.4:outcome='seated';break
  if s.rigid_tool_load>.2:outcome='rigid_link_contact';break
  if o['t']-c.start>8 and c.phase in ['acquire','unload']:outcome='load_regulation_failed';break
 frames.append(dict(s.observe(),phase=c.phase));qs.append(s.d.qpos.copy());r=dict(id=name,method='controlled slip',case='nominal',scope='drive-only controlled-slip mechanism test',outcome=outcome,success=outcome=='seated',final=s.observe(),frames=frames,trace=commands,commands=commands,cycle_trace=c.trace,grasp_calibration=s.grasp_calibration,parameters=dict(servo=servo,pad_time=pad_time,torsion=torsion,backend=backend,index_kp=index_kp,thread_torque=thread_torque,native_ccd=native_ccd,hydro_kh=hydro_kh,friction=friction,initial_tilt=initial_tilt,grasp=grasp,**options),metrics=dict(peak_finger_N=float(s.peak),max_penetration_mm=float(s.peakpen*1000),cycles=c.cycles,controlled_slip=evidence.report()),target_depth_mm=2,policy_graph=c.graph.schema if hasattr(c,'graph')else None,initial_state_sha256=hashlib.sha256(qs[0].tobytes()).hexdigest(),mujoco_version=mujoco.__version__,source_sha256=source_sha,backend_stats=s.hydro.stats if getattr(s,'hydro',None) else None)
 (OUT/(name+'.json')).write_text(json.dumps(r));np.savez_compressed(OUT/(name+'.npz'),qpos=np.array(qs));(OUT/(name+'.xml')).write_text(s.xml);print('RESULT',name,outcome,'cycles',c.cycles,'depth',r['final']['depth_mm'],flush=True)
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',default='slip-v1');p.add_argument('--timeout',type=float,default=15);p.add_argument('--servo',choices=['position','hybrid','alternating','paired','admittance'],default='position');p.add_argument('--grasp');p.add_argument('--pad-time',type=float,default=.004);p.add_argument('--torsion',type=float,default=.006);p.add_argument('--index-kp',type=float,default=8);p.add_argument('--thread-torque',type=float,default=.0008);p.add_argument('--native-ccd',type=int,default=1);p.add_argument('--hydro-kh',type=float,default=5e10);p.add_argument('--friction',type=float,default=.9);p.add_argument('--initial-tilt',type=float,default=0);p.add_argument('--backend',choices=['mujoco','hydro'],default='mujoco')
 for arg,default in [('drive-load',.3),('slide-load',.012),('amplitude',24),('speed',16),('gain',.004),('pad-weight',.012),('surface-tracking',1),('force-p',.6),('force-i',4),('normal-d',2),('orientation-gain',.015),('velocity-gain',0),('posture-gain',0)]:p.add_argument('--'+arg,type=float,default=default)
 a=p.parse_args();run(**vars(a))
