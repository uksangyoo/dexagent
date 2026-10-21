"""Motor-only bridge from a measured supported grasp to bit engagement and drive.

All feedback is read from the plant. This module never assigns an object pose,
velocity, external wrench, or contact parameter.
"""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from dexagent.sim.controllers import ManipulationPolicy
from dexagent.sim.seating import geometry,flush
from dexagent.planning.kinematics import palm_velocity_for_slipping_tool
from dexagent.planning.metrics import valid_thread_motion

class AssemblyBridge:
 def __init__(self,h,drive_goal_deg=90,transport_speed=.06,interface='hom',drive_cycle='alternating',drive_load=.4,return_load=.06,drive_speed=45,reserve_margin=0.,drive_tangent_kp=None,drive_tangent_kd=3.,alignment_filter_s=0.):
  self.interface=interface;self.transport_speed=transport_speed;self.h=h;self.active=False;self.controller=None;self.stage='pickup';self.trace=[];self.goal=drive_goal_deg
  self.angles=np.zeros(2);self.previous=None;self.screw_start_z=None;self.hold_s=0.;self.outcome=None;self.last={}
  self.xy_integral=np.zeros(2)
  self.slot_alignment_hold=0.;self.slot_ready=False;self.insertion_retries=0
  self.drive_parameters=dict(cycle=drive_cycle,drive_load=drive_load,return_load=return_load,speed=drive_speed)
  if reserve_margin:self.drive_parameters['reserve_margin']=reserve_margin
  if drive_tangent_kp is not None:self.drive_parameters.update(tangent_kp=drive_tangent_kp,tangent_kd=drive_tangent_kd)
  if alignment_filter_s:self.drive_parameters['alignment_filter_s']=alignment_filter_s
  self.filtered_tool_omega=None
  self.workspace_plan=None
  self.hold_only=False;self.hold_reference_angles=None
 def start_drive(self,locals_,points):
  """Initialize motor references from measured contacts, without resetting physics."""
  h=self.h;d=h.d
  h.finger_impedances[1]=(200.,3.)
  initial=[dict(local=np.asarray(l).tolist(),target=np.asarray(p).tolist())for l,p in zip(locals_,points)]
  parameters=self.drive_parameters.copy();margin=parameters.pop('reserve_margin',0.);kp=parameters.pop('tangent_kp',None);kd=parameters.pop('tangent_kd',3.);parameters.pop('alignment_filter_s',None)
  if kp is not None:
   for i in [0,1,2]:h.finger_impedances[i]=(kp,kd)
  self.controller=ManipulationPolicy(h,initial,self.interface,**parameters,hybrid=True,hold_load=parameters['drive_load'],hold_cap_load=.3)
  if margin:
   from dexagent.planning.workspace_plan import plan
   target,self.workspace_plan=plan(h.m,d.qpos.copy(),locals_,margin)
   if not self.workspace_plan['accepted']:self.outcome='workspace_plan_infeasible'
   else:self.controller.prepare_posture(target[h.qi])
  self.controller.phase_start=float(d.time)
  tool=h.m.body('screwdriver').id;screw=h.m.body('screw').id
  self.screw_start_z=float(d.xpos[screw,2])
  self.previous=np.array([np.arctan2(d.xmat[b].reshape(3,3)[1,0],d.xmat[b].reshape(3,3)[0,0])for b in [tool,screw]])
  self.route('drive')
 def start(self):
  h=self.h;d=h.d;tool=h.m.body('screwdriver').id
  self.active=True;self.height=max(float((d.xpos[tool]-.123*d.xmat[tool].reshape(3,3)[:,2])[2]),float(d.xpos[h.m.body('screw').id,2]+.003));self.route('transport')
 def route(self,stage):
  self.trace.append(dict(t=float(self.h.d.time),from_node=self.stage,to=stage));self.stage=stage
 def update(self,locals_,points):
  h=self.h;d=h.d;m=h.m;tool=m.body('screwdriver').id;screw=m.body('screw').id
  R=d.xmat[tool].reshape(3,3);axis=R[:,2];tip=d.xpos[tool]-.123*axis
  screw_R=d.xmat[screw].reshape(3,3);screw_pos=d.xpos[screw].copy();yaw=np.arctan2(R[1,0],R[0,0]);syaw=np.arctan2(screw_R[1,0],screw_R[0,0]);yaw_error=(syaw-yaw+np.pi/2)%np.pi-np.pi/2
  bit_load=0.
  for contact in h.contacts():
   names=set(contact['names'])
   if 'driver_bit'in names and any(n.startswith('slot_')for n in names):bit_load+=abs(float(contact['force'][2]))
  relative_height=float(tip[2]-screw_pos[2]);xy_error=screw_pos[:2]-tip[:2];tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
  # Preserve the upright transport servo. Lower cap impedance only for
  # fine insertion, where the stiff servo excites contact-scale oscillations.
  if self.controller is None and (relative_height<.004 or self.stage=='align_slot'):
   h.finger_impedances[1]=(200.,3.)
  self.last=dict(stage=self.stage,tip_xy_error_mm=float(np.linalg.norm(xy_error)*1000),tip_height_above_slot_floor_mm=relative_height*1000,blade_yaw_error_deg=float(np.degrees(yaw_error)),bit_axial_load_N=bit_load,tilt_deg=tilt,tool_deg=float(self.angles[0]),screw_deg=float(self.angles[1]))
  if self.controller is not None:
   c=self.controller;c.measure();yaws=np.array([yaw,syaw])
   if c.preparation_failed:self.outcome='workspace_preparation_failed'
   if self.previous is not None:self.angles-=np.degrees(np.arctan2(np.sin(yaws-self.previous),np.cos(yaws-self.previous)))
   self.previous=yaws
   depth=(self.screw_start_z-screw_pos[2])*1000
   speed=0. if not hasattr(self,'last_screw_deg') else abs(self.angles[1]-self.last_screw_deg)/.002
   self.last_screw_deg=float(self.angles[1]);seat=geometry(m,d)
   if c.phase!='hold' and c.posture_preparation is None and ((self.goal and self.angles[1]>=self.goal)or(flush(seat)and c.seat_load>.2)):
    c.hold_at_contact();self.hold_reference_angles=self.angles.copy();self.route('drive_hold')
   hold_error=abs(float(self.angles[1]-self.hold_reference_angles[1]))if self.hold_reference_angles is not None else None
   stable=(c.phase=='hold'and hold_error is not None and hold_error<=5 and speed<5 and tilt<3 and min(c.loads[[0,2]])>.05 and c.loads[1]>.1 and c.unexpected_load<.1)
   self.hold_s=self.hold_s+.002 if stable else 0.
   if self.hold_s>=.5:
    if flush(seat)and c.seat_load>.2:
     self.outcome='full_task_succeeded'if valid_thread_motion(float(self.angles[1]),float(depth))else'seated_without_valid_thread_motion'
    else:self.outcome='prepared_hold_succeeded'if self.hold_only else'drive_subgoal_succeeded'
   self.last.update(tool_deg=float(self.angles[0]),screw_deg=float(self.angles[1]),depth_mm=float(depth),accepted_hold_s=self.hold_s,seat_clearance_mm=seat['flat_seat_remaining_mm'],controller_phase=c.phase,stroke_deg=float(np.degrees(c.stroke)),strokes_deg=np.degrees(c.strokes).tolist(),normal_targets_N=c.normal_targets.tolist(),measured_slip_mm_s=c.slips.tolist(),drive_cycles=c.cycles)
   return np.zeros(3),np.zeros(3)
  if bit_load>2:self.outcome='excessive_bit_load';return np.zeros(3),np.zeros(3)
  # Bounded integral velocity removes the loaded grasp's steady alignment
  # error. Integrate only near the target; never alter a body state or wrench.
  if np.linalg.norm(xy_error)<.003:
   self.xy_integral=np.clip(self.xy_integral+.002*.8*xy_error,-.002,.002)
  velocity=np.r_[np.clip(3*xy_error+self.xy_integral,-self.transport_speed,self.transport_speed),0.]
  # The cap finger stabilizes tilt while the side contacts permit pivoting.
  # Arm tilt is not a rigid tool-orientation actuator in this grasp.
  omega=np.array([0.,0.,float(np.clip(2*yaw_error,-.15,.15))])
  if self.stage=='transport':
   velocity[2]=np.clip(3*(self.height-tip[2]),-.025,.025)
   if np.linalg.norm(xy_error)<.00005 and tilt<1 and abs(yaw_error)<np.radians(1):self.route('approach')
  if self.stage=='approach':
   if relative_height<.0035 and not self.slot_ready:self.route('align_slot')
   velocity[2]=np.clip(3*(screw_pos[2]-.0001-tip[2]),-.0015 if relative_height<.004 else-.04,.01)
   if bit_load>.02:
    velocity[2]=0.
    if relative_height<.0005 and np.linalg.norm(xy_error)<.0003 and tilt<3:
     self.start_drive(locals_,points)
    elif relative_height>.0008:
     self.insertion_retries+=1;self.slot_ready=False;self.slot_alignment_hold=0.
     if self.insertion_retries>3:self.outcome='head_contact_outside_slot'
     else:self.route('align_slot')
   elif self.slot_ready and relative_height<.0035 and np.linalg.norm(xy_error)>.00007:
    self.slot_ready=False;self.slot_alignment_hold=0.;self.route('align_slot')
  if self.stage=='align_slot':
   velocity[2]=np.clip(3*(screw_pos[2]+.003-tip[2]),-.003,.003)
   aligned=np.linalg.norm(xy_error)<.00004 and abs(yaw_error)<np.radians(.3) and tilt<1 and abs(relative_height-.003)<.0001
   self.slot_alignment_hold=self.slot_alignment_hold+.002 if aligned else 0.
   if self.slot_alignment_hold>=.2:self.slot_ready=True;self.route('approach')
  # A slipping tool need not follow the commanded arm angular velocity.
  # Use its measured rotation for the shaft lever, and the arm command only
  # for the palm-to-grasp-center lever. All outputs remain motor velocities.
  Jr=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,Jr,tool)
  tool_omega=Jr@d.qvel;tau=self.drive_parameters.get('alignment_filter_s',0.)
  if self.filtered_tool_omega is None:self.filtered_tool_omega=tool_omega.copy()
  self.filtered_tool_omega+=(1-np.exp(-.002/tau))*(tool_omega-self.filtered_tool_omega)if tau else tool_omega-self.filtered_tool_omega
  palm_velocity=palm_velocity_for_slipping_tool(velocity,omega,self.filtered_tool_omega,tip,d.xpos[tool],d.xpos[m.body('palm').id])
  self.last.update(insertion_retries=self.insertion_retries,slot_alignment_hold_s=self.slot_alignment_hold)
  return palm_velocity,omega
