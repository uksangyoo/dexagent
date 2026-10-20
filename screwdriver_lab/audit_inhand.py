"""Audit axis reorientation separately from axial spin using recorded rigid poses."""
import json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from screwdriver_lab.full_scene import OUT

def audit(name,save=True):
 r=json.loads((OUT/(name+'.json')).read_text());q=np.load(OUT/(name+'.npz'))['qpos'];m=mujoco.MjModel.from_xml_path(str(OUT/(name+'.xml')));d=mujoco.MjData(m);assert len(q)==len(r['frames']) and np.isfinite(q).all() and m.nu==26
 tool=m.body('screwdriver').id;palm=m.body('palm').id;axes=[];hands=[];rotations=[];pads=[[],[]]
 for pose in q:
  d.qpos[:]=pose;mujoco.mj_forward(m,d);R=d.xmat[tool].reshape(3,3).copy();axes.append(R[:,2]);rotations.append(R);hands.append(d.xmat[palm].reshape(3,3).copy())
  for slot,i in enumerate([1,3]):pads[slot].append(d.xmat[m.body(f'link_{i}_4').id].reshape(3,3).copy())
 ts=np.array([f['t']for f in r['frames']]);start=next((x['t']for x in r['graph_trace']if x['from_node']=='grasp'and x['exit']=='complete'),0);end=next((x['t']for x in r['graph_trace']if x['from_node']=='reorient'and x['exit']=='complete'),ts[-1]);drive=next((x['t']for x in r['graph_trace']if x['from_node']=='align'and x['exit']=='complete'),None)
 a=int(np.argmin(abs(ts-start)));b=int(np.argmin(abs(ts-end)));h0=hands[a];v0=h0.T@axes[a];spin=np.zeros(len(q))
 for i in range(1,len(q)):
  axis=axes[i]+axes[i-1];axis/=np.linalg.norm(axis);spin[i]=spin[i-1]-np.rad2deg(np.dot(Rotation.from_matrix(rotations[i]@rotations[i-1].T).as_rotvec(),axis))
 d0=int(np.argmin(abs(ts-drive)))if drive is not None else len(ts)-1
 for i,f in enumerate(r['frames']):
  f['inhand_axis_turn_deg']=float(np.rad2deg(np.arccos(np.clip(v0@(hands[i].T@axes[i]),-1,1))));f['palm_turn_since_lift_deg']=float(np.rad2deg(Rotation.from_matrix(hands[i]@h0.T).magnitude()));f['handle_spin_since_engagement_deg']=float(spin[i]-spin[d0])if i>=d0 else 0.
 pad_angles=[float(np.rad2deg(np.arccos(np.clip((p[a].T@axes[a])@(p[b].T@axes[b]),-1,1))))for p in pads]
 summary=dict(id=name,pivot_start_s=float(ts[a]),pivot_end_s=float(ts[b]),inhand_axis_turn_deg=r['frames'][b]['inhand_axis_turn_deg'],palm_turn_during_pivot_deg=r['frames'][b]['palm_turn_since_lift_deg'],max_palm_rotation_during_pivot_deg=max(f['palm_turn_since_lift_deg']for f in r['frames'][a:b+1]),axis_turn_relative_thumb_middle_deg=pad_angles,drive_start_s=drive,handle_spin_since_engagement_deg=float(spin[-1]-spin[d0])if drive is not None else None,screw_rotation_deg=r['final']['screw_angle_deg'],depth_mm=r['final']['depth_mm'],outcome=r['outcome'],pose_sample_rate_Hz=30,spin_measurement='Incremental rotation-vector projection onto the mean tool axis; avoids Euler-yaw singularity during the horizontal pivot',source_json_sha256=hashlib.sha256((OUT/(name+'.json')).read_bytes()).hexdigest(),source_qpos_sha256=hashlib.sha256(q.tobytes()).hexdigest(),robot_actuators=26,tool_free=bool(m.jnt_type[m.joint('tool_free').id]==mujoco.mjtJoint.mjJNT_FREE))
 if save:(OUT/(name+'-audit.json')).write_text(json.dumps(summary,indent=2))
 return r,summary
if __name__=='__main__':
 import sys
 for name in sys.argv[1:]:print(json.dumps(audit(name)[1],indent=2))
