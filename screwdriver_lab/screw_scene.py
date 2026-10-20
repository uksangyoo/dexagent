"""Precision screwdriver, contact-driven slotted screw and ideal helical thread."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from screwdriver_lab.scene import ROOT,nums
OUT=ROOT/'screwdriver_lab/artifacts/screw_in'
CENTER=np.array([0.,0.,.25112]);R=np.array([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])
PITCH=.001;DEPTH=.001;HEAD=.128;CAP=.290

def build(case=None,palm_pos=None):
 c=case or {};radius=c.get('radius_mm',12)/1000
 root=ET.parse(ROOT/'third-party/tesollo_model/dg5fs/dg5fs_right.xml').getroot();root.set('model','precision_screw_in')
 root.find('compiler').set('meshdir',str(ROOT/'third-party/tesollo_model/dg5fs/meshes'))
 ET.SubElement(root,'option',timestep='.001',integrator='implicitfast',cone='elliptic',iterations='80',gravity='0 0 -9.81')
 default=ET.SubElement(root,'default');ET.SubElement(default,'geom',friction=f"{c.get('friction',.9)} .0005 .00002",solref='.004 1',solimp='.95 .99 .0005',condim='3')
 ET.SubElement(default,'joint',damping='.06',armature='.001')
 w=root.find('worldbody');palm=ET.Element('body',name='palm',pos=nums(palm_pos if palm_pos is not None else [.10,0,.32]),quat=nums(Rotation.from_matrix(R).as_quat(scalar_first=True)),gravcomp='1')
 ET.SubElement(palm,'joint',name='feed',type='slide',axis='1 0 0',range='-.01 .015',damping='20',armature='.1')
 for e in list(w):w.remove(e);palm.append(e)
 w.append(palm);act=ET.SubElement(root,'actuator')
 for b in palm.findall('.//body'):b.set('gravcomp','1')
 for i in range(1,6):
  b=palm.find(f".//body[@name='link_{i}_4']");tip=b.find(f"geom[@mesh='link_{i}_tip']");tip.set('name',f'finger_tip_{i}')
  ET.SubElement(b,'site',name=f'tip_{i}',pos=tip.get('pos'),size='.002',rgba='1 1 1 0')
 for j in palm.findall('.//joint'):
  if j.get('name')!='feed':ET.SubElement(act,'position',name='servo_'+j.get('name'),joint=j.get('name'),kp='2',kv='.08',forcerange='-.3 .3')
 ET.SubElement(act,'position',name='feed_servo',joint='feed',kp='3000',kv='40',forcerange='-15 15')
 contact=ET.SubElement(root,'contact')
 for i in range(1,6):ET.SubElement(contact,'exclude',body1='palm',body2=f'link_{i}_1')
 ET.SubElement(contact,'exclude',body1='palm',body2='link_1_2')
 yaw=c.get('slot_yaw_deg',0);quat=nums(Rotation.from_euler('z',-yaw,degrees=True).as_quat(scalar_first=True))
 tool=ET.SubElement(w,'body',name='screwdriver',pos=nums(CENTER),quat=quat);ET.SubElement(tool,'freejoint',name='tool_free')
 ET.SubElement(tool,'geom',name='handle',type='cylinder',size=f'{radius} .035',mass='.040',rgba='.035 .04 .05 1')
 ET.SubElement(tool,'geom',name='shaft',type='cylinder',size='.0018 .040',pos='0 0 -.075',mass='.012',rgba='.65 .7 .75 1')
 ET.SubElement(tool,'geom',name='driver_bit',type='box',size='.003 .00075 .004',pos='0 0 -.119',mass='.001',rgba='.65 .7 .75 1')
 ET.SubElement(tool,'geom',name='stripe',type='box',size='.0005 .001 .028',pos=f'{radius} 0 0',contype='0',conaffinity='0',mass='.00001')
 cap=ET.SubElement(tool,'body',name='swivel_cap',pos='0 0 .039');ET.SubElement(cap,'joint',name='cap_bearing',type='hinge',axis='0 0 1',limited='false',damping='.000001',armature='.000001')
 ET.SubElement(cap,'geom',name='cap',type='cylinder',size=f'{radius*1.1} .004',mass='.006',rgba='.6 .03 .02 1')
 # Head slot has real side walls and a floor. The bit is free to lose engagement.
 screw=ET.SubElement(w,'body',name='screw',pos=f'0 0 {HEAD}',quat=quat)
 ET.SubElement(screw,'joint',name='screw_depth',type='slide',axis='0 0 -1',range=f'-.0001 {DEPTH}',damping='1',armature='.001',solreflimit='.004 1')
 ET.SubElement(screw,'joint',name='screw_angle',type='hinge',axis='0 0 -1',limited='false',damping='.00002',frictionloss=str(c.get('thread_torque_Nm',.0008)),armature='.000001')
 ET.SubElement(screw,'geom',name='screw_core',type='cylinder',size='.003 .008',pos='0 0 -.008',mass='.004')
 ET.SubElement(screw,'geom',name='slot_floor',type='cylinder',size='.005 .0005',pos='0 0 -.0005',mass='.002')
 for sign in [-1,1]:ET.SubElement(screw,'geom',name=f'slot_wall_{sign}',type='box',size='.004 .0018 .0015',pos=f'0 {sign*.0028} .0015',mass='.001')
 eq=ET.SubElement(root,'equality');ET.SubElement(eq,'joint',name='helical_thread',joint1='screw_angle',joint2='screw_depth',polycoef=f'0 {2*np.pi/PITCH} 0 0 0',solref='.002 1',solimp='.999 .999 .0001')
 # Hole is represented by four blocks; screw guidance is the helical constraint.
 top=HEAD-.001-DEPTH
 for axis in [0,1]:
  for sign in [-1,1]:
   size=[.055,.055,.008];pos=[0.,0.,top-.008];size[axis]=.026;pos[axis]=sign*.0293
   if axis==1:size[0]=.0033
   ET.SubElement(w,'geom',name=f'workpiece_{axis}_{sign}',type='box',size=nums(size),pos=nums(pos),rgba='.16 .27 .30 1')
 ET.SubElement(w,'geom',name='bench',type='box',size='.3 .25 .012',pos='0 0 .098',rgba='.36 .22 .12 1')
 # Thread silhouette is visual only; the equality models pitch and reaction.
 for k in range(9):ET.SubElement(screw,'geom',name=f'thread_ring_{k}',type='cylinder',size='.0034 .00015',pos=f'0 0 {-.001-k*.001}',mass='.000001',contype='0',conaffinity='0')
 return ET.tostring(root,encoding='unicode')

def fit():
 m=mujoco.MjModel.from_xml_string(build());d=mujoco.MjData(m);palm=m.body('palm').id
 ji=[m.joint(f'joint_{i}_{j}').id for i in range(1,6) for j in range(1,5)];qi=m.jnt_qposadr[ji]
 q0=np.array([.6,-1.,.8,.2,0,1.,.5,.2,0,1.,.5,.2,0,.7,.3,.2,.2,.6,.3,.2]);q0[12:]=json.loads((ROOT/'screwdriver_lab/artifacts/grasp.json').read_text())[12:];lo=m.jnt_range[ji,0]+1e-4;hi=m.jnt_range[ji,1]-1e-4;q0=np.clip(q0,lo,hi)
 targets=np.array([[0,0,.30212],[0,-.022,.278],[0,.022,.278]])
 def setstate(x):m.body_pos[palm]=x[20:];d.qpos[qi]=x[:20];mujoco.mj_forward(m,d)
 def residual(x):
  setstate(x);return np.r_[(d.site_xpos[:3]-targets).ravel()*200,.04*(x[:20]-q0),2*(x[20:]-[.10,0,.32])]
 low=np.r_[lo,[.065,-.04,.29]];high=np.r_[hi,[.14,.04,.36]]
 low[5]=.2;high[6]=1.2;high[7]=1.
 x=least_squares(residual,np.r_[q0,[.10,0,.32]],bounds=(low,high),max_nfev=250).x
 desired=[np.array([0,0,.29412]),np.array([0,-.012,.276]),np.array([0,.012,.276])]
 pairs=[(m.geom('finger_tip_1').id,m.geom('cap').id),(m.geom('finger_tip_2').id,m.geom('handle').id),(m.geom('finger_tip_3').id,m.geom('handle').id)]
 ref=x.copy()
 def surface(x):
  setstate(x);terms=[]
  for (a,b),target in zip(pairs,desired):
   line=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,a,b,.3,line);terms.extend([dist*1200]);terms.extend((line[3:]-target)*150)
  pen=sum(max(0,-c.dist-.00005)**2 for c in d.contact if m.geom_bodyid[c.geom1]!=m.body('screw').id and m.geom_bodyid[c.geom2]!=m.body('screw').id)
  return np.r_[terms,np.sqrt(pen)*3000,.05*(x[:20]-ref[:20]),(x[20:]-ref[20:])]
 x=least_squares(surface,x,bounds=(low,high),max_nfev=350,diff_step=1e-4).x;setstate(x)
 print('Fit residual',surface(x)[:9],flush=True)
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'precision_grasp.json').write_text(json.dumps(dict(q=x[:20].tolist(),palm=x[20:].tolist())))
 xml=build(palm_pos=x[20:]);(OUT/'preview.xml').write_text(xml);m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qpos[qi]=x[:20];mujoco.mj_forward(m,d)
 from screwdriver_lab.export_rtx import export
 export(xml,[d.qpos],OUT/'preview',dict(task='screw_in',frames=[dict(t=0,phase='reset')]))
 print('palm',x[20:],'contacts',[(m.geom(c.geom1).name,m.geom(c.geom2).name,round(c.dist*1000,3)) for c in d.contact],flush=True)
if __name__=='__main__':fit()
