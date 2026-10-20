"""UR5e articulated dynamics with the supplied Tesollo hand rigidly mounted."""
from copy import deepcopy
import xml.etree.ElementTree as E
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from screwdriver_lab.screw_scene import ROOT,R,nums
ARM_NAMES=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']

def attach_arm(xml):
 root=E.fromstring(xml);w=root.find('worldbody');stage=w.find("body[@name='feed_stage']");palm=stage.find('.//body[@name="palm"]');w.remove(stage)
 arm_dir=ROOT/'third-party/universal_robots_ur5e';arm=E.parse(arm_dir/'ur5e.xml').getroot()
 for e in arm.find('asset'):
  if e.tag=='mesh':e.set('file',str(arm_dir/'assets'/e.get('file')))
  root.find('asset').append(deepcopy(e))
 root.find('default').append(deepcopy(arm.find('default/default')))
 base=deepcopy(arm.find('worldbody/body'));base.set('pos','-.52 .25 .11');w.append(base)
 for b in base.iter('body'):b.set('gravcomp','1')
 wrist=base.find('.//body[@name="wrist_3_link"]');site=wrist.find('site');palm.set('pos',site.get('pos'));palm.set('quat',site.get('quat'));palm.set('childclass','hand_defaults')
 E.SubElement(root.find('default'),'default',{'class':'hand_defaults'})
 wrist.append(palm)
 for j in base.findall('.//joint'):
  if j.get('name') in ARM_NAMES:j.set('armature','.02');j.set('damping','1')
 act=root.find('actuator')
 for e in list(act)[20:]:act.remove(e)
 for i,name in enumerate(ARM_NAMES):E.SubElement(act,'position',name='arm_'+name,joint=name,kp='3000' if i<3 else '800',kv='120' if i<3 else '35',forcerange='-150 150' if i<3 else '-28 28')
 E.SubElement(root.find('contact'),'exclude',body1='wrist_3_link',body2='palm')
 # Visual meshes retain exact source geometry; collision capsules remain active.
 for k,g in enumerate(base.findall('.//geom')):
  if g.get('class') in ['collision','eef_collision']:g.set('name','arm_collision_'+str(k))
 return E.tostring(root,encoding='unicode')

class ArmIK:
 def __init__(self,m):
  self.m=m;self.d=mujoco.MjData(m);self.bid=m.body('palm').id;self.jids=np.array([m.joint(n).id for n in ARM_NAMES]);self.qi=m.jnt_qposadr[self.jids];self.vi=m.jnt_dofadr[self.jids];self.last=np.array([-1.57,-1.57,1.57,-1.57,-1.57,0.])
 def solve(self,pos,rot,seed=None,exact=False):
  q=self.last.copy() if seed is None else np.array(seed).copy()
  if exact:
   def fun(q):
    self.d.qpos[self.qi]=q;mujoco.mj_forward(self.m,self.d)
    return np.r_[(self.d.xpos[self.bid]-pos)*5,Rot.from_matrix(rot@self.d.xmat[self.bid].reshape(3,3).T).as_rotvec()]
   best=None
   for seed0 in [q,np.array([0,-1.2,1.8,-2.,-1.57,0]),np.array([-2.,-1.,1.5,-2.,1.57,0])]:
    f=least_squares(fun,seed0,bounds=(self.m.jnt_range[self.jids,0]+1e-4,self.m.jnt_range[self.jids,1]-1e-4),max_nfev=150)
    if best is None or np.linalg.norm(f.fun)<best[0]:best=(np.linalg.norm(f.fun),f.x.copy())
   q=best[1]
  else:
   for _ in range(4):
    self.d.qpos[self.qi]=q;mujoco.mj_forward(self.m,self.d);Jp=np.zeros((3,self.m.nv));Jr=Jp.copy();mujoco.mj_jacBody(self.m,self.d,Jp,Jr,self.bid)
    J=np.vstack([Jp[:,self.vi]*5,Jr[:,self.vi]]);err=np.r_[(pos-self.d.xpos[self.bid])*5,Rot.from_matrix(rot@self.d.xmat[self.bid].reshape(3,3).T).as_rotvec()]
    q+=np.clip(J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),err),-.04,.04);q=np.clip(q,self.m.jnt_range[self.jids,0]+1e-4,self.m.jnt_range[self.jids,1]-1e-4)
  self.last=q.copy();return q
