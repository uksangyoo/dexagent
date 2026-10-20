"""Open-hand pickup, reorientation and two-turn screw insertion scene."""
import json,xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from screwdriver_lab.screw_scene import ROOT,CENTER,build,nums
OUT=ROOT/'screwdriver_lab/artifacts/full_motion'
PICK=np.array([-.12,.05,.23012]);TRAVEL=.002

def build_full(case=None):
 g=json.loads((ROOT/'screwdriver_lab/artifacts/screw_in/precision_grasp.json').read_text());p0=np.array(g['palm']);root=E.fromstring(build(case,p0));root.set('model','full_screw_motion');root.find('default/geom').set('condim','6');root.find('default/geom').set('friction',f"{(case or {}).get('friction',.9)} .006 .00002");world=root.find('worldbody');palm=world.find("body[@name='palm']");world.remove(palm);palm.remove(palm.find("joint[@name='feed']"));palm.set('pos','0 0 0')
 parent=world
 for name,axis in [('feed','0 0 -1'),('cart_x','1 0 0'),('cart_y','0 1 0'),('wrist_tilt','0 1 0')]:
  body=E.SubElement(parent,'body',name=name+'_stage',pos=nums(p0) if name=='feed' else '0 0 0',gravcomp='1');E.SubElement(body,'inertial',pos='0 0 0',mass='.1',diaginertia='.001 .001 .001')
  E.SubElement(body,'joint',name=name,type='hinge' if name=='wrist_tilt' else 'slide',axis=axis,range='-1.8 1.8' if name=='wrist_tilt' else '-.4 .4',damping='1',armature='.01');parent=body
 parent.append(palm);act=root.find('actuator');feed=act.find("position[@joint='feed']");feed.set('kp','10000');feed.set('kv','160');feed.set('forcerange','-200 200')
 for name in ['cart_x','cart_y','wrist_tilt']:E.SubElement(act,'position',name=name+'_servo',joint=name,kp='50' if name=='wrist_tilt' else '10000',kv='5' if name=='wrist_tilt' else '160',forcerange='-10 10' if name=='wrist_tilt' else '-200 200')
 
 for actuator in list(act)[:20]:actuator.set('kp','8');actuator.set('kv','.18')
 # A cruciform bit/socket supplies lateral guidance in both directions through contact.
 tool=world.find("body[@name='screwdriver']")
 E.SubElement(tool,'geom',name='driver_bit_cross',type='box',size='.00075 .003 .004',pos='0 0 -.119',mass='.0005',rgba='.65 .7 .75 1')
 screw=world.find("body[@name='screw']")
 for geom in list(screw.findall('geom')):
  if geom.get('name','').startswith('slot_wall'):screw.remove(geom)
 for x in [-1,1]:
  for y in [-1,1]:E.SubElement(screw,'geom',name=f'slot_wall_{x}_{y}',type='box',size='.0018 .0018 .0015',pos=nums([x*.0028,y*.0028,.0015]),mass='.0005')
 screw.find("geom[@name='slot_floor']").set('size','.0065 .0005')
 world.remove(screw);joint=screw.find("joint[@name='screw_depth']");screw.remove(joint);joint.set('range',f'-.0001 {TRAVEL}')
 screw.find("joint[@name='screw_angle']").set('solreffriction','.002 1');screw.find("joint[@name='screw_angle']").set('solimpfriction','.9999 .9999 .0001')
 slide=E.SubElement(world,'body',name='screw_slide',pos=screw.get('pos'));slide.append(joint);E.SubElement(slide,'inertial',pos='0 0 0',mass='.00001',diaginertia='1e-9 1e-9 1e-9');screw.set('pos','0 0 0');slide.append(screw)
 for geom in world.findall('geom'):
  if geom.get('name','').startswith('workpiece') or geom.get('name')=='bench':pos=np.fromstring(geom.get('pos'),sep=' ');pos[2]-=.001;geom.set('pos',nums(pos))
 # Two narrow supports leave the handle clear for opposed fingertips.
 for name,x,z,width in [('cap',-.042,.21682,.004),('shaft',.070,.22822,.004)]:
  bottom=.11
  E.SubElement(world,'geom',name='cradle_'+name,type='box',pos=nums([PICK[0]+x,PICK[1],(z+bottom)/2]),size=nums([.001,width,(z-bottom)/2]),rgba='.28 .3 .32 1')
 return E.tostring(root,encoding='unicode'),g
