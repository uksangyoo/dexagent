"""Real Tesollo mesh model and an unactuated screwdriver; SI units throughout."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CENTER = np.array([.062, .012, .145])
COLORS = ['.36 .78 .71 1', '.43 .64 .94 1', '.73 .56 .88 1', '.95 .64 .37 1', '.88 .78 .49 1']

def nums(x): return ' '.join(str(float(v)) for v in x)

def build_scene(radius=.018, friction=.8, offset_mm=0., yaw_deg=0., fixture=False):
    root = ET.parse(ROOT/'third-party/tesollo_model/dg5fs/dg5fs_right.xml').getroot()
    root.set('model','tesollo_screwdriver')
    root.find('compiler').set('meshdir',str(ROOT/'third-party/tesollo_model/dg5fs/meshes'))
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast',cone='elliptic',iterations='60',gravity='0 0 -9.81')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction=f'{friction} .002 .0001',solref='.008 1',solimp='.95 .99 .001',condim='3')
    ET.SubElement(default,'joint',damping='.04',armature='.002')
    world=root.find('worldbody')
    mount=ET.Element('body',name='palm',pos='0 0 .16',quat='1 0 0 0')
    for e in list(world): world.remove(e); mount.append(e)
    world.append(mount)
    actuator=ET.SubElement(root,'actuator')
    for j in mount.findall('.//joint'):
        ET.SubElement(actuator,'position',name='servo_'+j.get('name'),joint=j.get('name'),kp='1.5',kv='.06',forcerange='-.3 .3')
    for i in range(1,6):
        b=mount.find(f".//body[@name='link_{i}_4']")
        tip=b.find(f"geom[@mesh='link_{i}_tip']")
        ET.SubElement(b,'site',name=f'tip_{i}',pos=tip.get('pos'),size='.002',rgba='1 1 1 0')
        for body in mount.findall('.//body'):
            if body.get('name','').startswith(f'link_{i}_'):
                for g in body.findall('geom'): g.set('rgba',COLORS[i-1])
    contact=ET.SubElement(root,'contact')
    for i in range(1,6): ET.SubElement(contact,'exclude',body1='palm',body2=f'link_{i}_1')
    ET.SubElement(contact,'exclude',body1='palm',body2='link_1_2')
    R=np.eye(3)
    center=CENTER+np.array([0,offset_mm/1000,0])
    pos=R@center+np.array([0,0,.16])
    tool=ET.SubElement(world,'body',name='screwdriver',pos=nums(pos),quat=nums(Rotation.from_euler('z',yaw_deg,degrees=True).as_quat(scalar_first=True)))
    if fixture: ET.SubElement(tool,'joint',name='tool_spin',type='hinge',axis='0 0 1',limited='false',damping='.0001')
    else: ET.SubElement(tool,'freejoint',name='tool_free')
    ET.SubElement(tool,'geom',name='handle',type='cylinder',size=f'{radius} .042',mass='.065',rgba='.13 .16 .19 1')
    ET.SubElement(tool,'geom',name='cap',type='cylinder',pos='0 0 .045',size=f'{radius*1.08} .005',mass='.008',rgba='.94 .27 .23 1')
    ET.SubElement(tool,'geom',name='shaft',type='cylinder',pos='0 0 -.090',size='.002 .048',mass='.012',rgba='.66 .72 .78 1')
    ET.SubElement(tool,'geom',name='stripe',type='box',pos=f'{radius} 0 0',size='.0007 .002 .034',mass='.0001',contype='0',conaffinity='0',rgba='.98 .71 .25 1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .01',rgba='.13 .17 .21 1')
    ET.SubElement(world,'light',pos='.2 -.3 .9',dir='-.2 .3 -.9',diffuse='.8 .8 .8')
    ET.SubElement(world,'light',pos='-.5 .2 .5',diffuse='.4 .4 .4')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.5 .5 .5')
    return ET.tostring(root,encoding='unicode')

def grasp(model):
    """Fit a nominal grasp once; test conditions reuse it without refitting."""
    d=mujoco.MjData(model)
    lo,hi=model.jnt_range[:20,0]+1e-5,model.jnt_range[:20,1]-1e-5
    q0=np.clip(np.array([.4,-.9,.2,.2,0,.7,.3,.2,0,.7,.3,.2,0,.7,.3,.2,.2,.6,.3,.2]),lo,hi)
    targets=np.array([[.063,.035,.113],[.063,.031,.148],[.063,-.009,.145],[.056,-.030,.129],[.046,-.043,.113]])
    rot=np.eye(3)
    targets=targets@rot.T+np.array([0,0,.16])
    def residual(q):
        d.qpos[:20]=q; mujoco.mj_forward(model,d)
        return np.r_[(d.site_xpos[:5]-targets).ravel()*100,.02*(q-q0)]
    q=least_squares(residual,q0,bounds=(lo,hi),max_nfev=150).x
    # Adjust three opposed fingertips to actual mesh surface contact, not tip-site overlap.
    handle=model.geom('handle').id
    ids=[model.body(f'link_{i}_4').id for i in [1,2,3]]
    geoms=[np.where(model.geom_bodyid==b)[0] for b in ids]
    qref=q.copy()
    def surface_residual(x):
        d.qpos[:20]=x; mujoco.mj_forward(model,d)
        distances=[min(mujoco.mj_geomDistance(model,d,int(g),handle,.2,None) for g in gs) for gs in geoms]
        penetration=[]
        for c in d.contact:
            if c.dist<-.0001: penetration.append(max(0,-c.dist-.0001))
        # Fixed residual size via total squared penetration.
        return np.r_[np.array(distances)*1000,np.sqrt(sum(p*p for p in penetration))*2000,.05*(x-qref)]
    fit=least_squares(surface_residual,q,bounds=(lo,hi),max_nfev=180,diff_step=1e-4)
    return fit.x

if __name__=='__main__':
    import json
    out=ROOT/'screwdriver_lab/artifacts';out.mkdir(exist_ok=True)
    xml=build_scene();m=mujoco.MjModel.from_xml_string(xml);q=grasp(m)
    (out/'scene.xml').write_text(xml);(out/'grasp.json').write_text(json.dumps(q.tolist()))
    d=mujoco.MjData(m);d.qpos[:20]=q;mujoco.mj_forward(m,d)
    print('q',q.tolist()); print('contacts',[(m.geom(c.geom1).name,m.geom(c.geom2).name,c.dist) for c in d.contact])
    cam=mujoco.MjvCamera();cam.lookat[:]=[.035,0,.26];cam.distance=.46;cam.azimuth=35;cam.elevation=-15
    renderer=mujoco.Renderer(m,480,640);renderer.update_scene(d,cam)
    from PIL import Image
    Image.fromarray(renderer.render()).save(out/'grasp.jpg');renderer.close()
