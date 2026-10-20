"""Declared precision slotted driver / M3-like fastener geometry.

Dimensions are explicit design assumptions, not measurements of the hidden
reference tip or screw. Collision and rendering use the same surfaces. The
fastener is free and its thread is resolved through contact, without a pitch
constraint. Earlier records remain immutable.
"""
import json,hashlib,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np,trimesh,mujoco
from screwdriver_lab.hom.model import prepare,OUT
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'screwdriver_lab/artifacts/sequence_revision'

def nums(x):return ' '.join(format(float(v),'.15g')for v in np.asarray(x).ravel())
def mesh_geom(root,body,name,vertices,mass='0',color='.6 .64 .68 1',mask=None):
    mesh=trimesh.Trimesh(vertices=vertices,process=False).convex_hull
    content=mesh.export(file_type='obj',digits=17);sha=hashlib.sha256(content.encode()).hexdigest()[:16]
    dest=ART/'meshes'/(name+'-'+sha+'.obj');dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_text(content)
    E.SubElement(root.find('asset'),'mesh',name='seq_'+name,file=str(dest))
    attrs=dict(name=name,type='mesh',mesh='seq_'+name,mass=mass,rgba=color)
    if mask:attrs.update(contype=mask,conaffinity=mask)
    return E.SubElement(body,'geom',**attrs),mesh

def build(segments=48):
    ART.mkdir(parents=True,exist_ok=True);root=E.parse(prepare(True,48)).getroot();world=root.find('worldbody')
    tool=next(b for b in root.iter('body')if b.get('name')=='screwdriver');screw=next(b for b in root.iter('body')if b.get('name')=='screw');female=next(b for b in root.iter('body')if b.get('name')=='female_thread')
    for body in [tool,screw]:
        inert=body.find('inertial')
        if inert is not None:body.remove(inert)
    for g in list(tool.findall('geom')):
        if g.get('name')in ['handle','driver_bit','driver_bit_cross']:tool.remove(g)
    # One convex, gently tapered handle, maintaining the observed slender proportions.
    vertices=[]
    for z,r in [(-.035,.0038),(-.028,.0048),(-.015,.0052),(.015,.0052),(.029,.0049),(.035,.0043)]:
        vertices += [[r*np.cos(a),r*np.sin(a),z]for a in np.linspace(0,2*np.pi,64,endpoint=False)]
    handle,hm=mesh_geom(root,tool,'handle',vertices,mass='.009',color='.035 .04 .05 1')
    # Single slotted blade: 2.5 mm wide, 0.6 mm thick at the tip, tapered to shaft.
    vertices=[]
    for z,w,h in [(-.123,.00125,.0003),(-.120,.00125,.00045),(-.115,.001,.001)]:
        vertices += [[sx*w,sy*h,z]for sx in [-1,1]for sy in [-1,1]]
    bit,bm=mesh_geom(root,tool,'driver_bit',vertices,mass=str(bm_mass:=0.00018))
    # Remove the block cross recess and construct a true slot in a round head.
    for g in list(screw.findall('geom')):
        if g.get('name','').startswith(('slot_','male_thread_')):screw.remove(g)
    for g in list(female.findall('geom')):female.remove(g)
    for asset in list(root.find('asset')):
        if asset.get('name','').startswith(('male_thread_','female_thread_')):root.find('asset').remove(asset)
    core=screw.find("geom[@name='screw_core']");core.set('size','.00118 .0038');core.set('pos','0 0 -.0038')
    E.SubElement(screw,'geom',name='slot_floor',type='cylinder',size='.00275 .0004',pos='0 0 -.0004',mass='0')
    for sign in [-1,1]:
        vertices=[]
        for z,r in [(0,.00275),(.0009,.00275),(.0012,.00245)]:
            a0=np.arcsin(.0004/r)
            vertices += [[r*np.cos(a),sign*r*np.sin(a),z]for a in np.linspace(a0,np.pi-a0,33)]
        mesh_geom(root,screw,'slot_wall_'+str(sign),vertices)
    pitch=.0005
    # Symmetric 60-degree flanks, truncated crests/roots, and declared radial clearance.
    for body,prefix,turns,start,inner,outer,wi,wo,phase in [(screw,'male_thread',12,-.007,.00118,.0015,.00022,.000035,0),(female,'female_thread',6,-.006,.00124,.00156,.000035,.00022,pitch/2)]:
        for k in range(turns*segments):
            vertices=[]
            for a in [k*2*np.pi/segments,(k+1)*2*np.pi/segments]:
                for r,w in [(inner,wi),(outer,wo)]:
                    for sign in [-1,1]:vertices.append([r*np.cos(a),r*np.sin(a),start+pitch*a/(2*np.pi)+phase+sign*w])
            mesh_geom(root,body,prefix+'_'+str(k),vertices,mask='2')
    # Same 2 mm initial under-head clearance; fixture hole now matches M3 envelope.
    for geom in world.findall('geom'):
        if geom.get('name','').startswith('workpiece_'):
            axis=int(geom.get('name').split('_')[1]);pos=np.fromstring(geom.get('pos'),sep=' ');size=np.fromstring(geom.get('size'),sep=' ')
            pos[axis]=np.sign(pos[axis])*(size[axis]+.00165)
            if axis==1:size[0]=.00165
            geom.set('pos',nums(pos));geom.set('size',nums(size))
    # Mass properties from assembled head/core/thread surface volumes, steel 7800 kg/m³.
    # Thread wedge volumes are disjoint except their shared boundaries.
    parts=[]
    for g in screw.findall('geom'):
        if g.get('type')=='mesh':
            asset=next(x for x in root.findall('asset/mesh')if x.get('name')==g.get('mesh'));part=trimesh.load(asset.get('file'),force='mesh')
        else:
            radius,half=np.fromstring(g.get('size'),sep=' ');part=trimesh.creation.cylinder(radius=radius,height=2*half,sections=96);part.apply_translation(np.fromstring(g.get('pos','0 0 0'),sep=' '))
        part.density=7800;parts.append(part)
    mass=sum(p.mass for p in parts);com=sum(p.mass*p.center_mass for p in parts)/mass
    inertia=sum(p.moment_inertia+p.mass*((np.sum((p.center_mass-com)**2))*np.eye(3)-np.outer(p.center_mass-com,p.center_mass-com))for p in parts)
    E.SubElement(screw,'inertial',pos=nums(com),mass=str(mass),fullinertia=nums([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
    # Tool mass is explicit by component; compile once to freeze matching parser inertias.
    compiled=mujoco.MjModel.from_xml_string(E.tostring(root,encoding='unicode'));bid=compiled.body('screwdriver').id
    E.SubElement(tool,'inertial',pos=nums(compiled.body_ipos[bid]),mass=str(compiled.body_mass[bid]),quat=nums(compiled.body_iquat[bid]),diaginertia=nums(compiled.body_inertia[bid]))
    E.indent(root);path=ART/('precision-slotted.xml'if segments==48 else f'precision-slotted-s{segments}.xml');path.write_text(E.tostring(root,encoding='unicode'))
    manifest=dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dimensions_mm=dict(handle_length=70,handle_max_diameter=10.4,shaft_diameter=2,shaft_length=80,blade_width=2.5,blade_tip_thickness=.6,head_diameter=5.5,head_height=2,slot_width=.8,slot_depth=1.2,thread_major_diameter=3,pitch=.5),screw_mass_kg=mass,tool_mass_kg=float(compiled.body_mass[bid]),assumptions='Declared dimensionally matched slotted precision tool and M3-like fastener; not a measured replica or ISO-certified geometry',collision_matches_visual=True,thread_segments=segments)
    (ART/('geometry.json'if segments==48 else f'geometry-s{segments}.json')).write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest));return path
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--segments',type=int,default=48);a=p.parse_args();build(a.segments)
