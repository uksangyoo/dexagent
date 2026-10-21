"""Geometry preparation for independent Drake hydroelastic validation.

Original recorded experiments are immutable. Mesh format conversion preserves
vertices/faces; physical dimensions and materials are explicit in a manifest.
"""
from pathlib import Path
import hashlib,json,os,xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/hom_revision'
# Frozen inputs recorded by the original physical-task pipeline (Oct 15).
SOURCE=Path(__file__).resolve().parent/'data/alternating-seat-nominal-34.xml'
PREFIX=SOURCE.with_name('matched-prefix-y0.npz')

def source_model():
    return mujoco.MjModel.from_xml_string(SOURCE.read_text().replace('/home/uyoo/Projects/dexagent/',str(ROOT)+'/'))

def prepare(slim=False, thread_segments=24):
    OUT.mkdir(parents=True,exist_ok=True);meshes=OUT/'meshes';meshes.mkdir(exist_ok=True)
    root=ET.fromstring(SOURCE.read_text().replace('/home/uyoo/Projects/dexagent/',str(ROOT)+'/'));compiler=root.find('compiler');meshdir=Path(compiler.get('meshdir'))
    converted=[]
    for asset in root.findall('asset/mesh'):
        if not asset.get('file'):
            vertices=np.fromstring(asset.get('vertex',''),sep=' ').reshape(-1,3)
            faces=np.fromstring(asset.get('face',''),sep=' ',dtype=int).reshape(-1,3)
            dest=meshes/(asset.get('name')+'.obj')
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            if not len(faces):mesh=mesh.convex_hull  # MuJoCo's vertex-only mesh semantics.
            content=mesh.export(file_type='obj',digits=17)
            if not dest.exists()or dest.read_text()!=content:
                temporary=dest.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(content);temporary.replace(dest)
            asset.set('file',str(dest));asset.attrib.pop('vertex',None);asset.attrib.pop('face',None)
            continue
        source=Path(asset.get('file'));source=source if source.is_absolute()else meshdir/source
        if source.suffix.lower()=='.obj':continue
        dest=meshes/(hashlib.sha256(str(source).encode()).hexdigest()[:10]+'_'+source.stem+'.obj')
        if not dest.exists():
            mesh=trimesh.load(source,force='mesh');mesh.export(dest)
        asset.set('file',str(dest));converted.append(dict(source=str(source),target=str(dest)))
    if thread_segments != 24:
        from dexagent.sim.thread_geometry import add_helix, PITCH
        for asset in list(root.find('asset')):
            if asset.get('name','').startswith(('male_thread_','female_thread_')):root.find('asset').remove(asset)
        for body in root.iter('body'):
            for g in list(body.findall('geom')):
                if g.get('name','').startswith(('male_thread_','female_thread_')):body.remove(g)
        screw=next(b for b in root.iter('body')if b.get('name')=='screw')
        female=next(b for b in root.iter('body')if b.get('name')=='female_thread')
        start=len(root.find('asset'))
        add_helix(root.find('asset'),screw,'male_thread',12,-.014,.0027,.00325,.00042,.00010,thread_segments)
        add_helix(root.find('asset'),female,'female_thread',6,-.012,.00285,.0034,.00010,.00042,thread_segments,PITCH/2)
        for asset in list(root.find('asset'))[start:]:
            mesh=trimesh.Trimesh(vertices=np.fromstring(asset.get('vertex'),sep=' ').reshape(-1,3),process=False).convex_hull
            dest=meshes/f's{thread_segments}-{asset.get("name")}.obj'
            if not dest.exists():
                temporary=dest.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(mesh.export(file_type='obj',digits=17));temporary.replace(dest)
            asset.set('file',str(dest));asset.attrib.pop('vertex',None)
    # The former wrist capsule extended past the physical flange and into
    # the mounted thumb base. Use the supplied wrist CAD convex envelope;
    # preserve the original rigid-body inertia and all collision masks.
    wrist=next(g for g in root.iter('geom')if g.get('name')=='arm_collision_27')
    wrist.set('type','mesh');wrist.set('mesh','wrist3')
    for key in ['pos','quat','size']:wrist.attrib.pop(key,None)
    # Direct effort actuators: controller owns the shared bounded motor loop.
    for act in root.find('actuator'):
        act.tag='motor'
        for key in ['kp','kv']:act.attrib.pop(key,None)
    # This passive bearing has no motor rotor. Its inertia is the rigid cap
    # inertia. Make that explicit instead of leaving unsupported MJCF armature
    # on an unactuated joint, which Drake otherwise silently omits.
    next(j for j in root.iter('joint')if j.get('name')=='cap_bearing').set('armature','0')
    # A true free rigid body avoids parser-dependent compound-joint ordering.
    world=root.find('worldbody');slide=world.find("body[@name='screw_slide']")
    screw=slide.find("body[@name='screw']");slide.remove(screw);screw.set('pos',slide.get('pos'))
    for joint in list(screw.findall('joint')):screw.remove(joint)
    ET.SubElement(screw,'freejoint',name='screw_free');world.remove(slide);world.append(screw)
    if slim:
        geom=lambda name:next(g for g in root.iter('geom')if g.get('name')==name)
        geom('handle').set('size','.005 .035');geom('handle').set('mass','.009')
        geom('shaft').set('size','.001 .040');geom('shaft').set('mass','.002')
        geom('driver_bit').set('size','.0015 .0005 .004');geom('driver_bit').set('mass','.0002')
        geom('cap').set('size','.006 .004');geom('cap').set('mass','.001')
        geom('stripe').set('pos','.005 0 0')
    for i,geom in enumerate(root.find("worldbody").iter("geom")):
        if not geom.get("name"):geom.set("name",f"source_geometry_{i}")
    # Preserve authoritative compiled mass properties across parser differences.
    compiled=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    for body in root.iter('body'):
        bid=compiled.body(body.get('name')).id
        if compiled.body_mass[bid]<=0:continue
        existing=body.find('inertial')
        if existing is not None:body.remove(existing)
        nums=lambda x:' '.join(format(float(v),'.17g')for v in x)
        ET.SubElement(body,'inertial',mass=str(compiled.body_mass[bid]),pos=nums(compiled.body_ipos[bid]),quat=nums(compiled.body_iquat[bid]),diaginertia=nums(compiled.body_inertia[bid]))
    for g in root.iter('geom'):
        if g.get('friction'):g.set('friction',g.get('friction').split()[0]+' 0 0')
    path=OUT/(('slim'if slim else'conversion')+(f'-s{thread_segments}'if thread_segments!=24 else'')+'.xml');ET.indent(root)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(ET.tostring(root,encoding='unicode'));temporary.replace(path)
    (OUT/'conversion.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),meshes=converted),indent=2))
    return path
def reached_state(xml):
    old=source_model();d=mujoco.MjData(old)
    d.qpos[:]=np.load(PREFIX)['qpos'][-1];mujoco.mj_forward(old,d)
    new=mujoco.MjModel.from_xml_path(str(xml));state=mujoco.MjData(new)
    for j in range(new.njnt):
        a=new.jnt_qposadr[j]
        if new.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            b=old.body(new.body(int(new.jnt_bodyid[j])).name).id
            state.qpos[a:a+3]=d.xpos[b];mujoco.mju_mat2Quat(state.qpos[a+3:a+7],d.xmat[b])
        else:state.qpos[a]=d.qpos[old.joint(new.joint(j).name).qposadr[0]]
    return state.qpos.copy()

if __name__=='__main__':print(prepare())

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
