"""Explicit, mating trapezoidal thread surfaces, with no pitch constraint.

This is a declared 6.5 mm diameter / 1 mm pitch test fastener, not a scan of
the hidden screw in the human video. Convex sectors approximate each helix.
The screw has all six rigid-body degrees of freedom; only surface contacts
with the fixed female thread provide guidance and axial reaction.
"""
import xml.etree.ElementTree as E
import numpy as np

PITCH = .001

def nums(v):
    return ' '.join(f'{x:.10g}' for x in np.asarray(v).ravel())

def add_helix(asset, body, prefix, turns, z_start, inner, outer,
              inner_halfwidth, outer_halfwidth, segments=32, phase=0):
    for k in range(round(turns*segments)):
        vertices=[]
        for a in [k*2*np.pi/segments, (k+1)*2*np.pi/segments]:
            for r, halfwidth in [(inner,inner_halfwidth),(outer,outer_halfwidth)]:
                for sign in [-1,1]:
                    vertices.append([r*np.cos(a),r*np.sin(a),
                                     z_start+PITCH*a/(2*np.pi)+phase+sign*halfwidth])
        name=f'{prefix}_{k}'
        E.SubElement(asset,'mesh',name=name,vertex=nums(vertices))
        E.SubElement(body,'geom',name=name,type='mesh',mesh=name,
                     mass='0',friction='.15 0 0',condim='3',
                     solref='.001 1',solimp='.99 .99 .0001',
                     contype='2',conaffinity='2',rgba='.55 .57 .6 1')

def replace_thread(root, segments=32):
    world=root.find('worldbody');asset=root.find('asset')
    if asset is None:asset=E.SubElement(root,'asset')
    equality=root.find('equality')
    if equality is not None:
        for element in list(equality):
            if element.get('name')=='helical_thread':equality.remove(element)
    slide=world.find("body[@name='screw_slide']")
    screw=slide.find("body[@name='screw']")
    origin=np.fromstring(slide.get('pos'),sep=' ')
    # Four unconstrained coordinates plus the existing depth and angle give
    # the screw six DOFs. There is no radial/tilt guide, equality, or actuator.
    for name,kind,axis in [('screw_x','slide','1 0 0'),('screw_y','slide','0 1 0'),
                           ('screw_tilt_x','hinge','1 0 0'),('screw_tilt_y','hinge','0 1 0')]:
        E.SubElement(slide,'joint',name=name,type=kind,axis=axis,
                     limited='false',damping='0',armature='0',frictionloss='0')
    for joint in [slide.find("joint[@name='screw_depth']"),screw.find("joint[@name='screw_angle']")]:
        joint.set('limited','false');joint.set('damping','0');joint.set('armature','0');joint.set('frictionloss','0')
    for geom in list(screw.findall('geom')):
        if geom.get('name','').startswith('thread_ring'):screw.remove(geom)
    core=screw.find("geom[@name='screw_core']");core.set('size','.0027 .008')
    core.set('contype','2');core.set('conaffinity','2')
    # Preserve the declared fastener mass and inertia, independent of mesh
    # tessellation. Threads themselves are included in this 8 g assembly.
    for geom in screw.findall('geom'):geom.set('mass','0')
    E.SubElement(screw,'inertial',pos='0 0 -.004',mass='.008',
                 diaginertia='2.1e-7 2.1e-7 6e-8')
    add_helix(asset,screw,'male_thread',12,-.014,.0027,.00325,.00042,.00010,segments)
    nut=E.SubElement(world,'body',name='female_thread',pos=nums(origin))
    add_helix(asset,nut,'female_thread',6,-.012,.00285,.0034,.00010,.00042,segments,PITCH/2)
    # Move the existing workpiece hole outside the female thread root.
    for geom in world.findall('geom'):
        if geom.get('name','').startswith('workpiece_'):
            axis=int(geom.get('name').split('_')[1]);pos=np.fromstring(geom.get('pos'),sep=' ')
            size=np.fromstring(geom.get('size'),sep=' ')
            pos[axis]=np.sign(pos[axis])*(size[axis]+.0035)
            if axis==1:size[0]=.0035
            geom.set('pos',nums(pos));geom.set('size',nums(size))
    return root

def bench_xml(segments=32):
    root=E.fromstring('''<mujoco><option timestep=".00025" gravity="0 0 -9.81"
      integrator="implicitfast" cone="elliptic" iterations="100"/><asset/>
      <worldbody><body name="screw_slide" pos="0 0 .02">
      <inertial pos="0 0 0" mass=".00001" diaginertia="1e-10 1e-10 1e-10"/>
      <joint name="screw_depth" type="slide" axis="0 0 -1"/>
      <body name="screw"><joint name="screw_angle" type="hinge" axis="0 0 -1"/>
      <geom name="screw_core" type="cylinder" size=".003 .008" pos="0 0 -.008"/>
      </body></body></worldbody></mujoco>''')
    replace_thread(root,segments)
    return E.tostring(root,encoding='unicode')

if __name__=='__main__':
    import argparse,json,mujoco
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--segments',type=int,default=24)
    p.add_argument('--duration',type=float,default=2);p.add_argument('--torque',type=float,default=.0002)
    p.add_argument('--dt',type=float,default=.00025)
    p.add_argument('--speed',type=float);p.add_argument('--load',type=float,default=0);a=p.parse_args()
    xml=bench_xml(a.segments);m=mujoco.MjModel.from_xml_string(xml);m.opt.timestep=a.dt;d=mujoco.MjData(m)
    rows=[];v=m.joint('screw_angle').dofadr[0]
    for k in range(round(a.duration/a.dt)):
        d.qfrc_applied[:]=0
        # External wrench ONLY in this independent mechanics test.
        d.qfrc_applied[v]=(a.torque if a.speed is None else np.clip(.0001*(a.speed-d.qvel[v]),-a.torque,a.torque)) if d.time>.2 else 0
        d.qfrc_applied[m.joint('screw_depth').dofadr[0]]=a.load
        mujoco.mj_step(m,d)
        if k%20==0:rows.append([d.time,*d.qpos,d.ncon,min([c.dist for c in d.contact]+[0])])
    out=Path(__file__).parent/'artifacts/physical_task';out.mkdir(exist_ok=True)
    name=f'thread-bench-s{a.segments}-dt{a.dt}-tau{a.torque}-speed{a.speed}-load{a.load}'
    (out/(name+'.xml')).write_text(xml);np.savez_compressed(out/(name+'.npz'),trace=rows)
    result=dict(parameters=vars(a),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),
                contacts=d.ncon,pitch_error_mm=float((d.qpos[0]-PITCH*d.qpos[-1]/(2*np.pi))*1000))
    (out/(name+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
