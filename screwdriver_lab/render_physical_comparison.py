"""RTX hardware path tracing through Blender Cycles + NVIDIA OptiX.

Run with a bpy-enabled Python; CUDA_VISIBLE_DEVICES selects the owned GPU.
No pose correction, mesh scaling, or simulated dynamics occurs here.
"""
import argparse,hashlib,json,math,os,subprocess,time,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
import bpy,bmesh,numpy as np
from mathutils import Vector,Quaternion
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=24);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=960);p.add_argument('--fps',type=int,default=30);p.add_argument('--still',action='store_true');p.add_argument('--view',choices=['hero','side','front'],default='hero');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);p.add_argument('--no-end-card',action='store_true');a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
info=json.loads((a.bundle/'scene.json').read_text());arr=np.load(a.bundle/'scene.npz');transforms=arr['transforms'][a.start:a.stop];info['metadata']['frames']=info['metadata']['frames'][a.start:a.stop]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='GPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=True;scene.cycles.denoiser='OPTIX';scene.cycles.use_adaptive_sampling=True;scene.cycles.adaptive_threshold=.03;scene.cycles.max_bounces=6
prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
for d in prefs.devices:d.use=d.type=='OPTIX'
active=[dict(name=d.name,type=d.type,id=d.id) for d in prefs.devices if d.use]
if not active or any('61:00'not in d['id']for d in active):raise RuntimeError('Only physical GPU 1 (61:00) may render')
print('RTX_DEVICES',json.dumps(active),flush=True)
scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.film_transparent=False;scene.render.image_settings.file_format='PNG' if a.still else 'JPEG';scene.render.image_settings.quality=95;scene.render.image_settings.color_mode='RGB';scene.render.use_persistent_data=True
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.15
scene.world=bpy.data.worlds.new('Studio');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.56,.62,.69,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35

def material(name,color,metal=.0,rough=.4):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes.get('Principled BSDF');n.inputs['Base Color'].default_value=(*color,1);n.inputs['Metallic'].default_value=metal;n.inputs['Roughness'].default_value=rough;return m
metal=material('Satin aluminum',(.54,.59,.64),.72,.30);base=material('Graphite chassis',(.085,.105,.12),.65,.32);rubber=material('Fingertip elastomer',(.032,.043,.052),0,.62)
handle=material('Screwdriver grip',(.023,.03,.037),0,.36);cap=material('Red polymer cap',(.55,.025,.018),0,.3);steel=material('Steel shaft',(.58,.65,.72),.95,.17);stripe=material('Rotation witness mark',(.92,.50,.075),.1,.33)
work=material('Teal fixture',(.055,.19,.21),.5,.32);wood=material('Workbench',(.21,.115,.055),0,.6)
geom_objects={}
for g in info['geometry']:
    name=g['name'];typ=g['type'];size=g['size']
    if name.startswith('arm_collision_'):continue
    if typ==7:
        mid=g['mesh'];mesh=bpy.data.meshes.new(name);mesh.from_pydata(arr[f'verts_{mid}'].tolist(),[],arr[f'faces_{mid}'].tolist());mesh.update();ob=bpy.data.objects.new(name,mesh);scene.collection.objects.link(ob)
        is_tip=name.startswith('finger_tip_')
        mat=base if g['body']=='palm' else rubber if is_tip else metal
        if g['body'] in ['base','shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']:mat=material(name+' finish',tuple(g['rgba'][:3]),.35,.45)
    elif typ==5:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=size[0],depth=2*size[1]);ob=bpy.context.object;mat=handle if name=='handle' else cap if name=='cap' else steel
        bevel=ob.modifiers.new('Machined edge bevel','BEVEL');bevel.width=.00025;bevel.segments=3
    elif typ==6:
        bpy.ops.mesh.primitive_cube_add(size=1);ob=bpy.context.object;ob.scale=tuple(2*x for x in size);mat=work if name.startswith('workpiece') else wood if name in ['bench','pickup_surface','pickup_cap','pickup_shaft'] else base if name.startswith('cradle') else stripe if name=='stripe' else steel
    else:raise RuntimeError(f'Unsupported visual geom {typ}')
    ob.name=name;ob.data.materials.append(mat);ob.rotation_mode='QUATERNION';geom_objects[g['id']]=ob
    for poly in ob.data.polygons:poly.use_smooth=typ in [5,7]
    if typ==7:
        bm=bmesh.new();bm.from_mesh(ob.data)
        for edge in bm.edges:edge.smooth=not(edge.is_manifold and edge.calc_face_angle()>math.radians(35))
        bm.to_mesh(ob.data);bm.free()
        normal=ob.modifiers.new('Weighted surface normals','WEIGHTED_NORMAL');normal.keep_sharp=True;normal.weight=50
# Join rigid thread sectors for rendering only. Relative geometry is preserved;
# all sectors in a group share one simulated body and one rigid transform.
thread_groups={}
for g in info['geometry']:
    if g['name'].startswith(('male_thread_','female_thread_')):
        thread_groups.setdefault(g['body'],[]).append(g['id'])
merged_sectors=0
for body,ids in thread_groups.items():
    if len(ids)<2:continue
    bpy.ops.object.select_all(action='DESELECT')
    for gid in ids:
        ob=geom_objects[gid];ob.location=transforms[0,gid,:3];ob.rotation_quaternion=Quaternion(transforms[0,gid,3:]);ob.select_set(True)
    primary=ids[0];bpy.context.view_layer.objects.active=geom_objects[primary]
    bpy.ops.object.join();merged_sectors+=len(ids)
    for gid in ids[1:]:del geom_objects[gid]
print('RTX_RIGID_THREAD_MERGE',merged_sectors,len(thread_groups),flush=True)
# Display-only witness flag, rigidly parented to the recorded handle transform.
handle_id=next(g['id']for g in info['geometry']if g['name']=='handle')
bpy.ops.mesh.primitive_cube_add(size=1);witness=bpy.context.object;witness.name='visual_rotation_witness';witness.parent=geom_objects[handle_id];witness.location=(.019,0,-.028);witness.scale=(.014,.0003,.006);witness.data.materials.append(material('Purple visual witness',(.40,.08,.65),0,.5))
bpy.ops.mesh.primitive_plane_add(size=200);bpy.context.object.data.materials.append(material('Warm studio surface',(.48,.52,.50),0,.65))
# Display-only support under the fixed arm base; never modifies robot poses.
bpy.ops.mesh.primitive_cube_add(size=1,location=(-.52,.25,.055));bpy.context.object.scale=(.19,.19,.11);bpy.context.object.data.materials.append(base)

def light(name,pos,power,size,color):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size;data.color=color
    ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob);ob.location=pos;ob.rotation_euler=(Vector((.035,.01,.27))-ob.location).to_track_quat('-Z','Y').to_euler()
light('Large soft key',(.3,-.3,1.5),30,1.,(1,.91,.80));light('Cool fill',(.1,.75,1.),12,.8,(.77,.86,1));light('Rim',(-.8,-.05,1.2),15,.8,(1,.97,.91))
camdata=bpy.data.cameras.new('Inspection camera');cam=bpy.data.objects.new('Inspection camera',camdata);scene.collection.objects.link(cam);scene.camera=cam
cam.location={'hero':(-.35,-.55,.95),'side':(-.65,.15,.52),'front':(.05,-.8,.5)}[a.view];target=Vector((.0,.0,.345));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();camdata.type='ORTHO';camdata.ortho_scale=.32;camdata.lens=55
close_data=bpy.data.cameras.new('Bit and screw inspection');close=bpy.data.objects.new('Bit and screw inspection',close_data);scene.collection.objects.link(close)
close.location=(.035,-.05,.282);close.rotation_euler=(Vector((0,0,.247))-close.location).to_track_quat('-Z','Y').to_euler();close_data.type='ORTHO';close_data.ortho_scale=.025;close_data.clip_start=.0001
armdata=bpy.data.cameras.new('UR5e context');armcam=bpy.data.objects.new('UR5e context',armdata);scene.collection.objects.link(armcam)
armcam.location=(1.,-1.5,1.1);armcam.rotation_euler=(Vector((-.2,.1,.55))-armcam.location).to_track_quat('-Z','Y').to_euler();armdata.type='ORTHO';armdata.ortho_scale=1.35
scene.render.filepath=str(a.output.with_suffix('.png'))
# Reuse only identical path-traced geometry frames across paired trials.
# Labels are applied afterward, so each clip retains its own method and trace.
cache=a.bundle.parent/'frame-cache';cache.mkdir(exist_ok=True)
fingerprint=hashlib.sha256(Path(__file__).read_bytes())
fingerprint.update(json.dumps([info['geometry'],a.width,a.height,a.samples,a.view,bpy.app.version_string],sort_keys=True).encode())
for key in sorted(arr.files):
    if key!='transforms':fingerprint.update(key.encode()+arr[key].tobytes())
prefix=fingerprint.digest();cache_hits=0
started=time.monotonic();temp=a.output.with_suffix('.frame.jpg')
proc=None
if not a.still:
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','image2pipe','-vcodec','mjpeg','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(a.output)],stdin=subprocess.PIPE)
tool_geom=next(g['id']for g in info['geometry']if g['name']=='handle')
for i,frame in enumerate(transforms):
    target=Vector(frame[tool_geom,:3])+Vector((.02,0,0));cam.location=target+Vector({'hero':(.25,.65,.10),'side':(-.15,-.5,.65),'front':(0,.65,.05)}[a.view]);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();camdata.ortho_scale=.30
    close.location=target+Vector((-.15,-.5,.65));close.rotation_euler=(target-close.location).to_track_quat('-Z','Y').to_euler();close_data.ortho_scale=.30
    for g,ob in geom_objects.items():ob.location=frame[g,:3];ob.rotation_quaternion=Quaternion(frame[g,3:])
    scene.render.filepath=str(a.output.with_suffix('.png') if a.still else temp)
    cached=cache/(hashlib.sha256(prefix+frame.tobytes()).hexdigest()+'.jpg')
    if not a.still and cached.exists():
        shutil.copyfile(cached,temp);cache_hits+=1
    else:
        bpy.ops.render.render(write_still=True)
        main_path=Path(scene.render.filepath)
        if not a.still:
            pending=cached.with_suffix(f'.{os.getpid()}.tmp');shutil.copyfile(temp,pending);pending.replace(cached)
    if not a.still:
        im=Image.open(temp).convert('RGB');draw=ImageDraw.Draw(im)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        record=info.get('metadata',{});obs=record.get('frames',[{}])[min(i,len(record.get('frames',[{}]))-1)]
        draw.rounded_rectangle((22,18,a.width-22,70),radius=8,fill='#182a31')
        interface_label={"point":"CONTACT POINTS","pose":"CONTACT-POSE ADMITTANCE","wrench":"CONTACT WRENCH","velocity":"TANGENTIAL VELOCITY"}.get(record.get("interface"),record.get("method","inspection").upper())
        draw.text((40,34),f"TESOLLO DG5F-S  /  {interface_label}  /  {record.get('case','nominal')}",font=font,fill='#e9eeee')
        draw.text((a.width-180,34),'RTX · OPTIX',font=font,fill='#a9d4b7')
        caption=f"{obs.get('t',0):.2f} s  |  {obs.get('phase','reset').upper()}  |  screw {obs.get('screw_angle_deg',0):.1f}°  |  depth {obs.get('depth_mm',0):.3f} / 2.000 mm"
        if 'axial_force_N' in obs:caption+=f"  |  tilt {obs.get('tilt_deg',0):.1f}°"
        draw.text((30,85),caption,font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        if obs.get('phase') in ['lift','reorient']:
            detail=f"IN-HAND AXIS TURN {obs.get('inhand_axis_turn_deg',0):.1f}°  /  PALM TURN {obs.get('palm_turn_since_lift_deg',0):.2f}°"
        elif obs.get('phase') in ['turn','seat']:
            detail=f"HANDLE SPIN {obs.get('handle_spin_since_engagement_deg',0):.1f}°  /  {obs.get('contact_phase','turn').upper()}"
        else:detail='Purple flag: display-only witness on recorded handle'
        draw.text((30,112),detail,font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        draw.text((30,a.height-35),f"{obs.get('display_scale',record.get('video_time_scale',1)):g}× SIMULATION TIME · recorded physics poses",font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        loads=obs.get('handle_forces_N',[0,0]);slips=obs.get('handle_slip_signed_mm_s',[0,0])
        bx=a.width-510
        draw.rounded_rectangle((bx,150,a.width-22,252),radius=8,fill='#182a31')
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        draw.text((bx+14,158),'MEASURED LOAD / RELATIVE SLIP',font=small,fill='#bbc7c7')
        for row,name in enumerate(['Thumb','Middle']):
            color='#a9d4b7'if loads[row]>.01 else '#f5bb8a'
            draw.text((bx+14,180+row*27),f'{name}: {loads[row]:.3f} N  /  {slips[row]:+.2f} mm/s',font=font,fill=color)
        draw.text((bx+14,234),'Negative relative slip = sliding return',font=small,fill='#bbc7c7')
        im.save(temp,quality=95)
        raw=temp.read_bytes();proc.stdin.write(raw)
        if i==0:a.output.with_suffix('.jpg').write_bytes(raw)
    if i%30==0:print('RTX_FRAME',i,len(transforms),'elapsed',round(time.monotonic()-started,1),flush=True)
    if a.still:break
if proc and not a.no_end_card:
    draw.rounded_rectangle((22,a.height-80,a.width-22,a.height-22),radius=8,fill='#182a31')
    draw.text((40,a.height-60),f"{record.get('outcome','end').replace('_',' ').upper()}  /  FINAL FRAME HELD",font=font,fill='#e9eeee')
    im.save(temp,quality=95)
    for _ in range(a.fps):proc.stdin.write(temp.read_bytes())
if proc:
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    temp.unlink(missing_ok=True)
meta=dict(layout='single contact inspection view',renderer='Blender Cycles 4.5 / NVIDIA OptiX',RTX=True,devices=active,resolution=[a.width,a.height],fps=a.fps,samples=a.samples,denoiser='OPTIX',view=a.view,frame_start=a.start,frame_stop=a.start+len(transforms),identical_geometry_cache_hits=cache_hits,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=1 if a.still else len(transforms)+(0 if a.no_end_card else a.fps),end_card_s=0 if a.still or a.no_end_card else 1,elapsed_s=time.monotonic()-started,merged_rigid_thread_sectors=merged_sectors,geometry='Recorded MuJoCo meshes and poses; analytic cylinder edges use 0.25 mm display bevels',display_changes='PBR materials, shading normals, lighting, cameras, floor and fixed arm-base pedestal; tracking cameras, a purple display-only witness flag attached to recorded handle, and 0.25 mm analytic-cylinder display bevels; robot/tool poses from physics')
a.output.with_suffix('.render.json').write_text(json.dumps(meta,indent=2));print('RTX_DONE',json.dumps(meta),flush=True)
