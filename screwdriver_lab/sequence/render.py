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

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=24);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=960);p.add_argument('--fps',type=int,default=25);p.add_argument('--still',action='store_true');p.add_argument('--view',choices=['hero','side','front','overview'],default='hero');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);p.add_argument('--no-end-card',action='store_true');p.add_argument('--azimuth',type=float,default=-90);p.add_argument('--elevation',type=float,default=20);p.add_argument('--geometry-only',action='store_true');p.add_argument('--overview-inset',action='store_true');a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
info=json.loads((a.bundle/'scene.json').read_text());is_cube=info.get('metadata',{}).get('case')=='cube rotation';is_assembly=info.get('metadata',{}).get('case')in ['continuous assembly','prepared assembly diagnostic'];is_pickup=info.get('metadata',{}).get('case')=='pickup development' or is_cube or is_assembly;arr=np.load(a.bundle/'scene.npz');transforms=arr['transforms'][a.start:a.stop];info['metadata']['frames']=info['metadata']['frames'][a.start:a.stop]
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
handle=material('Screwdriver grip',(.023,.03,.037),0,.36);cap=material('Red polymer cap',(.55,.025,.018),0,.3);steel=material('Brushed steel',(.58,.65,.72),.65,.42);stripe=material('Rotation witness mark',(.92,.50,.075),.1,.33)
work=material('Teal fixture',(.055,.19,.21),.5,.32);wood=material('Workbench',(.21,.115,.055),0,.6)
geom_objects={}
for g in info['geometry']:
    name=g['name'];typ=g['type'];size=g['size']
    if name.startswith('arm_collision_'):continue
    if a.geometry_only and g['body']not in ['screwdriver','swivel_cap','screw','female_thread']and not name.startswith('workpiece_'):continue
    if typ==7:
        mid=g['mesh'];mesh=bpy.data.meshes.new(name);mesh.from_pydata(arr[f'verts_{mid}'].tolist(),[],arr[f'faces_{mid}'].tolist());mesh.update();ob=bpy.data.objects.new(name,mesh);scene.collection.objects.link(ob)
        is_tip=name.startswith('finger_tip_')
        mat=handle if name=='handle' else steel if name=='driver_bit' or name.startswith(('slot_wall','male_thread','female_thread')) else base if g['body']=='palm' else rubber if is_tip else metal
        if g['body'] in ['base','shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']:mat=material(name+' finish',tuple(g['rgba'][:3]),.35,.45)
    elif typ==5:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=size[0],depth=2*size[1]);ob=bpy.context.object;mat=handle if name=='handle' else cap if name=='cap' else steel
        bevel=ob.modifiers.new('Machined edge bevel','BEVEL');bevel.width=.00025;bevel.segments=3
    elif typ==6:
        bpy.ops.mesh.primitive_cube_add(size=1);ob=bpy.context.object;ob.scale=tuple(2*x for x in size);mat=work if name.startswith('workpiece') else wood if name in ['bench','pickup_surface','pickup_cap','pickup_shaft'] else base if name.startswith('cradle') else stripe if name=='stripe' else steel
    else:raise RuntimeError(f'Unsupported visual geom {typ}')
    if name=='cube':mat=material('Cube polymer',(.8,.15,.045),0,.4)
    if name=='cube_pedestal':mat=work
    ob.name=name;ob.data.materials.append(mat);
    if name=='handle' and 'geom_to_body_rotation'in g:
        # Paint the recorded surface; add no marker geometry or pose changes.
        B=np.asarray(g['geom_to_body_rotation']);pbody=np.asarray(g['geom_to_body_translation'])
        ob.data.materials.append(stripe)
        for poly in ob.data.polygons:
            center=B@np.asarray(poly.center)+pbody;normal=B@np.asarray(poly.normal)
            if abs(math.atan2(center[1],center[0]))<.22 and abs(normal[2])<.5:poly.material_index=1
    if name=='cube':
        face_colors=[(.03,.32,.7),(.92,.67,.05),(.9,.12,.04),(.08,.6,.28),(.8,.8,.73),(.55,.08,.5)]
        for face_index,color in enumerate(face_colors):ob.data.materials.append(material('Cube face '+str(face_index),color,0,.45))
        for face_index,poly in enumerate(ob.data.polygons):poly.material_index=face_index+1
    ob.rotation_mode='QUATERNION';geom_objects[g['id']]=ob
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
handle_id=next(g['id']for g in info['geometry']if g['name']==('cube'if is_cube else'handle'))
bpy.ops.mesh.primitive_cube_add(size=1);witness=bpy.context.object;witness.name='visual_rotation_witness';witness.parent=geom_objects[handle_id]
# A fixed initial marker phase makes the display-only witness visible from
# the inspection camera. It thereafter follows the recorded handle exactly.
direction=Quaternion(transforms[0,handle_id,3:]).to_matrix().transposed()@Vector((-1.,0.,0.));direction.z=0;direction.normalize()
radius=next(g['size'][0]for g in info['geometry']if g['name']==('cube'if is_cube else'handle'))
witness.hide_render=True
witness.location=(radius+.007)*direction+Vector((0,0,-.028));witness.rotation_euler.z=math.atan2(direction.y,direction.x);witness.scale=(.014,.0003,.006);witness.data.materials.append(material('Purple visual witness',(.40,.08,.65),0,.5))
bpy.ops.mesh.primitive_plane_add(size=200);bpy.context.object.data.materials.append(material('Warm studio surface',(.48,.52,.50),0,.65))
# Display-only support under the fixed arm base; never modifies robot poses.
bpy.ops.mesh.primitive_cube_add(size=1,location=(-.52,.25,.055));bpy.context.object.scale=(.19,.19,.11);bpy.context.object.data.materials.append(base)

def light(name,pos,power,size,color):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size;data.color=color
    ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob);ob.location=pos;ob.rotation_euler=(Vector((.035,.01,.27))-ob.location).to_track_quat('-Z','Y').to_euler()
light('Large soft key',(.3,-.3,1.5),30,1.,(1,.91,.80));light('Cool fill',(.1,.75,1.),12,.8,(.77,.86,1));light('Rim',(-.8,-.05,1.2),15,.8,(1,.97,.91))
camdata=bpy.data.cameras.new('Inspection camera');cam=bpy.data.objects.new('Inspection camera',camdata);scene.collection.objects.link(cam);scene.camera=cam
cam.location={'hero':(-.35,-.55,.95),'side':(-.65,.15,.52),'front':(.05,-.8,.5),'overview':(1.,-1.5,1.1)}[a.view];target=Vector((.0,.0,.345));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();camdata.type='ORTHO';camdata.ortho_scale=.32;camdata.lens=55
close_data=bpy.data.cameras.new('Bit and screw inspection');close=bpy.data.objects.new('Bit and screw inspection',close_data);scene.collection.objects.link(close)
close.location=(.035,-.05,.282);close.rotation_euler=(Vector((0,0,.247))-close.location).to_track_quat('-Z','Y').to_euler();close_data.type='ORTHO';close_data.ortho_scale=.025;close_data.clip_start=.0001
armdata=bpy.data.cameras.new('UR5e context');armcam=bpy.data.objects.new('UR5e context',armdata);scene.collection.objects.link(armcam)
armcam.location=(1.,-1.5,1.1);armcam.rotation_euler=(Vector((-.2,.1,.55))-armcam.location).to_track_quat('-Z','Y').to_euler();armdata.type='ORTHO';armdata.ortho_scale=1.35
scene.render.filepath=str(a.output.with_suffix('.png'))
# Reuse only identical path-traced geometry frames across paired trials.
# Labels are applied afterward, so each clip retains its own method and trace.
cache=a.bundle.parent/'frame-cache';cache.mkdir(exist_ok=True)
fingerprint=hashlib.sha256(Path(__file__).read_bytes())
fingerprint.update(json.dumps([info['geometry'],a.width,a.height,a.samples,a.view,a.azimuth,a.elevation,is_pickup,list(direction),bpy.app.version_string],sort_keys=True).encode())
for key in sorted(arr.files):
    if key!='transforms':fingerprint.update(key.encode()+arr[key].tobytes())
prefix=fingerprint.digest();cache_hits=0
started=time.monotonic();temp=a.output.with_suffix('.frame.jpg')
proc=None
if not a.still:
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','image2pipe','-vcodec','mjpeg','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(a.output)],stdin=subprocess.PIPE)
tool_geom=next(g['id']for g in info['geometry']if g['name']==('cube'if is_cube else'handle'))
for i,frame in enumerate(transforms):
    record=info.get('metadata',{});obs=record.get('frames',[{}])[min(i,len(record.get('frames',[{}]))-1)]
    bit_view=not is_pickup or (is_assembly and obs.get('phase')in ['approach','align_slot','drive','drive_hold'] and obs.get('tip_height_above_slot_floor_mm',1e9)<12)
    target=Vector(frame[tool_geom,:3])+Vector((.02,0,-.025));cam.location=target+Vector((.65*math.cos(math.radians(a.azimuth)),.65*math.sin(math.radians(a.azimuth)),.65*math.tan(math.radians(a.elevation))));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();camdata.ortho_scale=.36
    if a.view=='overview':
        cam.location=(1.,-1.5,.95);cam.rotation_euler=(Vector((-.2,.05,.32))-cam.location).to_track_quat('-Z','Y').to_euler();camdata.ortho_scale=1.05
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
        # A second RTX view shows actual bit/head contact from the same state.
        im=Image.open(temp).convert('RGB')
        head_id=next(g['id']for g in info['geometry']if g['name']==('cube'if is_cube else'slot_floor'))
        head=Vector(frame[head_id,:3])+Vector((0,0,.001))if bit_view else Vector(frame[handle_id,:3])
        close.location=head+Vector((.035,-.05,.035)if bit_view else(.0,.65,.237)if is_cube else(-.035,-.05,.06));close.rotation_euler=(head-close.location).to_track_quat('-Z','Y').to_euler();close_data.ortho_scale=.014 if bit_view else .15
        inset=a.output.with_suffix('.inset.jpg');scene.camera=close;scene.render.resolution_x=320;scene.render.resolution_y=240;scene.render.filepath=str(inset)
        bpy.ops.render.render(write_still=True)
        scene.camera=cam;scene.render.resolution_x=a.width;scene.render.resolution_y=a.height
        im.paste(Image.open(inset).convert('RGB'),(a.width-342,a.height-300));inset.unlink(missing_ok=True)
        if a.overview_inset:
            close.location=(1.,-1.5,.95);close.rotation_euler=(Vector((-.2,.05,.32))-close.location).to_track_quat('-Z','Y').to_euler();close_data.ortho_scale=1.05
            scene.camera=close;scene.render.resolution_x=320;scene.render.resolution_y=240;scene.render.filepath=str(inset);bpy.ops.render.render(write_still=True)
            im.paste(Image.open(inset).convert('RGB'),(24,a.height-300));inset.unlink(missing_ok=True)
            scene.camera=cam;scene.render.resolution_x=a.width;scene.render.resolution_y=a.height
        draw=ImageDraw.Draw(im)
        draw.rectangle((a.width-344,a.height-302,a.width-20,a.height-58),outline='#e4ece2',width=2)
        small_label=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
        if a.overview_inset:
            draw.rectangle((22,a.height-302,346,a.height-58),outline='#e4ece2',width=2);draw.rectangle((22,a.height-324,346,a.height-302),fill='#182a31');draw.text((32,a.height-322),'UR5e + TESOLLO · SAME RECORDED STATE',font=small_label,fill='#e9eeee')
        draw.rectangle((a.width-344,a.height-324,a.width-20,a.height-302),fill='#182a31')
        draw.text((a.width-334,a.height-322),('BIT / SCREW · SAME RECORDED STATE'if bit_view else'CONTACTS · SAME RECORDED STATE'),font=small_label,fill='#e9eeee')
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        record=info.get('metadata',{});obs=record.get('frames',[{}])[min(i,len(record.get('frames',[{}]))-1)]
        draw.rounded_rectangle((22,18,a.width-22,70),radius=8,fill='#182a31')
        interface_label={"point":"CONTACT POINTS","pose":"CONTACT-POSE ADMITTANCE","wrench":"CONTACT WRENCH","velocity":"TANGENTIAL VELOCITY"}.get(record.get("interface"),record.get("method","inspection").upper())
        draw.text((40,34),f"TESOLLO DG5F-S  /  {interface_label}  /  {record.get('case','nominal')}",font=font,fill='#e9eeee')
        draw.text((a.width-180,34),'RTX · OPTIX',font=font,fill='#a9d4b7')
        draw.rounded_rectangle((22,76,a.width-22,142),radius=8,fill='#182a31')
        caption=f"{obs.get('t',0):.2f} s  |  {obs.get('phase','reset').upper()}  |  screw {obs.get('screw_angle_deg',0):.1f}°  |  depth {obs.get('depth_mm',0):.3f} / {record.get('seating_target_mm',2.):.3f} mm"
        if is_pickup:caption=f"{obs.get('t',0):.2f} s  |  {obs.get('phase','pinch').upper()}  |  tilt {obs.get('tilt_deg',0):.1f}°  |  index cap load {obs.get('cap_force_N',0):.3f} N"
        if is_assembly and obs.get('phase')in ['transport','approach','align_slot','drive','drive_hold']:caption=f"{obs.get('t',0):.2f} s  |  {obs['phase'].upper()}  |  screw {obs.get('screw_angle_deg',0):.1f}°  |  depth {obs.get('depth_mm',0):.3f} / {record.get('seating_target_mm',0):.3f} mm"
        if is_assembly and obs.get('controller_phase'):caption=caption.replace(obs['phase'].upper(),obs['phase'].upper()+' / '+obs['controller_phase'].upper(),1)
        if is_cube:caption=f"{obs.get('t',0):.2f} s  |  {obs.get('phase','acquire').upper()}  |  in-hand {obs.get('inhand_turn_deg',0):.1f}°  |  goal error {obs.get('orientation_error_deg',0):.1f}°"
        if is_cube and record.get('prepared_start'):caption+='  |  PREPARED GRASP'
        if 'axial_force_N' in obs:caption+=f"  |  tilt {obs.get('tilt_deg',0):.1f}°"
        draw.text((30,85),caption,font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        if obs.get('phase') in ['lift','reorient']:
            detail=f"IN-HAND AXIS TURN {obs.get('inhand_axis_turn_deg',0):.1f}°  /  PALM TURN {obs.get('palm_turn_since_lift_deg',0):.2f}°"
        elif obs.get('phase') in ['turn','seat']:
            detail=f"HANDLE SPIN {obs.get('handle_spin_since_engagement_deg',0):.1f}°  /  {obs.get('contact_phase','turn').upper()}"
        else:detail='Matched slotted driver / M3-like fastener · declared dimensions'
        if is_pickup:detail='PICKUP DEVELOPMENT · recorded forces and poses · no screw-in credit'
        if is_assembly:
            detail='CONTINUOUS PHYSICAL PICKUP → PIVOT → ENGAGE → DRIVE'
            if 'tip_xy_error_mm'in obs:detail=f"TIP OFFSET {obs['tip_xy_error_mm']:.3f} mm  /  TIP HEIGHT {obs['tip_height_above_slot_floor_mm']:.2f} mm  /  BIT LOAD {obs['bit_axial_load_N']:.3f} N"
        if record.get('case')=='prepared assembly diagnostic':detail='PREPARED STATE · '+detail
        if is_cube:detail=f"FREE CUBE · {record.get('goal_deg',90):g}° {record.get('goal_axis','z').upper()} GOAL · palm turn {obs.get('palm_turn_deg',0):.2f}° · lift {obs.get('lift_mm',0):.1f} mm"
        draw.text((30,112),detail,font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        draw.text((30,a.height-35),f"{obs.get('display_scale',record.get('video_time_scale',1)):g}× SIMULATION TIME · recorded Drake dynamics",font=font,fill='#edf2ef',stroke_width=1,stroke_fill='#495657')
        loads=obs.get('handle_forces_N',[0,0]);slips=obs.get('handle_slip_signed_mm_s',[0,0])
        bx=370 if a.overview_inset else 24;by=a.height-170
        draw.rounded_rectangle((bx,by,bx+488,by+102),radius=8,fill='#182a31')
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        show_slip=not is_pickup or bool(obs.get('slip_recorded'))
        draw.text((bx+14,by+8),('MEASURED LOAD / RELATIVE SLIP'if show_slip else'MEASURED CONTACT LOAD'),font=small,fill='#bbc7c7')
        for row,name in enumerate(['Thumb','Middle']):
            color='#a9d4b7'if loads[row]>.01 else '#f5bb8a'
            draw.text((bx+14,by+30+row*27),(f'{name}: {loads[row]:.3f} N'if not show_slip else f'{name}: {loads[row]:.3f} N  /  {slips[row]:+.2f} mm/s'),font=font,fill=color)
        draw.text((bx+14,by+84),('Cube rotation trial; actual physical contacts'if is_cube else'Negative relative slip = sliding return'if show_slip else'Physical contacts; no linear-slip metric recorded'),font=small,fill='#bbc7c7')
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
meta=dict(full_arm_inset=a.overview_inset,layout='hand contact view with synchronized RTX contact inset'if is_pickup else'hand contact view with synchronized RTX bit/screw inset',renderer='Blender Cycles 4.5 / NVIDIA OptiX',RTX=True,devices=active,resolution=[a.width,a.height],fps=a.fps,samples=a.samples,denoiser='OPTIX',view=a.view,camera_azimuth_deg=a.azimuth,camera_elevation_deg=a.elevation,frame_start=a.start,frame_stop=a.start+len(transforms),identical_geometry_cache_hits=cache_hits,rotation_witness_visible=False,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=1 if a.still else len(transforms)+(0 if a.no_end_card else a.fps),end_card_s=0 if a.still or a.no_end_card else 1,elapsed_s=time.monotonic()-started,merged_rigid_thread_sectors=merged_sectors,geometry='Recorded Drake poses and corresponding CAD visual meshes; analytic cylinder edges use 0.25 mm display bevels',display_changes='PBR materials, shading normals, lighting, cameras, floor and fixed arm-base pedestal; tracking cameras, a painted axial stripe on the recorded handle surface, and 0.25 mm analytic-cylinder display bevels; robot/tool poses from physics')
a.output.with_suffix('.render.json').write_text(json.dumps(meta,indent=2));print('RTX_DONE',json.dumps(meta),flush=True)
