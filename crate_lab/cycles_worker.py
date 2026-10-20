"""Persistent GPU-1 OptiX rendering worker; receives geometry, never physics."""
import sys,json,math
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector,Matrix

path=Path(sys.argv[1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.render.engine="CYCLES"
scene.cycles.device="GPU"
scene.cycles.samples=32
scene.cycles.use_denoising=True
scene.cycles.denoiser="OPTIX"
scene.cycles.max_bounces=6
scene.render.threads_mode="FIXED"
scene.render.threads=4
scene.render.use_persistent_data=True
scene.render.image_settings.file_format="PNG"
scene.render.image_settings.color_mode="RGB"
scene.view_settings.view_transform="AgX"
prefs=bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type="OPTIX"
prefs.get_devices()
for d in prefs.devices:d.use=d.type=="OPTIX"
active=[{"id":d.id,"name":d.name} for d in prefs.devices if d.use]
if len(active)!=1:raise RuntimeError(f"Expected only physical GPU 1 visible, got {active}")
(path/"gpu.json").write_text(json.dumps(active))
scene.world=bpy.data.worlds.new("Room daylight")
scene.world.use_nodes=True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value=(.8,.86,1,1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value=.3
for name,xyz,power,size in [("Ceiling softbox",(1,0,2.6),180,2),("Window",(1,1.5,1.8),100,1.4)]:
    light=bpy.data.lights.new(name,"AREA");light.energy=power;light.shape="DISK";light.size=size
    ob=bpy.data.objects.new(name,light);scene.collection.objects.link(ob);ob.location=xyz
    ob.rotation_euler=(Vector((1,0,.7))-ob.location).to_track_quat("-Z","Y").to_euler()
meta=json.loads((path/"geometry.json").read_text());data=np.load(path/"geometry.npz");objects=[];materials={}
for i,item in enumerate(meta):
    mesh=bpy.data.meshes.new(item["name"]);mesh.from_pydata(data[f"v{i}"].tolist(),[],data[f"f{i}"].tolist());mesh.update()
    ob=bpy.data.objects.new(item["name"],mesh);scene.collection.objects.link(ob)
    if "ur5e" in item["name"]:
        for polygon in mesh.polygons:polygon.use_smooth=True
    color=tuple(item["color"]);key=str(color)
    if key not in materials:
        mat=bpy.data.materials.new(key);mat.use_nodes=True
        bsdf=mat.node_tree.nodes["Principled BSDF"]
        # Newton colors are display RGB; Blender's shader expects linear RGB.
        linear=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in color]
        bsdf.inputs["Base Color"].default_value=(*linear,1)
        bsdf.inputs["Roughness"].default_value=.36 if max(color)<.3 else .48
        materials[key]=mat
    ob.data.materials.append(materials[key]);objects.append(ob)
camera=bpy.data.cameras.new("Reference camera");ob=bpy.data.objects.new("Reference camera",camera);scene.collection.objects.link(ob);scene.camera=ob
for line in sys.stdin:
    request=json.loads(line)
    for item,matrix in zip(objects,np.load(path/"transforms.npy")):item.matrix_world=Matrix(matrix.tolist())
    scene.render.resolution_x=request["width"];scene.render.resolution_y=request["height"];scene.render.resolution_percentage=100
    camera.sensor_fit="VERTICAL";camera.sensor_height=24
    camera.lens=12/math.tan(math.radians(request["fov_y"])/2)
    ob.location=request["eye"]
    forward=(Vector(request["look_at"])-ob.location).normalized()
    right=forward.cross(Vector(request["up"])).normalized()
    up=right.cross(forward).normalized()
    ob.rotation_euler=Matrix((right,up,-forward)).transposed().to_euler()
    scene.render.filepath=request["output"]
    bpy.ops.render.render(write_still=True)
    print("FRAME_DONE",flush=True)
