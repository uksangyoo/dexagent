"""Render recorded Newton qpos through MuJoCo OpenGL, without resimulating."""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PALETTE = [[.24,.73,.65,1],[.35,.59,.88,1],[.68,.51,.81,1],[.90,.52,.40,1],[.77,.66,.45,1]]
VIEWS = {"hand": (20, -18, .34), "side": (280, -15, .36), "robot": (35, -18, 1.55)}


def visual_model(scene):
    root = ET.parse(scene).getroot()
    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="800")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.22 0.22 0.22", diffuse="0.3 0.3 0.3", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "map", znear="0.001", zfar="50", fogstart="3", fogend="8")
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="review_sky", type="skybox", builtin="gradient",
                  rgb1="0.78 0.83 0.87", rgb2="0.95 0.96 0.95", width="512", height="3072")
    ET.SubElement(assets, "texture", name="review_grid", type="2d", builtin="checker",
                  rgb1="0.82 0.85 0.86", rgb2="0.87 0.90 0.91", width="256", height="256")
    ET.SubElement(assets, "material", name="review_floor", texture="review_grid", texrepeat="12 12", reflectance="0")
    world = root.find("worldbody")
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(world, "geom", name="review_floor", type="plane", pos="0 0 -0.014", size="2 2 .1",
                  material="review_floor", contype="0", conaffinity="0", group="2")
    for pos, direction, diffuse in [("0 -2 3", "0 .3 -1", ".5 .5 .5"), ("-2 0 1", "1 0 -.3", ".2 .22 .25"), ("1 1 2", "-1 -1 -1", ".15 .15 .15")]:
        ET.SubElement(world, "light", pos=pos, dir=direction, diffuse=diffuse, directional="true", castshadow="true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        name = model.geom(g).name or ""
        if name.endswith("_col"):
            model.geom_rgba[g, 3] = 0  # Arm collision hulls duplicate its visual meshes.
        if body.startswith("link_"):
            parts = body.split("_")
            if len(parts) == 3 and parts[1].isdigit():
                model.geom_matid[g] = -1
                model.geom_rgba[g] = PALETTE[int(parts[1]) - 1]
            else:
                model.geom_matid[g] = -1
                model.geom_rgba[g] = [.24,.29,.33,1]
        if body == "dg5fs_mount":
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [.25,.29,.32,1]
        if body == "nut":
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [.96,.66,.20,1] if name != "nut_face_0" else [.85,.24,.12,1]
    return model


def camera(data, view):
    azimuth, elevation, distance = VIEWS[view]
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    cam.lookat[:] = data.body("nut").xpos + [0, 0, .025]
    if view == "robot":
        cam.lookat[:] = [0, -.28, .22]
    return cam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheet", action="store_true")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    # Arm render meshes are group 1 and hand physical meshes are group 0.
    for metadata_path in sorted(args.input.glob("*.json")):
        meta = json.loads(metadata_path.read_text())
        name = meta["id"]
        if args.only and name not in args.only:
            continue
        q = np.load(args.input / f"{name}.npz")["qpos"]
        model = visual_model(meta["scene"])
        data = mujoco.MjData(model)
        data.qpos[:] = q[0]
        mujoco.mj_forward(model, data)
        if args.sheet:
            renderer = mujoco.Renderer(model, height=360, width=540)
            tiles = []
            for angle in [0, 60, 120, 155, 210, 255]:
                c = camera(data, "hand"); c.azimuth = angle
                renderer.update_scene(data, c, scene_option=options)
                frame = Image.fromarray(renderer.render())
                ImageDraw.Draw(frame).text((14, 14), f"azimuth {angle}", fill="black", font_size=22)
                tiles.append(frame)
            sheet = Image.new("RGB", (1620, 720))
            for i, tile in enumerate(tiles): sheet.paste(tile, ((i % 3)*540, (i // 3)*360))
            sheet.save(args.output / "camera_sheet.jpg")
            renderer.close()
            break
        renderer = mujoco.Renderer(model, height=600, width=960)
        meta["views"] = {}
        for view in VIEWS:
            data.qpos[:] = q[0]
            mujoco.mj_forward(model, data)
            c = camera(data, view)
            video_path = args.output / f"{name}_{view}.mp4"
            if args.resume and video_path.exists() and all(
                (args.output / f"{name}_{view}_{i:04}.jpg").exists()
                for i in {0, len(q)//4, len(q)//2, 3*len(q)//4, len(q)-1}
            ):
                meta["views"][view] = {"video": f"assets/{video_path.name}", "poster": f"assets/{name}_{view}_0000.jpg"}
                continue
            with imageio.get_writer(video_path, fps=meta["fps"], codec="libx264", quality=7,
                                    macro_block_size=2, ffmpeg_log_level="error",
                                    output_params=["-movflags", "+faststart", "-threads", "2"]) as video:
                for index, pose in enumerate(q):
                    data.qpos[:] = pose
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    mujoco.mj_camlight(model, data)
                    renderer.update_scene(data, c, scene_option=options)
                    # Optional overlays are recorded Newton contacts, never
                    # recomputed from the renderer's collision geometry.
                    for contact in meta["frames"][index].get("contacts", []):
                        scene = renderer.scene
                        if scene.ngeom + 2 >= scene.maxgeom:
                            break
                        point = np.asarray(contact["point"])
                        force = np.asarray(contact["force_on_object_N"])
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.full(3, .0012), point, np.eye(3).ravel(),
                                           np.array([1., .98, .85, 1.]))
                        scene.ngeom += 1
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                           np.array([.9, .17, .08, 1.]))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                            .0006, point, point + force*.01)
                        scene.ngeom += 1
                    pixels = renderer.render()
                    video.append_data(pixels)
                    if index in {0, len(q)//4, len(q)//2, 3*len(q)//4, len(q)-1}:
                        image = Image.fromarray(pixels)
                        image.save(args.output / f"{name}_{view}_{index:04}.jpg", quality=88)
            meta["views"][view] = {"video": f"assets/{video_path.name}", "poster": f"assets/{name}_{view}_0000.jpg"}
            print(json.dumps({"rendered": video_path.name, "bytes": video_path.stat().st_size}), flush=True)
        renderer.close()
        meta["rendering"] = {"renderer": "MuJoCo OpenGL", "backend": os.environ.get("MUJOCO_GL", "default"),
                             "RTX": False, "resolution": [960, 600],
                             "display_changes": "Finger colors, highlighted nut face, lighting and a nonphysical ground plane; no pose interpolation or generated imagery."}
        # Keep server paths out of the hosted review. State/scene hashes identify provenance.
        meta["scene"] = Path(meta["scene"]).name
        (args.output.parent / "data" / f"{name}.json").write_text(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
