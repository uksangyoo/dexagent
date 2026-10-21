"""Live reconstruction of reference/crate_human.mp4; work in progress."""
from pathlib import Path
import os
import json
os.environ["ROBOSIMSTUDIO_RENDERER"]="cycles"
import warp as wp
wp.set_device("cuda:0")
from dataclasses import replace
import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec, ObjectSpec, CameraSpec
from robosimstudio.specs.arrangement import backend
from robot import register
from policy import first_cycle, grasp_probe, lift_probe
from robosimstudio.solvers.render import register_renderer
register_renderer("cycles", "dexagent.rss.cycles_renderer.CyclesRenderer")

HERE = Path(__file__).resolve().parent
physics = replace(rss.MUJOCO_500HZ, substeps=20,
                  backends=(("*", backend("mujoco", nconmax=8192, njmax=16384, iterations=50, ls_iterations=30, enable_multiccd=True)),))
scene = rss.Scene(cell="void", name="crate_reference", physics=physics)
scene.add(StaticSpec(name="washer", asset=AssetSpec(path=str(HERE / "assets/washer.xml"))))
scene.add(rss.slab(name="cart_deck", half=(.2,.28,.025), at=(1.38,0,.06), color=(.13,.15,.16)))
scene.add(rss.robot(register()))
for i in range(11):
    scene.add(ObjectSpec(name=f"tray_{i:02}",
                         body_name=f"tray_{i:02}", joint_name=f"tray_{i:02}_free",
                         pose=(1.38,0,.1225+i*.075),
                         obb_grasp=False))
camera=json.loads((HERE/"reference/camera-fit.json").read_text())
scene.add(CameraSpec(name="reference", width=1280, height=720,
                     **{k:tuple(camera[k]) if k!="fov_y" else camera[k] for k in ("eye","look_at","up","fov_y")}))
scene.add(CameraSpec(name="overview", width=1280, height=720, fov_y=55,
                     eye=(3,-2.3,2.15), look_at=(.95,0,.8)))
BUILD = {"device": "cuda:0"}
routines={"1 grasp probe":grasp_probe,"2 lift probe":lift_probe,"3 first full cycle":first_cycle}
