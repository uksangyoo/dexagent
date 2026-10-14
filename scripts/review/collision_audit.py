"""Show the visible meshes and their convex collision hulls at the same recorded pose."""
from pathlib import Path
import mujoco,numpy as np
from PIL import Image
from scripts.review.render_rollouts import visual_model,camera
ROOT=Path(__file__).resolve().parents[2];run=ROOT/'runs/physics_revision_20261012'
for kind,name in [('nut','nut_hold_validity'),('screw','screw_hold_validity'),('wrench','wrench_hold_passive')]:
    q=np.load(run/f'{name}.npz')['qpos'];model=visual_model(run/f'free_{kind}.xml');data=mujoco.MjData(model)
    data.qpos[:]=q[len(q)//2];mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=450,width=720)
    c=camera(data,'hand');c.azimuth=80;c.distance=.26
    option=mujoco.MjvOption();option.sitegroup[:]=0
    for hull in [False,True]:
        option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]=hull
        renderer.update_scene(data,c,scene_option=option)
        Image.fromarray(renderer.render()).save(run/'rendered/assets'/f'{kind}_{"hull" if hull else "surface"}.jpg',quality=90)
    renderer.close()
