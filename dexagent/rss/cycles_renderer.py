"""RoboSimStudio renderer plugin: unchanged Newton geometry/state -> OptiX.

OVRTX in this environment fails CUDA/Vulkan interop under a GPU visibility
mask. Cycles respects the mask; use it for the live page's snapshots and takes.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import imageio.v3 as iio
from robosimstudio.core.render.base import RendererBase
from robosimstudio.core.render.visual import model_visual_parts, transform_matrix


class CyclesRenderer(RendererBase):
    name = "cycles"
    devices = ("cuda", "cpu")
    supported_channels = ("rgb",)

    def __init__(self, model, cameras, *, channels=None, **kwargs):
        super().__init__(cameras, channels=channels)
        self.parts = model_visual_parts(model, plane_extent=5)
        self.directory = Path(tempfile.mkdtemp(prefix="rss_optix_"))
        meshes, meta = {}, []
        for i, p in enumerate(self.parts):
            meshes[f"v{i}"] = np.asarray(p.mesh.vertices)
            meshes[f"f{i}"] = np.asarray(p.mesh.faces)
            color = np.asarray(p.mesh.visual.material.baseColorFactor[:3])/255.
            meta.append({"name":p.name,"color":color.tolist()})
        np.savez(self.directory/"geometry.npz",**meshes)
        (self.directory/"geometry.json").write_text(json.dumps(meta))
        self.proc = subprocess.Popen([
            os.environ.get("DEXAGENT_RENDER_PYTHON", "/home/uyoo/miniconda3/envs/blender-render/bin/python"),
            str(Path(__file__).with_name("cycles_worker.py")),str(self.directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ,"CUDA_VISIBLE_DEVICES":"1","OMP_NUM_THREADS":"4"})
        self.log = (self.directory/"render.log").open("w")
        self.last_key = None
        self.last_frames = None

    def render(self,state):
        poses = state.body_q.numpy()
        key = poses.tobytes()
        if key == self.last_key: return self.last_frames
        transforms = np.array([(transform_matrix(poses[p.body]) if p.body>=0 else np.eye(4))@p.local for p in self.parts])
        np.save(self.directory/"transforms.npy",transforms)
        frames = {}
        for camera in self.cameras:
            output = self.directory/f"{camera.name}.png"
            msg = {"width":camera.width,"height":camera.height,"fov_y":camera.fov_y,
                   "eye":camera.eye,"look_at":camera.look_at,"up":camera.up,"output":str(output)}
            self.proc.stdin.write(json.dumps(msg)+"\n");self.proc.stdin.flush()
            while True:
                line=self.proc.stdout.readline()
                if not line: raise RuntimeError(f"OptiX worker exited; see {self.directory}/render.log")
                self.log.write(line);self.log.flush()
                if line.startswith("FRAME_DONE"):break
            frames[camera.name] = {"rgb":np.asarray(iio.imread(output))[...,:3]}
        self.last_key,self.last_frames = key,frames
        return frames

    def close(self):
        if getattr(self,"proc",None):
            self.proc.stdin.close()
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.proc.terminate()
            self.log.close()
