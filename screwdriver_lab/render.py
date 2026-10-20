"""Render recorded physics coordinates only; never re-simulate for the video."""
import argparse,json,os,subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/dexagent-mesa-cache')
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from screwdriver_lab.scene import ROOT

def render(record,source,out):
    xml=ET.parse(source/f'{record["id"]}.xml').getroot()
    xml.find('compiler').set('meshdir',str(ROOT/'third-party/tesollo_model/dg5fs/meshes'))
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
    q=np.load(source/f'{record["id"]}.npz')['qpos'];frames=record['frames']
    renderer=mujoco.Renderer(m,420,640);cam=mujoco.MjvCamera();cam.lookat[:]=[-.11,.005,.25];cam.distance=.43;cam.azimuth=130;cam.elevation=-23
    options=mujoco.MjvOption();options.sitegroup[:]=0
    stem=f'g{record["goal_deg"]:g}-{record["id"]}'
    path=out/f'{stem}.mp4'
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x420','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(path)],stdin=subprocess.PIPE)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    def frame_at(i,ending=False):
        d.qpos[:]=q[i];mujoco.mj_forward(m,d);renderer.update_scene(d,cam,scene_option=options)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,640,35),fill=(17,25,31))
        draw.text((16,10),f'TESOLLO DG5F-S    /    {record["mode"].upper()}    /    {record["case"]}',fill='#e6ede9',font=font)
        o=frames[i];draw.rectangle((0,384,640,420),fill=(17,25,31))
        caption=f'{o["t"]:.2f} s    |    rotation {o["angle_deg"]:+.1f}°    |    {o["phase"].upper()}'
        if ending:caption=f'{record["outcome"].upper().replace("_"," ")}  ·  final frame held for 1 s'
        draw.text((16,395),caption,fill='#f2bd85' if ending else '#b9cbc4',font=font)
        return im
    try:
        for i in range(len(q)):
            im=frame_at(i);proc.stdin.write(im.tobytes())
            if i==0:im.save(out/f'{stem}.jpg',quality=85)
        end=frame_at(len(q)-1,True)
        for _ in range(10):proc.stdin.write(end.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    finally:renderer.close()
    record['video']=f'assets/{path.name}';record['poster']=f'assets/{stem}.jpg';record['render_fps']=10;record['end_card_s']=1

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'screwdriver_lab/artifacts/evaluation');p.add_argument('--site',type=Path,default=ROOT/'screwdriver_site/dist');p.add_argument('--manifest-name',default='manifest.json');a=p.parse_args()
    manifest=json.loads((a.input/'manifest.json').read_text());out=a.site/'assets';out.mkdir(parents=True,exist_ok=True)
    for r in manifest['results']:
        render(r,a.input,out);print('rendered',r['id'],flush=True)
    (a.site/'data').mkdir(exist_ok=True)
    (a.site/'data'/a.manifest_name).write_text(json.dumps(manifest))

if __name__=='__main__':main()
