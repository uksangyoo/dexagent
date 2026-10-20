"""Decode exact RTX replays and audit device, bundle, frames and display changes."""
import argparse,json,subprocess,hashlib
from pathlib import Path
from screwdriver_lab.full_scene import OUT,ROOT
p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');a=p.parse_args();names=['slip-study-wrench-nominal','slip-study-pose-nominal','slip-study-velocity-nominal','slip-study-wrench-tilt-minus','slip-study-wrench-low-friction','slip-verified-seating'];checked=[]
for name in names:
 video=OUT/'rtx'/(name+'-contact.mp4');meta=video.with_suffix('.render.json')
 if a.available and not meta.exists():continue
 r=json.loads(meta.read_text());assert r['RTX'] and all('61:00'in d['id']and d['type']=='OPTIX'for d in r['devices'])
 assert r['source_bundle_sha256']==hashlib.sha256((OUT/'rtx'/name/'scene.npz').read_bytes()).hexdigest()
 assert all(c['renderer_sha256']==hashlib.sha256((ROOT/'screwdriver_lab/render_controlled_slip.py').read_bytes()).hexdigest()for c in r['chunks'])
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]));stream=next(s for s in probe['streams']if s['codec_type']=='video');assert [stream['width'],stream['height']]==r['resolution'];assert int(stream['nb_frames'])==r['frames'];num,den=map(int,stream['avg_frame_rate'].split('/'));assert abs(num/den-r['fps'])<1e-9
 subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
 # Correct the generic render description: analytic cylinder edges have display bevels.
 r['geometry']='Recorded MuJoCo visual mesh vertices and world transforms; analytic cylinders have 0.25 mm visual edge bevels.';r['display_changes']='PBR materials, shading normals, lighting, cameras, floor, fixed arm-base pedestal and analytic-cylinder edge bevels. Collision geometry and recorded robot/tool poses are unchanged.';r['audit_correction']='The generic renderer description omitted its analytic-cylinder bevel modifier; this audit makes it explicit.';r['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest();meta.write_text(json.dumps(r,indent=2))
 checked.append(dict(id=name,frames=r['frames'],fps=r['fps'],resolution=r['resolution'],duration_s=float(probe['format']['duration']),sha256=r['video_sha256'],gpu_devices=r['devices']))
result=dict(passed=True,scope='RTX device, exact pose bundle hash, frame count, dimensions, fps and complete decode',videos=checked);(ROOT/'screwdriver_site/qa/slip-media.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
