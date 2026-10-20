"""Small source bundle and separate RTX media packs for the corrected task."""
import json,re,zipfile
from pathlib import Path
from screwdriver_lab.screw_scene import ROOT,OUT

def main():
 site=ROOT/'screwdriver_site/dist';data=json.loads((site/'data/screw-in.json').read_text())
 assert len(data['results'])==15 and data['rendering']['GPU']==1
 media=[site/'assets/human-reference.mp4',site/'assets/human-reference.jpg',site/'assets/screw-in-seated-side.png']
 for r in data['results']:
  assert r['rendering']['RTX'];media.extend([site/r['video'],site/r['poster']])
 groups=[];group=[];size=0
 for path in media:
  assert path.exists()
  if group and size+path.stat().st_size>9*1024**2:groups.append(group);group=[];size=0
  group.append(path);size+=path.stat().st_size
 if group:groups.append(group)
 links=[]
 for i,group in enumerate(groups,1):
  pack=site/'assets'/f'screw-in-media-{i}.zip'
  with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for path in group:z.write(path,path.relative_to(ROOT))
  assert pack.stat().st_size<12*1024**2
  links.append(f'<a href="assets/{pack.name}" download>Media pack {i} · {pack.stat().st_size/1048576:.1f} MB ↗</a>')
 html=site/'index.html';content=html.read_text();block='<details id="screw-media"><summary>Download videos for offline use</summary><p>Extract the source bundle and every media pack into the same folder.</p><div class="artifact-links">'+''.join(links)+'</div></details>'
 if 'id="screw-media"' in content:content=re.sub(r'<details id="screw-media">.*?</details>',block,content)
 else:content=content.replace('<div class="artifact-links"><a href="data/screw-in.json"',block+'<div class="artifact-links"><a href="data/screw-in.json"')
 html.write_text(content)
 paths=[ROOT/'screwdriver_lab'/n for n in ['__init__.py','scene.py','screw_scene.py','screw_sim.py','screw_control.py','run_screw.py','render_screw.py','export_rtx.py','validate_screw.py','serve_screw.py','check_screw_media.py','package_screw.py','requirements.txt','SCREW_IN.md']]
 paths+=[ROOT/'dextrain/__init__.py',ROOT/'dextrain/resource_guard.py',ROOT/'third-party/tesollo_model/LICENSE',ROOT/'third-party/tesollo_model/dg5fs/dg5fs_right.xml',ROOT/'screwdriver_lab/artifacts/grasp.json']
 paths+=list((ROOT/'third-party/tesollo_model/dg5fs/meshes/dg5fs_right').glob('*'))
 paths+=[OUT/'precision_grasp.json',OUT/'motor_bank.npz',OUT/'manifest.json',OUT/'validation.json']
 for r in data['results']:
  paths += [OUT/(r['id']+ext) for ext in ['.xml','.npz','.json']]
  paths += [site/Path(r['video']).with_suffix('.render.json')]
 paths += [site/n for n in ['index.html','screw.js','screw.css','style.css','data/screw-in.json','data/screw-validation.json']]
 output=site/'assets/screw-in-pipeline.zip'
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for path in sorted(set(paths)):
   assert path.is_file(),path
   if path==html:z.writestr(str(path.relative_to(ROOT)),content.replace('href="rotation.html"','href="https://dexagent-screwdriver-lab.comet-basil-0008.chatgpt.site/rotation.html"'))
   else:z.write(path,path.relative_to(ROOT))
  z.writestr('README.md','# Screw-in control laboratory\n\nRead screwdriver_lab/SCREW_IN.md for the task, limits and reproduction commands. Extract all media packs into this same folder for offline video playback. Serve screwdriver_site/dist with an HTTP server.\n')
 with zipfile.ZipFile(output) as z:assert z.testzip() is None
 assert output.stat().st_size<16*1024**2
 print('Source bundle',round(output.stat().st_size/1048576,2),'MB;',len(groups),'media packs',flush=True)
if __name__=='__main__':main()
