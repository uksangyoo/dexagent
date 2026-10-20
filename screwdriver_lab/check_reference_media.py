"""Decode all published RTX clips and audit sampled world transforms."""
import concurrent.futures,hashlib,json,subprocess
import numpy as np,mujoco
from screwdriver_lab.scene import ROOT
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.export_rtx import relocate_xml
SITE=ROOT/'screwdriver_site/dist'
def check(r):
 name=r['id'];path=SITE/r['video'];meta=json.loads((SITE/r['render_metadata']).read_text());bundle=OUT/'rtx'/name;scene=json.loads((bundle/'scene.json').read_text());trans=np.load(bundle/'scene.npz')['transforms'];raw=json.loads((OUT/(name+'.json')).read_text());q=np.load(OUT/(name+'.npz'))['qpos'];ts=np.array([f['t']for f in raw['frames']]);model=mujoco.MjModel.from_xml_string(relocate_xml((OUT/(name+'.xml')).read_text()));d=mujoco.MjData(model)
 assert meta['RTX'] and all('61:00'in x['id']for x in meta['devices'])
 assert meta['renderer_sha256']==hashlib.sha256((ROOT/'screwdriver_lab/render_reference.py').read_bytes()).hexdigest()
 assert meta['source_bundle_sha256']==hashlib.sha256((bundle/'scene.json').read_bytes()+(bundle/'scene.npz').read_bytes()).hexdigest()
 assert abs(scene['metadata']['frames'][-1]['t']-raw['final']['t'])<1e-8
 error=0.
 for k in sorted(set([0,len(trans)//2,len(trans)-1])):
  t=scene['metadata']['frames'][k]['t'];i=int(np.argmin(abs(ts-t)));d.qpos[:]=q[i];mujoco.mj_forward(model,d);error=max(error,float(np.abs(d.geom_xpos-trans[k,:,:3]).max()));assert error<1e-10
  for g in range(model.ngeom):
   quat=np.zeros(4);mujoco.mju_mat2Quat(quat,d.geom_xmat[g]);assert abs(abs(float(quat@trans[k,g,3:]))-1)<1e-9
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]));stream=info['streams'][0];assert [stream['width'],stream['height']]==[1280,960];duration=float(info['format']['duration']);assert abs(duration-(len(trans)/30+1))<.04
 subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 return dict(id=name,decoded=True,RTX=True,physical_GPU=1,duration_s=duration,source_frames=len(trans),sampled_transform_error_m=error,video_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':
 data=json.loads((SITE/'data/reference-results.json').read_text());records=data['results']+[data['full']];assert len(records)==10 and all('video'in r for r in records)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:results=list(pool.map(check,records))
 output=dict(passed=True,clips=len(results),checks=['all clips decode through FFmpeg','1280×960 RTX/OptiX on physical GPU 1','duration matches exported samples plus labeled end card','first/middle/final exported transforms exactly match recorded physics poses'],results=results)
 (ROOT/'screwdriver_site/qa/reference-media.json').write_text(json.dumps(output,indent=2));print('10 RTX clips passed decoding, duration, device and pose checks.')
