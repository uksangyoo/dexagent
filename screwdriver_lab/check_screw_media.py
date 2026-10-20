"""Decode every published screw-in clip and verify dimensions/frame accounting."""
import concurrent.futures,json,subprocess
from screwdriver_lab.screw_scene import ROOT

def check(r):
 path=ROOT/'screwdriver_site/dist'/r['video']
 subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','null','-'],check=True,capture_output=True)
 meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
 assert (meta['width'],meta['height'])==(1280,960)
 assert int(meta['nb_frames'])==len(r['frames'])+30
 assert meta['r_frame_rate']=='30/1'
 assert len(r['rendering']['devices'])==1 and '61:00' in r['rendering']['devices'][0]['id']
 return dict(id=r['id'],frames=int(meta['nb_frames']),decoded=True,RTX_GPU_1=True)
if __name__=='__main__':
 d=json.loads((ROOT/'screwdriver_site/dist/data/screw-in.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:results=list(p.map(check,d['results']))
 (ROOT/'screwdriver_site/qa/screw-media.json').write_text(json.dumps(dict(passed=True,results=results),indent=2));print('Decoded and verified',len(results),'RTX clips')
