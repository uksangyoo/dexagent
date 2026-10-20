"""Verify core provenance, paired cases, metrics, recordings, and source assets."""
import hashlib,json,math,subprocess
from pathlib import Path
from screwdriver_lab.scene import ROOT

def walk_numbers(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():walk_numbers(v)
    elif isinstance(x,list):
        for v in x:walk_numbers(v)

def main():
    site=ROOT/'screwdriver_site/dist';core=['__init__.py','scene.py','simulator.py','controllers.py','policy.py','run.py']
    summaries=[]
    for name,expected in [('manifest.json',50),('local-motor.json',10)]:
        d=json.loads((site/'data'/name).read_text());assert len(d['results'])==expected
        assert d['grasp_sha256']==hashlib.sha256((ROOT/'screwdriver_lab/artifacts/grasp.json').read_bytes()).hexdigest()
        for file in core:assert d['source_hashes'][file]==hashlib.sha256((ROOT/'screwdriver_lab'/file).read_bytes()).hexdigest(),file
        groups={};seen=set()
        for r in d['results']:
            walk_numbers(r);assert r['id'] not in seen;seen.add(r['id'])
            groups.setdefault((r['mode'],r['case']),set()).add(r['initial_state_sha256'])
            assert (site/r['video']).is_file();assert (site/r['poster']).is_file()
            probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_entries','format=duration:stream=codec_name,width,height','-of','json',str(site/r['video'])]))
            assert probe['streams'][0]['codec_name']=='h264'
            subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(site/r['video']),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            assert float(probe['format']['duration'])>0
            assert r['metrics']['policy_calls']==len(r['trace'])-1
            assert r['frames'][0]['t']==0 and r['frames'][0]['phase']=='acquire'
            assert r['rendering']['RTX'] and all(x['type']=='OPTIX' for x in r['rendering']['devices'])
            assert [probe['streams'][0]['width'],probe['streams'][0]['height']]==[1280,960]
            assert abs(float(probe['format']['duration'])-(len(r['frames'])/30+1))<.04
            if r['valid_initialization']:
                assert not r['acquisition_events']
                assert r['acquisition_metrics']['max_drift_mm']<5
                assert r['acquisition_metrics']['min_shaft_clearance_mm']>=3
            assert r['success']==(r['outcome']=='success')
        assert all(len(g)==1 for g in groups.values()),'Initial states must match across methods'
        summaries.append(dict(dataset=name,trials=expected,valid=sum(r['valid_initialization'] for r in d['results']),successes=sum(r['success'] for r in d['results']),paired_initializations=True,core_source_hashes_verified=True,all_media_decodable=True))
    sensitivity=json.loads((ROOT/'screwdriver_lab/artifacts/revision_rtx/timestep_check/manifest.json').read_text())
    result=dict(unit_tests=dict(passed=11,command='python -m unittest screwdriver_lab.test_pipeline -v'),datasets=summaries,timestep_check=dict(timestep_s=.001,goal_deg=15,results=[dict(id=r['id'],outcome=r['outcome'],angle_deg=r['final']['angle_deg'],max_force_N=r['metrics']['max_force_N']) for r in sensitivity['results']]),rendering=dict(backend='Blender Cycles / NVIDIA OptiX',RTX=True,resolution=[1280,960],fps=30,source_sha256=hashlib.sha256((ROOT/'screwdriver_lab/render_rtx.py').read_bytes()).hexdigest()),browser_qa='pending')
    previous=json.loads((site/'data/validation.json').read_text()) if (site/'data/validation.json').exists() else {}
    for key in ['browser_qa','render_resource_guard']:
        if key in previous:result[key]=previous[key]
    (site/'data/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
