"""Build a compact, auditable inspector from actual agentic execution records."""
import json, shutil, zipfile, hashlib
from pathlib import Path
import numpy as np
from dexagent.agentic.contracts import ABSTRACTIONS
from dexagent.agentic.workflow import feedback, digest
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/agentic_workflow'
OUT=ROOT/'artifacts/hom_revision'
SITE=ROOT/'screwdriver_site/dist'
LABELS={'cartesian':'Fingertip paths','object':'Rigid object twist','hom':'Relative contact'}

def main():
    request=json.loads((ART/'goal-turn30/request.json').read_text());data={'request':request,'frames':[],'runs':[],'revisions':[]}
    for t,desc in [(1,'Lift'),(3,'Pivot and support'),(6,'Turn and reset')]:
        src=ART/f'goal-turn30/frame-{t:02d}.jpg';dst=SITE/f'assets/agentic-reference-{t}.jpg';shutil.copy2(src,dst)
        data['frames'].append(dict(time_s=t,description=desc,url=f'assets/{dst.name}'))
    files=set();snapshots={};validation=[]
    for method,version in [('cartesian',1),('object',1),('hom',1),('cartesian',2)]:
        name=f'agentic-{method}-turn30-v{version}';path=OUT/(name+'.json')
        if version==2 and not path.exists():continue
        r=json.loads(path.read_text());a=np.load(OUT/(name+'.npz'));s=a['samples'];bundle=ART/(method+'-policy'if version==1 else method+'-v2-policy')
        manifest=json.loads((bundle/'manifest.json').read_text());assert digest(bundle/'graph.json')==r['policy_sha256']==manifest['graph_sha256'];assert digest(bundle/'policy.py')==manifest['code_sha256']
        assert a['qref_500hz'].shape==a['vref_500hz'].shape==(len(s),26)
        assert np.isfinite(s).all()and np.isfinite(a['qref_500hz']).all()and np.isfinite(a['vref_500hz']).all()
        assert r['device_policy']['CUDA_VISIBLE_DEVICES']=='1'and r['full_task_success']is False
        # Independently recompute continuous acceptance from raw physical measurements.
        speed=np.abs(np.diff(s[:,2],prepend=s[0,2]))/.002
        ticks=r['agent_ticks'];accepted_hold=0.
        for k,row in enumerate(s):
            tick=max((x for x in ticks if x['observation']['elapsed_s']<=row[0]-.001),key=lambda x:x['observation']['elapsed_s'])
            holding=tick['decision'].get('action')=='hold'
            good=holding and abs(row[2]-30)<=5 and speed[k]<5 and min(row[4],row[6])>.05 and row[5]>.1 and row[13]<3
            accepted_hold=accepted_hold+.002 if good else 0.
        assert abs(accepted_hold-r['final']['settled_s'])<1e-8
        if r['success']:assert accepted_hold>=.3 and abs(s[-1,2]-30)<=5
        packet=feedback(path,ART/(name+'-feedback.json'));trace=[]
        for k in range(0,len(s),25):
            row=s[k];tick=max((x for x in ticks if x['observation']['elapsed_s']<=row[0]),key=lambda x:x['observation']['elapsed_s']);decision=tick['decision'];cmd=r['commands'][min(k//10,len(r['commands'])-1)]
            trace.append(dict(t=float(row[0]),screw_deg=float(row[2]),depth_mm=float(row[3]),loads=row[4:7].tolist(),tilt_deg=float(row[13]),qref=a['qref_500hz'][k].tolist(),vref=a['vref_500hz'][k].tolist(),effort=row[14:40].tolist(),node=decision['node'],operation=decision.get('action'),arguments=decision.get('arguments',{}),adapter_phase=cmd['phase']))
        video=ART/(name+'.mp4');assert video.exists(),video
        shutil.copy2(video,SITE/'assets'/video.name)
        shutil.copy2(video.with_suffix('.jpg'),SITE/'assets'/video.with_suffix('.jpg').name)
        outcome='Subgoal passed'if r['success']else'Overshoot; hold rejected'if method=='cartesian'else'Goal not reached within budget'
        item=dict(id=name,interface=method,label=LABELS[method],final=r['final'],outcome=r['outcome'],outcome_label=outcome,graph=r['agent_graph'],code=(bundle/'policy.py').read_text(),provenance=manifest['provenance'],contract=ABSTRACTIONS[method]['meaning'],transitions=r['agent_transitions'],feedback=packet,trace=trace,joints=r['actuator_layout']['hand']+r['actuator_layout']['arm'],video='assets/'+video.name,poster='assets/'+video.with_suffix('.jpg').name)
        data['runs'if version==1 else'revisions'].append(item)
        validation.append(dict(id=name,rows_500hz=len(s),motor_commands=26,finite=True,independent_hold_s=accepted_hold,outcome=r['outcome']))
        for ext in ['json','npz','xml']:files.add(OUT/(name+'.'+ext))
        files.add(ART/(name+'-feedback.json'))
        files.update(bundle.glob('*'))
        files.add(video.with_suffix('.render.json'))
        for source,sha in r['source_sha256'].items():snapshots[sha+'_'+Path(source).name]=OUT/'source_snapshots'/(sha+'_'+Path(source).name)
    # Keep the request immutable: provenance is a separate artifact.
    provenance=dict(original_reference=str(ROOT/'reference/screwdriver-five_hands-compressed.mp4'),original_sha256=digest(ROOT/'reference/screwdriver-five_hands-compressed.mp4'),input_video_sha256=request['video']['sha256'],derivation='Upper-left human panel cropped from supplied multiview reference; original playback speed.',goal_scope='Prepared-grasp validation subgoal, not the complete user task')
    (ART/'reference-provenance.json').write_text(json.dumps(provenance,indent=2));files.add(ART/'reference-provenance.json')
    (SITE/'data/agentic-workflow.json').write_text(json.dumps(data,separators=(',',':')))
    (ART/'verification.json').write_text(json.dumps(validation,indent=2));files.add(ART/'verification.json')
    files.add(ART/'goal-contract.json');files.update(ART.glob('*candidate.json'));files.update((ART/'goal-turn30').glob('*'));files.update(Path(__file__).parent.glob('*.py'));files.add(Path(__file__).parent/'README.md')
    with zipfile.ZipFile(SITE/'assets/agentic-workflow.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
        for name,p in sorted(snapshots.items()):z.write(p,'source_snapshots/'+name)
    print(json.dumps({'verification':validation,'zip_MiB':(SITE/'assets/agentic-workflow.zip').stat().st_size/2**20,'dist_MiB':sum(p.stat().st_size for p in SITE.rglob('*')if p.is_file())/2**20},indent=2))

if __name__=='__main__':main()
