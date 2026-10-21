"""Publishable evidence from completed runs only; never synthesize outcomes."""
import json, shutil, hashlib, zipfile, gzip
from pathlib import Path
import numpy as np
from dexagent.planning.geometry import ART, ROOT
from tasks.screwdriver.sequence.audit import audit
from dexagent.planning.policy import INTERFACES
from dexagent.sim.model import OUT
from dexagent.provenance import recorded_sha256

SITE=ROOT/'screwdriver_site/dist'

def record(name, task):
    path=OUT/(name+'.json')
    if not path.exists(): return None
    r=json.loads(path.read_text());a=np.load(OUT/(name+'.npz'));s=a['samples'];frames=[]
    for k,t in enumerate(a['time']):
        j=int(np.argmin(abs(s[:,0]-t)));v=s[j]
        frames.append(dict(t=float(t),phase=r['commands'][k]['phase'],tilt=float(v[13]),loads=v[[4,5,6]].tolist(),
            motor=v[14:40].tolist(),unexpected=float(v[40])if len(v)>40 else None,
            cap_top=float(v[41])if len(v)>41 else None,rotation_error=float(v[44])if len(v)>44 else None,
            screw_deg=float(v[2]),depth_mm=float(v[3])))
    video=ART/(name+'.mp4');poster=ART/(name+'.jpg')
    overview=ART/(name+'-overview.mp4')
    if name=='sequence-assembly-v2'and overview.exists():video=overview
    for file in [video,poster,overview]:
        if file.suffix=='.mp4'and not file.with_suffix('.render.json').exists():continue
        if file.exists(): shutil.copy2(file,SITE/'assets'/file.name)
    metrics=audit(name)
    policy=r.get('policy_graph');code=None
    if policy:
        expected=(r.get('policy_provenance')or{}).get('code_sha256')
        source=next((p for p in (ART/'policies').rglob('policy.py')if hashlib.sha256(p.read_bytes()).hexdigest()==expected),None)
        code=source.read_text()if source else None
    return dict(id=name,task=task,interface=r['interface'],outcome=r['outcome'],
        subgoal_success=r.get('subgoal_success',r.get('pickup_success',False)),
        full_task_success=r.get('success',False),checkpoint=r.get('checkpoint'),metrics=metrics,frames=frames,
        maximum_screw_turn_deg=float(max(s[:,2])),
        final_orientation_error_deg=float(s[-1,44])if s.shape[1]>44 else None,
        accepted_hold_s=r.get('lift_hold_s',0)if task=='lift_hold'else r.get('support_hold_s',0),
        video_excerpt=json.loads((ART/(name+'-video-scope.json')).read_text())if (ART/(name+'-video-scope.json')).exists()else None,video='assets/'+video.name if video.exists()and video.with_suffix('.render.json').exists()else None,poster='assets/'+poster.name if poster.exists()else None,
        overview='assets/'+overview.name if overview.exists()and overview.with_suffix('.render.json').exists()else None,graph=policy,code=code,transitions=r.get('agent_transitions',r['trace']),
        assembly=r.get('assembly_final'),assembly_transitions=r.get('assembly_trace',[]),parameters=r['parameters'],xml_sha256=recorded_sha256(OUT/(name+'.xml')))

def main():
    SITE.joinpath('data').mkdir(exist_ok=True)
    runs=[]
    for task,prefix in [('lift_hold','sequence-lift'),('controlled_pivot','sequence-pivot')]:
        for interface in INTERFACES:
            item=record(f'{prefix}-{interface}-v1',task)
            if item:runs.append(item)
    pickup=record('sequence-pivot-v17','controlled_pivot')
    assembly_history=[record(name,'pickup_pivot_drive')for name in ['sequence-assembly-v1','sequence-assembly-v2','sequence-clear07-hom-v1','sequence-offset20-hom-s48-v2','sequence-offset20-hom-s96-v2','sequence-offset20-hom-s48-v3','sequence-offset20-hom-s48-v4','sequence-offset20-hom-s96-v4','sequence-start02-hom-full-v8','sequence-start02-hom-full-v9','sequence-start02-hom-full-v13']if (OUT/(name+'.json')).exists()]
    assembly=assembly_history[-1]if assembly_history else None
    drive_diagnostics=[record(name,'prepared_drive')for name in ['sequence-offset20-insert-debug-v5','sequence-offset20-drive-paired-v6','sequence-offset20-drive-ramped-v7','sequence-offset20-drive-relative-v8','sequence-offset20-drive-loaded-v9','sequence-offset20-drive-s96dt125-v8','sequence-offset20-workspace-drive-v10','sequence-offset20-drive-impedance-v11','sequence-offset20-anchored-hold-v12','sequence-offset20-filtered-s96dt125-v12','sequence-offset20-posture-hold-v13']if (OUT/(name+'.json')).exists()]
    geometry=json.loads((ART/'geometry.json').read_text())
    shutil.copy2(ART/'geometry-inspection.jpg',SITE/'assets/sequence-geometry.jpg')
    history=[]
    for k in range(1,19):
        if (OUT/f'sequence-pivot-v{k}.json').exists():history.append(audit(f'sequence-pivot-v{k}'))
    benches=[]
    for file in sorted(ART.glob('thread-bench-slow-s*.json')):
        r=json.loads(file.read_text());benches.append(dict(id=file.stem,**{k:r[k]for k in ['dt','rotation_deg','depth_mm','max_tilt_deg']}))
    passive_benches=[dict(id=p.stem,**json.loads(p.read_text()))for p in ART.glob('thread-bench-passive-*.json')]
    result=dict(passive_thread_benches=passive_benches,geometry=geometry,interfaces=INTERFACES,pickup=pickup,assembly=assembly,assembly_history=assembly_history,drive_diagnostics=drive_diagnostics,runs=runs,history=history,thread_benches=benches,
        comparison_scope='One authored graph per task and interface, nominal initial condition. Not an estimate of general agent ability.',
        authoring='Policies authored by Codex in this conversation. Generated Python executes at 20 Hz; no separate runtime model invocation.',
        physics='CPU Drake SAP; hydroelastic pads; bounded 20 hand + 6 arm motors. GPU 1 is used for RTX rendering.',
        initial_condition='Horizontal tool on two physical supports, with fingers initially clear. This is not flat-table pickup.',
        full_sequence_success=any(r['full_task_success']for r in assembly_history),
        pickup_contact_motion=json.loads((ART/'pivot-contact-motion-summary.json').read_text())if (ART/'pivot-contact-motion-summary.json').exists()else None)
    with gzip.open(SITE/'data/sequence-results.json.gz','wt')as f:json.dump(result,f,separators=(',',':'))
    (SITE/'data/sequence-results.json').unlink(missing_ok=True)
    (ART/'site-results.json').write_text(json.dumps(result,indent=2))
    bundle=SITE/'assets/sequence-evidence.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for file in [*(ROOT/'dexagent/planning').glob('*.py'),*(ROOT/'tasks/screwdriver/sequence').glob('*.py')]:z.write(file,str(file.relative_to(ROOT)))
        for file in [*(ROOT/'dexagent/sim').glob('*.py'),*(ROOT/'tasks/screwdriver/study').glob('*.py')]:z.write(file,str(file.relative_to(ROOT)))
        source_records=[pickup]+runs+assembly_history+drive_diagnostics
        needed={digest+'_'+name for r in source_records if r for name,digest in json.loads((OUT/(r['id']+'.json')).read_text()).get('source_sha256',{}).items()}
        for filename in sorted(needed):
            file=OUT/'source_snapshots'/filename
            if file.exists():z.write(file,'source_snapshots/'+filename)
        for file in (ART/'policies').rglob('*'):
            if file.is_file()and '__pycache__'not in str(file):z.write(file,str(file.relative_to(ROOT)))
        for name in ['sequence-pivot-v17']+[r['id']for r in runs]+[r['id']for r in assembly_history+drive_diagnostics]:
            for suffix in ['.json','.xml']:
                p=OUT/(name+suffix);z.write(p,'records/'+p.name)
        z.write(OUT/'sequence-pivot-v17.npz','records/sequence-pivot-v17.npz')
        for file in [ART/'geometry.json',ROOT/'tasks/screwdriver/sequence/tasks.json']:z.write(file,file.name)
        for file in ART.glob('thread-bench-*.json'):z.write(file,'numerics/'+file.name)
        for file in OUT.glob('sequence-pivot-v17-contact-motion.json'):z.write(file,'contact_replay/'+file.name)
        if (ART/'pivot-contact-motion-summary.json').exists():z.write(ART/'pivot-contact-motion-summary.json','contact_replay/summary.json')
        z.writestr('READ-ME.txt','Continuous pickup/pivot development and nominal two-task controller comparison.\nThe raw v17 recording is included; other records include reports, source hashes and graphs.\nNo full assembly success is implied by a pickup or pivot result.\nHistorical prepared-grasp comparisons use a different model.\n')
    print(json.dumps(dict(runs=len(runs),full_sequence_success=result['full_sequence_success'],evidence_bytes=bundle.stat().st_size,
        total_site_bytes=sum(p.stat().st_size for p in SITE.rglob('*')if p.is_file()))))

if __name__=='__main__':main()
