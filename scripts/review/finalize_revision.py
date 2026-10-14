"""Publish measured free-body evidence, preserving the historical archive."""
import json,shutil,datetime,hashlib
from pathlib import Path
from scripts.review.export_data import progress
ROOT=Path(__file__).resolve().parents[2];run=ROOT/'runs/physics_revision_20261012';dist=ROOT/'review_site/dist'
ids=['nut_hold_validity','gap_repaired_transfer','screw_hold_validity','wrench_hold_passive','release_negative','rotate_x_integral','wrench_rotate_adaptive']
labels=['Nut · 3-second hold','GaP · support transfer','Screw · 3-second hold','Wrench · 3-second hold','Nut · deliberate release','Nut · 30° rotation','Wrench · 30° rotation']
explanations=['All hold gates passed','Receiver confirmed before donor release; 1-second hold','All hold gates passed','All hold gates passed, with fixed loaded servo targets','Falls to workbench; no hidden supporting joint','Orientation error and residual motion','Force guard stopped execution']
new=[];rows=[]
for name,label,why in zip(ids,labels,explanations,strict=True):
    meta=json.loads((run/'rendered/data'/f'{name}.json').read_text())
    meta['intrinsic_speed']=1
    if name=='gap_repaired_transfer':meta['tag']='FREE BODY · GRAPH PASSED'
    if meta['result']['task']=='release':meta['tag']='FREE BODY · GRAVITY CONTROL'
    result=meta['result'];quality=result.get('quality',{})
    meta['rendering']['display_changes']='Finger colors, highlighted object face, lighting and a display-only floor below the physical workbench. Optional white markers/red arrows are recorded Newton contact forces (1 cm/N). Poses are unmodified simulation states.'
    for view in meta['views'].values():
        mp4=view['video'];view.update(mp4=mp4,video=mp4.replace('.mp4','.webm'))
    for file in (run/'rendered/assets').glob(f'{name}_*'):
        shutil.copyfile(file,dist/'assets'/file.name)
    shutil.copyfile(run/f'{name}.npz',dist/'data'/f'{name}_states.npz')
    (dist/'data'/f'{name}.json').write_text(json.dumps(meta))
    new.append(meta)
    maximum=quality.get('max_depth_m',max(f['penetration_mm'] for f in meta['frames'])/1000)
    rows.append(dict(label=label,success=bool(result.get('success')),negative_control=name=='release_negative',orientation_error_deg=None if name=='release_negative' else result['final']['angle_error_deg'],peak_penetration_mm=maximum*1000,reason=why))
for file in (run/'rendered/assets').glob('*_hull.jpg'):shutil.copyfile(file,dist/'assets'/file.name)
for file in (run/'rendered/assets').glob('*_surface.jpg'):shutil.copyfile(file,dist/'assets'/file.name)
old=json.loads((dist/'data/snapshot.js').read_text().split(' = ',1)[1].rstrip(';\n'))
legacy=[c for c in old['clips'] if c['id'] not in ids and c.get('study_stage')!='controller_diagnostic_not_agent_ablation']
ledger=[]
for p in sorted(run.glob('*_result.json')):
    value=json.loads(p.read_text());ledger.append({'run_id':p.stem.removesuffix('_result'),'result':value})
revision=dict(captured_at=datetime.datetime.now(datetime.UTC).isoformat(),trials=len(ledger),results=rows,scope='Controller development diagnostics; no LLM agent comparison or industrial success-rate claim.',all_results='data/development_ledger.json')
state=progress();(dist/'data/progress.json').write_text(json.dumps(state))
(dist/'data/physics_revision.json').write_text(json.dumps(revision));(dist/'data/development_ledger.json').write_text(json.dumps(ledger))
(dist/'data/snapshot.js').write_text('window.REVIEW_DATA = '+json.dumps(dict(progress=state,evaluation=old['evaluation'],clips=new+legacy,revision=revision))+';\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'dexagent').rglob('*.py'))+list((ROOT/'scripts/physics').glob('*.py'))}
(run/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(clips=len(new+legacy),new_clips=len(new),trials=len(ledger))))
