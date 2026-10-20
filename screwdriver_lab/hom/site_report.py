"""Build the website's evidence from completed runs; never synthesize scores."""
import json, shutil
import numpy as np
from pathlib import Path
from screwdriver_lab.hom.model import OUT, ROOT
from screwdriver_lab.hom.metrics import summarize

DIST=ROOT/'screwdriver_site/dist'
LABELS=dict(cartesian='Fingertip paths',object='Rigid object twist',hom='Relative contact motion')


def build():
    data=DIST/'data';assets=DIST/'assets';data.mkdir(exist_ok=True)
    refined=all((OUT/(x+'-refined22-nominal.json')).exists() for x in LABELS)
    names=[x+'-refined22-nominal'for x in LABELS]if refined else['cartesian-slim-dev05','object-slim-dev06','hom-slim-dev04']
    rows=[]
    for name in names:
        row=summarize(name);row['label']=LABELS[row['interface']];row['cycles']=row['useful_loaded_sliding_episodes']
        stem=name+'-native25'if name=='object-slim-dev06'else name
        for suffix,source_stem in [('mp4',stem),('jpg',name)]:
            source=OUT/'rtx'/f'{source_stem}.{suffix}'
            if source.exists()and(OUT/'rtx'/f'{stem}.render.json').exists():shutil.copy2(source,assets/f'{name}.{suffix}')
        row['video']=f'assets/{name}.mp4';row['poster']=f'assets/{name}.jpg';rows.append(row)
    cases=[]
    for iface in LABELS:
        for condition,gap in [('near',.05),('nominal',.15),('far',.3)]:
            name=iface+'-refined22-'+condition
            if not (OUT/(name+'.json')).exists():continue
            row=summarize(name);row['label']=LABELS[iface];row['condition']=condition;row['gap_mm']=gap;cases.append(row)
    development=[]
    changes={
        'hom-slim-hybrid14':'Tangential-force cap; all contact angular coordinates released',
        'hom-slim-spin18':'Constrain relative contact spin; lower cap load',
        'hom-slim-spin19':'Stronger finger null-space servo; higher driving load',
        'hom-slim-spin20':'Executable graph and faster stroke command',
    }
    for name,change in changes.items():
        if (OUT/(name+'.json')).exists():
            r=summarize(name);r.update(change=change+'; excluded because the old wrist capsule overlaps the thumb base',success=False,outcome='invalid_initial_geometry');development.append(r)
    pad=json.loads((OUT/'pad-pressure-convergence.json').read_text())
    pad_resolutions=[dict(records=[r for r in pad['records']if r['resolution_m']==h])for h in [.002,.001]]
    benches=[]
    for path in sorted(OUT.glob('thread-validation-filtered*.json')):
        r=json.loads(path.read_text());r['file']=path.name;benches.append(r);shutil.copy2(path,data/path.name)
    for name in ['hydrostatic-validation.json','pad-pressure-convergence.json']:
        shutil.copy2(OUT/name,data/name)
    result=dict(
        status=dict(title='154° of screw rotation through loaded sliding.',detail='Relative-contact control advances 0.434 mm in the nominal six-second trial. The other interfaces advance 0.116 and 0.079 mm. These are prepared-grasp results; the complete pickup-to-seating task remains unverified.'),
        protocol=('Frozen 6-second comparison: identical prepared pose, 500 Hz motor control, 0.25 ms physics steps, 48 helical sectors per turn, explicit steel contact stiffness, and the same pressure and stroke settings. Three initial fingertip clearances test acquisition sensitivity. No pickup credit.' if refined else 'Early 4-second diagnostic pilot. Same starting pose, motor limits, contact geometry and parameters; only the controller interface changes. These coarse-integration results are provisional and do not establish a physically validated ranking.'),
        conclusion='Relative-contact control produces 3.9× the nominal screw rotation of fingertip paths and 5.9× that of rigid object twist. Fingertip paths initially turn the screw, then backdrive during resets. The relative-contact interface preserves more forward progress across all three clearances. This ranks these fixed implementations under the declared model, not every controller in each abstraction family.',
        duration_s=6 if refined else 4,comparison_revision='refined22'if refined else'coarse-pilot',matched=rows,conditions=cases,
        development_summary='These runs show development history, including stalls and reversals. Changes in gains, pressure targets and numerical settings make them unsuitable for an abstraction ranking.',development=development,
        pad_resolutions=pad_resolutions,thread_validation=benches,
        validation=[
          dict(check='Initial contact audit',evidence='No initial fingertip–tool hydroelastic overlap; residual non-task overlap below 1 μm. Same-rigid-assembly contacts excluded.',interpretation='The old wrist capsule was replaced by the supplied CAD envelope. Earlier trials with the overlapping capsule are excluded.'),
          dict(check='Robot import',evidence='Body mass, inertia and initial poses audited. Actuated mass matrices agree to 8.9×10⁻¹⁶.',interpretation='Correct conversion does not calibrate the source CAD inertias or motor armature values.'),
          dict(check='Fingertip pressure geometry',evidence='Watertight tetrahedral meshes; coarse/fine volume errors 0.160% / 0.047%; sampled surface deviations ≤66 / 33 μm.',interpretation='Geometry approximation is measured. This is not a certified Hausdorff bound or a pad thickness measurement.'),
          dict(check='Actual-pad force convergence',evidence='Refinement changes force by 10.2% at about 0.05 N, 4.3% at 0.22 N, and 0.6% near 0.9 N.',interpretation='Low-load slipping is less resolved than firm contact. Material calibration remains absent.'),
          dict(check='Free fastener benchmark',evidence='96 sectors/turn: 443.87° at 0.125 ms versus 443.70° at 0.0625 ms; insertion differs by 0.57 μm.',interpretation='Independent external-wrench bench only. The 48-sector / 0.25 ms configuration differs by 3.6% in angle from the finest case. Coarse 1 ms cases fail this check.'),
          dict(check='Actuation boundary',evidence='26 bounded motor efforts. Driver and screw are unactuated free bodies; no pitch constraint or pose correction.',interpretation='The swivel cap has a physical bearing joint. Robot gravity compensation passes through the same effort limits.'),
          dict(check='Model uncertainty',evidence='10 mm handle; 1.7 MPa hydroelastic pad modulus; friction 0.6 at the grip, 0.15 at threads.',interpretation='Declared assumptions, not measured materials. No claim of calibrated silicone deformation, hardware transfer, or Newton task success.'),
        ])
    checks=[]
    for name,change in [('cartesian-timestep-check','Fingertip paths: physics step halved'),('object-timestep-check','Rigid object twist: physics step halved'),('hom-timestep-check','Relative contact: physics step halved'),('hom-padmesh-check','Finer pad pressure mesh: 1 mm setting'),('hom-friction-low','Grip friction 0.4, controller unchanged'),('hom-friction-high','Grip friction 0.8, controller unchanged')]:
        if not (OUT/(name+'.json')).exists():continue
        r=summarize(name);r['change']=change
        baseline=np.load(OUT/(r['interface']+'-refined22-nominal.npz'))['samples'];idx=np.argmin(abs(baseline[:,0]-r['duration_s']))
        r['baseline_screw_deg_at_same_time']=float(baseline[idx,2]);checks.append(r)
    result['sensitivity']=checks
    extended_name='hom-seat-stop42'
    extended=OUT/(extended_name+'.json')
    if extended.exists():
        r=summarize(extended_name);full=json.loads(extended.read_text())
        r['manipulation_success']=full.get('manipulation_success',False);r['seating_hold_s']=full.get('seating_hold_s',0)
        r['checkpoint']=full['checkpoint'];r['final_seating_geometry']=full['final_seating_geometry']
        r['video_ready']=(OUT/'rtx'/(extended_name+'.render.json')).exists()
        r['video']=f'assets/{extended_name}.mp4';r['poster']=f'assets/{extended_name}.jpg'
        r['observations']=f'assets/{extended_name}-observations.npz'
        if r['video_ready']:
            for suffix in ['mp4','jpg','render.json']:shutil.copy2(OUT/'rtx'/(extended_name+'.'+suffix),assets/(extended_name+'.'+suffix))
        result['extended']=r
        result['stop_development']=[]
        for name in ['hom-seating23','hom-seat-stop40','hom-seat-stop41','hom-seat-stop42']:
            if not (OUT/(name+'.json')).exists():continue
            trial=json.loads((OUT/(name+'.json')).read_text())
            result['stop_development'].append(dict(id=name,outcome=trial['outcome'],final=trial['final'],hold_s=trial.get('seating_hold_s',0),parameters=trial['parameters']))
    if (OUT/'rtx/pickup-trial36.render.json').exists():
        for suffix in ['mp4','jpg','render.json']:shutil.copy2(OUT/'rtx'/('pickup-trial36.'+suffix),assets/('pickup-trial36.'+suffix))
        pickup=json.loads((OUT/'pickup-trial36.json').read_text());samples=np.load(OUT/'pickup-trial36.npz')['samples'];loaded=np.minimum(samples[:,4],samples[:,6])>.1
        result['pickup_development']=dict(id='pickup-trial36',outcome=pickup['outcome'],video='assets/pickup-trial36.mp4',poster='assets/pickup-trial36.jpg',closest_loaded_tilt_deg=float(min(samples[loaded,13])),loaded_pivot_deg=float(90-min(samples[loaded,13])),sample_timing_note='Prototype observations precede replay states by one 2 ms motor interval; rounded descriptive values only, not an accepted comparison.')
    result['sensitivity_note']='The policy and motor limits are frozen. Friction changes only in the plant, so the controller must tolerate a model mismatch. The fine-pad check is two seconds; it is compared with the nominal run at the same time. These tests do not establish calibrated material parameters or timestep convergence of the full task.'
    if (OUT/'corrected-scene-audit.json').exists():shutil.copy2(OUT/'corrected-scene-audit.json',data/'corrected-scene-audit.json')
    if refined:
        graph=json.loads((OUT/'hom-refined22-nominal.json').read_text())['graph'];(data/'hom-policy-graph.json').write_text(json.dumps(graph,indent=2))
    (data/'hom-results.json').write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps(dict(comparison=result['comparison_revision'],conditions=len(cases),development=len(development))))


if __name__=='__main__':build()
