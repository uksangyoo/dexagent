"""Publish measured physical task evidence; never promote a timeout to success."""
import csv,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from screwdriver_lab.physical_task import OUT
from screwdriver_lab.physical_evidence import join
SITE=Path(__file__).resolve().parents[1]/'screwdriver_site';DIST=SITE/'dist'

def read(name):return json.loads((OUT/(name+'.json')).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_raw(r):
    a=np.load(OUT/(r['id']+'.npz'));x=a['samples_1khz'];cols=list(a['sample_columns']);v=lambda k:x[:,cols.index(k)]
    for i,finger in enumerate(['thumb','middle']):
        ret=v(finger+'_return')>0;load=v(finger+'_N')>.01;slip=v(finger+'_slip_mm_s')<-.1
        for key,actual in [('return_loaded_fraction',np.mean(load[ret])),('return_loaded_sliding_fraction',np.mean((load&slip)[ret]))]:
            assert abs(actual-r['metrics'][key][i])<1e-8,(r['id'],key,actual)
    assert np.allclose(np.diff(v('t')),.001,atol=1e-8)
    assert max(r['motor']['peak_motor_torque_Nm'][:20])<=.11+1e-12
    return dict(samples=len(x),max_penetration_mm=float(v('penetration_mm').max()),engaged_fraction=float(v('engaged').mean()),source_npz_sha256=sha(OUT/(r['id']+'.npz')))

def run():
    runs=[];audits={}
    for interface in ['pose','velocity','wrench']:
        for condition,label in [('y0','Nominal'),('ym2','Tool offset −2 mm'),('yp2','Tool offset +2 mm')]:
            r=read(f'matched-{interface}-{condition}');audits[r['id']]=validate_raw(r)
            prefix_record=read(r['prefix']);assert prefix_record['initial_y_mm']=={'y0':0,'ym2':-2,'yp2':2}[condition]
            r.update(condition=label,backend='MuJoCo CPU / explicit thread contact',scope='35 s drive from physically reached engagement')
            r['frames']=r['frames'][::3]+r['frames'][-1:]
            vid=OUT/'rtx'/(r['id']+'.mp4')
            close=OUT/'rtx'/('contact-'+r['id']+'.mp4')
            if close.exists()and close.with_suffix('.render.json').exists():vid=close
            if vid.exists()and vid.with_suffix('.render.json').exists():
                r['video']='assets/'+vid.name;r['poster']='assets/'+vid.with_suffix('.jpg').name
                r['video_note']='6× simulation time. '+('Close contact view.'if vid==close else'Overview with arm and opposing contact views.')
                for p in [vid,vid.with_suffix('.jpg')]:shutil.copyfile(p,DIST/'assets'/p.name)
                scene=json.loads((OUT/'rtx'/r['id']/'scene.json').read_text())
                r['video_segments']=scene['metadata']['presentation_segments']
            runs.append(r)
    full,_,_=join('contact-reflex-long-01');final=full['final'];lim=full['metrics']
    report=dict(status=dict(title='Physical seating achieved. Continuous sliding remains incomplete.',detail='The contact-driven hand turns the screw about 701° and seats it at 1.996 mm. The reset still loses pad contact too often to claim faithful human-style controlled slipping.'),
        protocol='Nine matched trials using the paired drive/reset graph: three low-level adapters × nominal and ±2 mm initial tool offsets. Each condition begins with a physical pickup and shares an exact saved engagement state across adapters. Pickup, cap support and the arm servo are shared; the adapter changes apply to the driving fingers. Same contact model, bounded motors, policy guards and 35 s drive budget. Adapter gains differ; this compares these implementations, not an inherent ranking of abstractions. The fixed horizon measures progress; none seats within that budget.',
        validation_summary='Independent soft-finger tests passed six load/timestep cases at the chosen solver settings. Explicit threads produce near-pitch advance with approximately 0.05–0.066 mm clearance/backlash residual. These checks do not calibrate the task to hardware. Full-task results on this page use CPU MuJoCo. Earlier Newton/MuJoCo-Warp trials on GPU 1 lost engagement or violated penetration limits. The new 4 kHz external-contact run reaches physical pickup and engagement; full GPU screw-in remains unverified. RTX rendering uses GPU 1.',
        hero=dict(backend='CPU PHYSICS / RTX REPLAY',caption='Physical pickup → in-hand pivot → cap support → engagement → finger-driven screw insertion. Pickup is shown at 2×; insertion at 30×. The exact-state join is verified; no pose correction is applied.',limit='This seating run is a partial task result: thumb / middle remain loaded during only 30.9% / 22.0% of the commanded returns. Its final stop retained the previous bounded task effort; the later controller explicitly switches to a seating hold. Neither material calibration nor robust human-like sliding is established.',video='assets/physical-seating-overview.mp4',poster='assets/physical-seating-overview.jpg',stats=[['700.6°','Final screw rotation'],['1.996 mm','Insertion / 2 mm target'],['0.23°','Final tool tilt'],['<0.013°','Palm rotation during drive']]),
        default_run='matched-wrench-y0',runs=runs,full_task={k:v for k,v in full.items()if k!='frames'},raw_audits=audits,
        findings=[dict(title='Wrench control advances farther',text='In the matched 35 s trials, wrench control reaches 135–151°, compared with 31–73° for contact-pose admittance and about 66–67° for velocity control. All retain engagement, but none seats within the budget.'),dict(title='Contact is the remaining bottleneck',text='The completed 183 s drive seats the screw, but its lightly loaded returns frequently separate. Alternating strokes can retain contact much longer; increased return friction can also cancel forward progress. Both metrics must improve together.'),dict(title='Initial conditions matter',text='The +2 mm initial offset reduces pose-admittance progress to 31°. Velocity control changes little across these three conditions. Three deterministic offsets do not establish broad robustness.')])
    refinement_names=[('alternating-axis-long-23','Axis feedback / nominal'),('alternating-axis-offset-minus-24','Axis feedback / −2 mm'),('alternating-axis-offset-plus-25','Axis feedback / +2 mm'),('alternating-oriented-minus-27','Pad orientation / −2 mm'),('alternating-oriented-nominal-28','Pad orientation / nominal'),('alternating-oriented-plus-29','Pad orientation / +2 mm'),('alternating-recovery-minus-30','Balance recovery / −2 mm'),('alternating-seat-plus-32','Contact seating / +2 mm'),('alternating-arm-recovery-minus-33','Arm + contact recovery / −2 mm'),('alternating-seat-nominal-34','Contact seating / nominal'),('alternating-seat-recovery-minus-35','Recovery + contact seating / −2 mm')]
    refinements=[]
    for name,label in refinement_names:
        if not (OUT/(name+'.json')).exists():continue
        rr=read(name);aa=np.load(OUT/(name+'.npz'));raw=aa['samples_1khz'];cols=list(aa['sample_columns'])
        seats=[v['t']for v in rr['cycle_trace']if v.get('to')=='seat'];end=seats[0]if seats else float('inf')
        from screwdriver_lab.slip_metrics import SlipMetrics
        accumulator=SlipMetrics()
        for row in raw[raw[:,0]<end]:
            x=dict(zip(cols,row));accumulator.add(dict(t=x['t'],screw_angle_deg=x['screw_angle_deg'],handle_forces_N=[x['thumb_N'],x['middle_N']],handle_slip_signed_mm_s=[x['thumb_slip_mm_s'],x['middle_slip_mm_s']]),['slide_return'if x[f+'_return']else'hold'for f in ['thumb','middle']])
        analysis_metrics=accumulator.report()
        refinements.append(dict(id=name,label=label,outcome=rr['outcome'],success=rr['success'],final=rr['final'],parameters=rr['parameters'],metrics=analysis_metrics,score_window=dict(start_s=float(raw[0,0]),end_s=min(float(raw[-1,0]),end),excludes_seating_hold=True),source_sha256=rr['source_sha256']))
    aborted=OUT/'alternating-arm-recovery-minus-33.aborted.json'
    if aborted.exists():
        rr=json.loads(aborted.read_text());pr=rr['progress']
        refinements.append(dict(id=rr['id'],label='Arm + contact recovery / −2 mm',outcome=rr['outcome'],success=False,final=pr['observation'],metrics=pr['metrics'],parameters={},note=rr['available_evidence']))
    report['refinements']=refinements
    report['refinement_note']='Alternating strokes with positive return load improve contact, but the long runs expose failures hidden by short trials. These are successive controller refinements, with different gains and up to 200–240 s budgets; compare them separately from the matched 35 s adapter study. Return metrics exclude the seating hold.'
    report['refinement_video']=dict(video='assets/loaded-slip-cycle.mp4',poster='assets/loaded-slip-cycle.jpg',caption='Eight seconds at real time, from the 50 s nominal alternating trial (simulation 43–51 s). Middle-finger negative relative slip accompanies forward handle rotation. This excerpt is not a completed seating run.')
    latest='alternating-seat-nominal-34'
    if (OUT/(latest+'.json')).exists():
        rr=read(latest);aa=read(latest+'-audit');full_latest,_,_=join(latest)
        if rr['success']:
            report['status']=dict(title='The simulated hand completes screw insertion.',detail='Contact-driven pickup leads to alternating loaded finger strokes and verified seating: 702° and 2.007 mm. Contact retention improves substantially, while brief dropouts and sensitivity to the starting grasp remain.')
            report['hero']=dict(backend='CPU PHYSICS / RTX REPLAY',video='assets/physical-task-success.mp4',poster='assets/physical-task-success.jpg',caption='Nominal start. Open grasp → in-hand pivot → index-cap support → bit engagement → alternating finger drive and sliding return → physical seating. Pickup: 2× playback. Insertion: 20× playback. Total simulated time: 153.4 s.',limit='Raw 1 kHz measurements: thumb / middle retain contact during 89.7% / 74.0% of returns; loaded sliding occurs during 72.2% / 49.4%. This is task completion with intermittent contact, not perfectly continuous sliding or matched human speed. Material and hardware calibration remain unverified.',stats=[['702.3°','Final screw rotation'],['2.007 mm','Insertion / 2 mm target'],['89.7% / 74.0%','Loaded return · thumb / middle'],['0.11 N·m','Finger motor limit']])
            report['full_task']={k:v for k,v in full_latest.items()if k!='frames'}
            report['full_task_audit']=aa
            load_distribution=SITE/'qa/physical-return-load-distribution.json'
            if load_distribution.exists():report['return_load_distribution']=json.loads(load_distribution.read_text())
            report['refinement_video']=dict(video='assets/successful-slip-cycle.mp4',poster='assets/successful-slip-cycle.jpg',caption='One full alternating cycle at real time (simulation 29.96–43.20 s), from the successful nominal orientation-feedback run. Both fingers take a drive stroke and a sliding return. This run uses the same pre-seating control as the latest nominal completion.')
            for stem in ['physical-task-success','successful-slip-cycle']:
                path=OUT/'rtx'/(stem+'.mp4')
                if path.exists()and path.with_suffix('.render.json').exists():
                    for ext in ['.mp4','.jpg']:shutil.copyfile(path.with_suffix(ext),DIST/'assets'/(stem+ext))
            close_cycle=OUT/'rtx/successful-slip-closeup.mp4'
            if close_cycle.exists()and close_cycle.with_suffix('.render.json').exists():
                for ext in ['.mp4','.jpg']:shutil.copyfile(close_cycle.with_suffix(ext),DIST/'assets'/close_cycle.with_suffix(ext).name)
                report['refinement_video']['video']='assets/successful-slip-closeup.mp4'
                report['refinement_video']['poster']='assets/successful-slip-closeup.jpg'
            shutil.copyfile(OUT/(latest+'.npz'),DIST/'assets/physical-task-success-raw.npz')
            report['findings'][1]=dict(title='Loaded sliding now completes the task',text='The refined nominal controller seats the screw after about 138 s of driving and final holding. It retains contact during 89.7% / 74.0% of thumb / middle returns. Short contact losses remain; the real-time clip and raw measurements show them.')
            report['findings'].append(dict(title='Long runs expose hidden failures',text='Axis feedback alone fails on the nominal and −2 mm starts despite promising short trials. Pad-orientation feedback improves the nominal run; balancing grip forces during tilt recovery lets the −2 mm run finish. These are condition-specific refinements, not a single frozen controller with demonstrated robustness.'))
    (DIST/'data/physical-task-results.json.gz').write_bytes(gzip.compress(json.dumps(report,separators=(',',':')).encode()))
    with (DIST/'data/physical-task-summary.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['id','interface','condition','outcome','rotation_deg','depth_mm','thumb_return_loaded','middle_return_loaded','thumb_loaded_sliding','middle_loaded_sliding','backward_rotation_deg','max_penetration_mm','protocol','drive_duration_s','parameters_json'])
        for r in runs:w.writerow([r['id'],r['interface'],r['condition'],r['outcome'],r['final']['screw_angle_deg'],r['final']['depth_mm'],*r['metrics']['return_loaded_fraction'],*r['metrics']['return_loaded_sliding_fraction'],r['metrics']['backward_rotation_deg'],audits[r['id']]['max_penetration_mm'],'matched_35s',r['metrics']['drive_duration_s'],json.dumps(r['parameters'],sort_keys=True)])
        for r in refinements:w.writerow([r['id'],'wrench',r['label'],r['outcome'],r['final']['screw_angle_deg'],r['final']['depth_mm'],*r['metrics']['return_loaded_fraction'],*r['metrics']['return_loaded_sliding_fraction'],r['metrics']['backward_rotation_deg'],'','controller_refinement',r['metrics']['drive_duration_s'],json.dumps(r['parameters'],sort_keys=True)])
    (SITE/'qa/physical-raw-audit.json').write_text(json.dumps(audits,indent=2))
    print('Validated',len(runs),'matched runs;',sum(a['samples']for a in audits.values()),'raw 1 kHz samples')

if __name__=='__main__':run()
