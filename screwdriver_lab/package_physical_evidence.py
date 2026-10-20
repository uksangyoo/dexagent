"""Small website reproduction archive; full development traces remain local."""
from pathlib import Path
import json,zipfile
from screwdriver_lab.physical_task import OUT
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'screwdriver_site'
def package():
    names=['contact-reflex-long-01','physical-engagement-prefix']+[f'matched-{i}-{c}'for i in ['pose','velocity','wrench']for c in ['y0','ym2','yp2']]+[f'matched-prefix-{c}'for c in ['y0','ym2','yp2']]
    extra=['alternating-axis-long-23','alternating-axis-offset-minus-24','alternating-axis-offset-plus-25','alternating-oriented-minus-27','alternating-oriented-nominal-28','alternating-oriented-plus-29','alternating-recovery-minus-30','alternating-seat-plus-32','alternating-seat-nominal-34','alternating-seat-recovery-minus-35']
    names +=[n for n in extra if (OUT/(n+'.json')).exists()]
    zpath=SITE/'dist/assets/physical-task-code.zip';digests=set()
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for p in (ROOT/'screwdriver_lab').glob('*.py'):z.write(p,'screwdriver_lab/'+p.name)
        for folder in ['controlled_slip_tip','reference_strategy']:
            p=ROOT/'screwdriver_lab/artifacts'/folder/'grasp.json'
            if p.exists():z.write(p,str(p.relative_to(ROOT)))
        import importlib.metadata as metadata,sys
        versions={name:metadata.version(name)for name in ['mujoco','numpy','scipy','warp-lang','newton','trimesh','Pillow']}
        z.writestr('ENVIRONMENT.json',json.dumps(dict(python=sys.version,packages=versions),indent=2))
        z.write(ROOT/'screwdriver_lab/PHYSICAL_TASK.txt','REPRODUCE.txt')
        z.write(OUT/'matched-wrench-y0.xml','evidence/physical-model.xml')
        for n in names:
            r=json.loads((OUT/(n+'.json')).read_text())
            for key in ['source_sha256','physical_source_sha256','source_hashes']:
                if isinstance(r.get(key),dict):digests.update(r[key].values())
            r.pop('frames',None)
            z.writestr('evidence/'+n+'.json',json.dumps(r,indent=2))
        aborted=OUT/'alternating-arm-recovery-minus-33.aborted.json'
        if aborted.exists():z.write(aborted,'evidence/'+aborted.name)
        for p in (OUT/'source_snapshots').glob('*.py'):
            if p.name.split('_')[0]in digests:z.write(p,'evidence/source_snapshots/'+p.name)
        for n in ['matched-prefix-y0','matched-prefix-ym2','matched-prefix-yp2','physical-engagement-prefix']:
            for ext in ['json','npz']:
                p=OUT/'checkpoints'/(n+'.'+ext);z.write(p,'evidence/checkpoints/'+p.name)
        for n in ['matched-wrench-y0','matched-velocity-y0','matched-pose-y0']:
            z.write(OUT/(n+'.npz'),'evidence/raw/'+n+'.npz')
        for p in (SITE/'qa').glob('physical-*.json'):z.write(p,'validation/'+p.name)
        for p in [OUT/'soft-contact-validation-i100.0.json',OUT/'velocity-frame-validation.json']:
            z.write(p,'validation/'+p.name)
        for p in (OUT/'rtx').glob('*.render.json'):z.write(p,'validation/rtx/'+p.name)
    assert zpath.stat().st_size<16*2**20
    print(zpath,zpath.stat().st_size/2**20,'MiB')
if __name__=='__main__':package()
