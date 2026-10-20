"""Package immutable trial evidence separately from evolving implementation."""
import hashlib, json, subprocess, sys, zipfile
import numpy as np
from pathlib import Path
from screwdriver_lab.hom.model import ROOT, OUT


def archive(path,files):
    files=sorted(set(files));manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for f in files:z.write(f,str(f.relative_to(ROOT)))
        z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2))
    assert path.stat().st_size<16*1024*1024, 'Split evidence rather than exceed the asset limit'
    print(path.name,path.stat().st_size)


def build():
    assets=ROOT/'screwdriver_site/dist/assets'
    records=[];snapshots=[]
    for interface in ['cartesian','object','hom']:
        for condition in ['near','nominal','far']:
            name=interface+'-refined22-'+condition
            report=json.loads((OUT/(name+'.json')).read_text())
            xml=OUT/(name+'.xml');assert hashlib.sha256(xml.read_bytes()).hexdigest()==report['xml_sha256']
            records.extend(OUT/(name+suffix)for suffix in ['.json','.xml','.npz'])
            for filename,digest in report['source_sha256'].items():
                source=OUT/'source_snapshots'/(digest+'_'+filename)
                assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
                snapshots.append(source)
    archive(assets/'hom-recordings.zip',records)
    environment=OUT/'hom-environment.txt'
    environment.write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    files=list((ROOT/'screwdriver_lab/hom').glob('*.py'))+snapshots+[environment,ROOT/'screwdriver_lab/HOM_REVISION.txt',ROOT/'screwdriver_lab/thread_geometry.py',ROOT/'screwdriver_lab/export_rtx.py',ROOT/'screwdriver_lab/scene.py']
    files.extend(ROOT/'screwdriver_lab/artifacts/physical_task'/name for name in ['alternating-seat-nominal-34.xml','matched-prefix-y0.npz'])
    # Preserve exact converted CAD and pressure meshes, rather than requiring
    # the reviewer to regenerate them with a different geometry-library build.
    files.extend((OUT/'meshes').glob('*.obj'))
    files.extend((OUT/'meshes').glob('pad-*-isotropic-h*.*'))
    files.extend(OUT/name for name in ['corrected-scene-audit.json','hydrostatic-validation.json','pad-pressure-convergence.json','seating-guard-audit.json','checkpoint-validation.json','pickup-development-summary.json','pickup-trial47-aborted.json'])
    files.extend(OUT.glob('thread-validation-filtered*.json'))
    for name in ['cartesian-timestep-check','object-timestep-check','hom-timestep-check','hom-padmesh-check','hom-friction-low','hom-friction-high','hom-modal26','hom-seating23','hom-seat-stop40','hom-seat-stop41','hom-seat-stop42']:
        if (OUT/(name+'.json')).exists():
            files.append(OUT/(name+'.json'))
            if name not in ['hom-modal26','hom-seating23','hom-seat-stop40','hom-seat-stop41','hom-seat-stop42']:files.append(OUT/(name+'.npz'))
            if name=='hom-seat-stop42':
                a=np.load(OUT/(name+'.npz'))
                # Exact observations and replay states; motor-reference arrays
                # remain in the original local recording, without modification.
                np.savez_compressed(assets/(name+'-observations.npz'),**{k:a[k]for k in a.files if k not in ['qref_500hz','vref_500hz']})
            report=json.loads((OUT/(name+'.json')).read_text())
            files.extend(OUT/'source_snapshots'/(v+'_'+k)for k,v in report['source_sha256'].items())
    for name in ['pickup-trial39','pickup-trial43','pickup-trial44','pickup-trial45','pickup-trial46','pickup-trial48','pickup-trial49']:
        path=OUT/(name+'.json')
        if path.exists():
            files.append(path);report=json.loads(path.read_text())
            files.extend(OUT/'source_snapshots'/(v+'_'+k)for k,v in report['source_sha256'].items())
    for name in ['cartesian','object','hom']:
        files.append(OUT/'rtx'/(name+'-refined22-nominal.render.json'))
    for path in (OUT/'source_snapshots').glob('*_render.py'):files.append(path)
    archive(assets/'hom-code-evidence.zip',files)


if __name__=='__main__':build()
