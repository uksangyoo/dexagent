"""Prepare an isolated checkout of the exact refined22 dynamics sources."""
import argparse, hashlib, json, shutil
from pathlib import Path
from dexagent.sim.model import ROOT, OUT, SOURCE, PREFIX


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Use a new destination; existing work is preserved')
    destination.mkdir(parents=True)
    package=destination/'screwdriver_lab';package.mkdir()
    (package/'__init__.py').write_text('')
    # The frozen sources still import screwdriver_lab.hom.*; rebuild that layout.
    (package/'hom').mkdir();(package/'hom/__init__.py').write_text('')
    shutil.copy2(ROOT/'dexagent/sim/seating.py',package/'hom/seating.py')
    report=json.loads((OUT/'hom-refined22-nominal.json').read_text())
    for name,digest in report['source_sha256'].items():
        source=OUT/'source_snapshots'/(digest+'_'+name)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        shutil.copy2(source,package/'hom'/name)
    shutil.copy2(ROOT/'dexagent/sim/thread_geometry.py',package/'thread_geometry.py')
    # Frozen sources use the pre-third-party/ layout; the frozen XML uses third-party/.
    (destination/'third-party').symlink_to(ROOT/'third-party',target_is_directory=True)
    (destination/'tesollo_model').symlink_to(ROOT/'third-party/tesollo_model',target_is_directory=True)
    (package/'assets').mkdir()
    (package/'assets/ur5e').symlink_to(ROOT/'third-party/universal_robots_ur5e',target_is_directory=True)
    fixture=package/'artifacts/physical_task';fixture.mkdir(parents=True)
    (fixture/SOURCE.name).write_text(SOURCE.read_text().replace('/home/uyoo/Projects/dexagent/',str(destination)+'/'))
    shutil.copy2(PREFIX,fixture/PREFIX.name)
    # The frozen mesher selects a canonical pad by its original converted
    # basename. Preserve it and the exact cached tetrahedra when relocating.
    meshes=package/'artifacts/hom_revision/meshes';meshes.mkdir(parents=True)
    for pattern in ['*.obj','pad-*-isotropic-h*.*']:
        for source in (OUT/'meshes').glob(pattern):shutil.copy2(source,meshes/source.name)
    print('Frozen sources verified. Working directory:',destination)
    print('CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 python -m tasks.screwdriver.study.experiment --name reproduced-hom --interface hom --duration 6 --speed 60 --dt .00025 --thread-segments 48 --explicit-steel --beta 0 --cycle alternating --drive-load .4 --return-load .06 --hybrid --tilt 0')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    prepare(parser.parse_args().destination)
