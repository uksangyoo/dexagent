"""Package corrected source, recorded physics, meshes and the offline website."""
import json
import re
import zipfile
from screwdriver_lab.scene import ROOT


def main():
    site = ROOT / 'screwdriver_site/dist'
    for name, count in [('manifest.json', 50), ('local-motor.json', 10)]:
        data = json.loads((site / 'data' / name).read_text())
        assert len(data['results']) == count and data['rendering']['RTX']
        assert all((site / r['video']).exists() for r in data['results'])
    output = site / 'assets/screwdriver-pipeline.zip'
    media = sorted(p for p in (site / 'assets').iterdir() if p.suffix in ['.mp4','.jpg','.png'])
    packs=[];group=[];size=0
    for path in media:
        if group and size+path.stat().st_size>10*1024**2:
            packs.append(group);group=[];size=0
        group.append(path);size+=path.stat().st_size
    if group:packs.append(group)
    links=[]
    for i,group in enumerate(packs,1):
        pack=site/'assets'/f'rollout-media-{i}.zip'
        with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in group:z.write(path,path.relative_to(ROOT))
        assert pack.stat().st_size<12*1024**2
        links.append(f'<a href="assets/{pack.name}" download>Media pack {i} ({pack.stat().st_size/1048576:.1f} MB) ↗</a>')
    html=site/'index.html'
    html.write_text(re.sub(r'<div id="media-downloads" class="artifact-links">.*?</div>', '<div id="media-downloads" class="artifact-links">'+''.join(links)+'</div>', html.read_text()))
    paths = list((ROOT / 'screwdriver_lab').glob('*.py'))
    paths += [ROOT / 'screwdriver_lab/README.md', ROOT / 'screwdriver_lab/requirements.txt',
              ROOT / 'screwdriver_lab/artifacts/grasp.json', ROOT / 'third-party/tesollo_model/LICENSE',
              ROOT / 'third-party/tesollo_model/dg5fs/dg5fs_right.xml', ROOT / 'dextrain/__init__.py',
              ROOT / 'dextrain/resource_guard.py', ROOT / 'screwdriver_site/README.md']
    paths += list((ROOT / 'third-party/tesollo_model/dg5fs/meshes/dg5fs_right').glob('*'))
    for folder in ['main', 'local_motor', 'timestep_check']:
        paths += list((ROOT / 'screwdriver_lab/artifacts/revision_rtx' / folder).glob('*'))
    paths += [p for p in site.rglob('*') if p.is_file() and p.suffix not in ['.mp4','.jpg','.png','.zip']]
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(set(paths)):
            if path.is_file():
                z.write(path, path.relative_to(ROOT))
        z.writestr('README.md', '# Corrected screwdriver control laboratory\n\n'
                   'Start with screwdriver_lab/README.md. The website is in screwdriver_site/dist/.\n'
                   'All 60 videos use Cycles/OptiX RTX rendering and include acquisition.\n'
                   'Legacy invalid-grasp results are excluded from this bundle.\n'
                   'For offline playback, download all rollout-media packs from the website and extract\n'
                   'them into this same directory. Source, meshes, physics and traces are in this main bundle.\n')
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert not any(n.endswith('.mp4') for n in z.namelist())
    assert output.stat().st_size<16*1024**2
    print(output, output.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
