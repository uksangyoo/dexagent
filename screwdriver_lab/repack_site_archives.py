"""Losslessly change historical ZIP downloads to smaller tar.xz archives."""
from pathlib import Path
import zipfile,tarfile,io,shutil,json,gzip
ROOT=Path(__file__).resolve().parent.parent;DIST=ROOT/'screwdriver_site/dist';ASSETS=DIST/'assets';KEEP=ROOT/'screwdriver_lab/artifacts/site_original_trace_packs';KEEP.mkdir(exist_ok=True);mapping={}
for name in ['screwdriver-pipeline.zip','screw-in-pipeline.zip','reference-revision.zip','full-pipeline.zip']:
 src=ASSETS/name;dst=src.with_suffix('.tar.xz')
 if not src.exists():continue
 with zipfile.ZipFile(src)as z,tarfile.open(dst,'w:xz',preset=9)as t:
  for item in z.infolist():
   b=z.read(item.filename);i=tarfile.TarInfo(item.filename);i.size=len(b);i.mtime=0;t.addfile(i,io.BytesIO(b))
 mapping['assets/'+src.name]='assets/'+dst.name;shutil.move(src,KEEP/src.name);print(name,round((KEEP/src.name).stat().st_size/2**20,2),'->',round(dst.stat().st_size/2**20,2),flush=True)
for p in DIST.rglob('*'):
 if not p.is_file():continue
 if p.suffix in ['.html','.js','.json','.txt']:
  s=p.read_text()
  for a,b in mapping.items():s=s.replace(a,b)
  p.write_text(s)
 elif p.name.endswith('.json.gz'):
  s=gzip.decompress(p.read_bytes()).decode()
  for a,b in mapping.items():s=s.replace(a,b)
  p.write_bytes(gzip.compress(s.encode(),mtime=0))
