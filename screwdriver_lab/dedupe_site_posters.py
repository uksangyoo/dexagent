"""Reuse byte-identical historical posters without changing image content."""
from pathlib import Path
import hashlib,collections,json,gzip,shutil
ROOT=Path(__file__).resolve().parent.parent;DIST=ROOT/'screwdriver_site/dist';KEEP=ROOT/'screwdriver_lab/artifacts/site_duplicate_posters';KEEP.mkdir(exist_ok=True);groups=collections.defaultdict(list)
for p in sorted((DIST/'assets').glob('*.jpg')):groups[hashlib.sha256(p.read_bytes()).hexdigest()].append(p)
mapping={};saved=0
for paths in groups.values():
 for p in paths[1:]:mapping['assets/'+p.name]='assets/'+paths[0].name;saved+=p.stat().st_size;shutil.move(p,KEEP/p.name)
for p in DIST.rglob('*'):
 if not p.is_file():continue
 zipped=p.name.endswith('.json.gz')
 if not zipped and p.suffix not in ['.html','.js','.json']:continue
 s=gzip.decompress(p.read_bytes()).decode()if zipped else p.read_text();new=s
 for a,b in mapping.items():new=new.replace(a,b)
 if new!=s:
  if zipped:p.write_bytes(gzip.compress(new.encode(),mtime=0))
  else:p.write_text(new)
(ROOT/'screwdriver_site/qa/poster-deduplication.json').write_text(json.dumps(dict(mapping=mapping,bytes_saved=saved,criterion='SHA256 equality; originals retained outside deployment'),indent=2));print('Freed MiB',saved/2**20)
