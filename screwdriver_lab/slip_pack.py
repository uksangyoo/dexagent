"""Lossless raw evidence packs; exact frozen controller sources accompany runs."""
from pathlib import Path
import tarfile,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'screwdriver_lab/artifacts/full_motion';ASSETS=ROOT/'screwdriver_site/dist/assets'
def pack(label,names):
 path=ASSETS/f'controlled-slip-{label}.tar.xz'
 with tarfile.open(path,'w:xz',preset=9)as t:
  for name in names:
   for suffix in ['.json','.npz','.xml']:
    p=OUT/(name+suffix);t.add(p,arcname=str(p.relative_to(ROOT)))
 print(path.name,round(path.stat().st_size/2**20,2),flush=True)
if __name__=='__main__':
 for method in ['pose','velocity','wrench']:
  pack(method,[p.stem for p in sorted(OUT.glob(f'slip-study-{method}-*.json'))])
 pack('seating',['slip-verified-seating'])
 pack('integration',['slip-full-fixed-tripod','slip-hold-control','slip-no-unload-control','slip-hydro-material','slip-return-load-9','slip-return-load-10','slip-return-load-11','slip-velocity-integral','slip-speed-36','slip-speed-60','slip-progress-graph','slip-fast-light-contact'])
