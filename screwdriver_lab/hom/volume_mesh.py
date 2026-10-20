"""Measured CAD simplification and tetrahedral pressure meshes.

Geometry processing uses millimeters to avoid absolute geometric tolerances
at meter-scale tiny facets. VTK dynamics geometry is always written in meters.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import tetgen,trimesh,pymeshlab as ml
from screwdriver_lab.hom.model import OUT

def surface_error(a,b):
    points=np.r_[a.vertices,a.triangles_center];d=[]
    for i in range(0,len(points),64):d.extend(trimesh.proximity.closest_point_naive(b,points[i:i+64])[1])
    return float(max(d))

def tetrahedral_pad(source,resolution=.002):
    source=Path(source);canonical=OUT/'meshes/5d9ee0d505_link_1_tip.obj'
    original=trimesh.load(source,force='mesh');original.apply_scale(1000)
    if canonical.exists() and canonical!=source:
        reference=trimesh.load(canonical,force='mesh');reference.apply_scale(1000)
        # The CAD pads share vertices and surface, but triangulate some flat
        # faces differently. Verify the surface, not triangle index identity.
        assert surface_error(original,reference)<1e-8 and surface_error(reference,original)<1e-8
        original=reference;source=canonical
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    out=OUT/'meshes'/f'pad-{digest}-isotropic-h{resolution:g}.vtk'
    if out.exists() and out.with_suffix('.json').exists():return out
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(original.vertices,original.faces))
    surface_resolution_mm=.8 if resolution>=.002 else .4
    ms.apply_filter('meshing_isotropic_explicit_remeshing',iterations=8,targetlen=ml.PureValue(surface_resolution_mm),maxsurfdist=ml.PureValue(.04),featuredeg=40)
    remeshed=ms.current_mesh();mesh=trimesh.Trimesh(remeshed.vertex_matrix(),remeshed.face_matrix())
    assert mesh.is_watertight and mesh.volume>0
    distances=[surface_error(original,mesh)/1000,surface_error(mesh,original)/1000]
    assert max(distances)<.00008,distances
    nodes,elements,_,_=tetgen.TetGen(mesh.vertices,mesh.faces).tetrahedralize(order=1,quality=True,minratio=2.,quiet=True,maxvolume=(resolution*1000)**3/6,fixedvolume=True)
    nodes/=1000
    e=nodes[elements];dets=np.linalg.det(np.stack([e[:,1]-e[:,0],e[:,2]-e[:,0],e[:,3]-e[:,0]],axis=1))
    negative=dets<0;elements[negative,:2]=elements[negative,1::-1]
    volume=abs(dets).sum()/6
    assert abs(volume/(mesh.volume*1e-9)-1)<1e-8
    assert abs(volume/(original.volume*1e-9)-1)<.01
    with out.open('w')as f:
        f.write('# vtk DataFile Version 3.0\nTesollo pad tetrahedra in meters\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {len(nodes)} double\n');np.savetxt(f,nodes,fmt='%.17g')
        f.write(f'CELLS {len(elements)} {len(elements)*5}\n');np.savetxt(f,np.c_[np.full(len(elements),4),elements],fmt='%d')
        f.write(f'CELL_TYPES {len(elements)}\n');np.savetxt(f,np.full(len(elements),10),fmt='%d')
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),requested_resolution_m=resolution,vertices=len(nodes),tetrahedra=len(elements),original_volume_m3=original.volume*1e-9,tetrahedral_volume_m3=volume,relative_volume_error=abs(volume/(original.volume*1e-9)-1),sampled_bidirectional_surface_error_m=distances,physical_geometry_scale=1,boundary_approximation='Isotropic surface remeshing with 0.04 mm surface-distance constraint; proximity queries in mm; sampled vertices and triangle centroids, not a certified Hausdorff bound')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print('TETRAHEDRAL_PAD',report,flush=True)
    return out

from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
