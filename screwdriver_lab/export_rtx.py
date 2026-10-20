"""Export exact MuJoCo visual geometry and recorded poses for RTX rendering."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from screwdriver_lab.scene import ROOT

def relocate_xml(xml):
    """Resolve recorded asset paths against the current extracted workspace."""
    root=ET.fromstring(xml);root.find('compiler').set('meshdir',str(ROOT/'third-party/tesollo_model/dg5fs/meshes'))
    for mesh in root.findall('asset/mesh'):
        filename=mesh.get('file','')
        # Older recordings predate the move of the UR5e assets into third-party/.
        for marker in ['third-party/universal_robots_ur5e/','screwdriver_lab/assets/ur5e/']:
            if marker in filename:mesh.set('file',str(ROOT/'third-party/universal_robots_ur5e'/filename.split(marker,1)[1]));break
    return ET.tostring(root,encoding='unicode')

def export(xml,poses,out,metadata=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_string(relocate_xml(xml));d=mujoco.MjData(m)
    geometry=[];meshes={}
    for g in range(m.ngeom):
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:continue
        entry=dict(id=g,name=m.geom(g).name or f'geom_{g}',body=m.body(int(m.geom_bodyid[g])).name,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),rgba=m.geom_rgba[g].tolist())
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[g])
        entry['geom_to_body_rotation']=rotation.reshape(3,3).tolist()
        entry['geom_to_body_translation']=m.geom_pos[g].tolist()
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[g]);entry['mesh']=mid
            if f'verts_{mid}' not in meshes:
                a,n=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid]);fa,fn=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
                meshes[f'verts_{mid}']=m.mesh_vert[a:a+n].copy();meshes[f'faces_{mid}']=m.mesh_face[fa:fa+fn].copy()
        if m.geom_matid[g]>=0:entry['rgba']=m.mat_rgba[m.geom_matid[g]].tolist()
        geometry.append(entry)
    transforms=[]
    for q in poses:
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        positions=d.geom_xpos.copy();quats=np.zeros((m.ngeom,4))
        for g in range(m.ngeom):mujoco.mju_mat2Quat(quats[g],d.geom_xmat[g])
        transforms.append(np.concatenate([positions,quats],axis=1))
    np.savez_compressed(out/'scene.npz',transforms=np.asarray(transforms),**meshes)
    (out/'scene.json').write_text(json.dumps(dict(geometry=geometry,metadata=metadata or {})))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--id');p.add_argument('--out',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.preview:
        from screwdriver_lab.simulator import Simulator
        s=Simulator({});export(s.xml,[s.data.qpos],a.out,dict(id='clearance-preview',frames=[dict(t=0,phase='reset')]))
    else:
        r=json.loads((a.input/f'{a.id}.json').read_text());q=np.load(a.input/f'{a.id}.npz')['qpos'];export((a.input/f'{a.id}.xml').read_text(),q,a.out,r)
if __name__=='__main__':main()
