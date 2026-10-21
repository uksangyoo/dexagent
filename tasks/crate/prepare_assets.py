"""Reproducible MJCF assembly; no simulation or object-pose manipulation."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "assets"
VAB = ROOT / "third-party/vab_crate_washing"


def save(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def reconstructed_rack():
    """Reference-derived shell, reinforcing ribs and retaining clips.

    Nominal outer dimensions borrow the benchmark (370 x 525 x 75 mm).
    These are not measured dimensions of the filmed rack.
    """
    parts=[]
    def box(name,center,half):parts.append((name,np.array(center),np.array(half)))
    box("floor",(0,0,-.0335),(.181,.2585,.004))
    for s in [-1,1]:
        box(f"long_wall{s}",(s*.179,0,0),(.003,.2585,.0335))
        box(f"short_wall{s}",(0,s*.2565,0),(.181,.003,.0335))
        box(f"long_rim{s}",(s*.179,0,.0345),(.006,.2625,.003))
        box(f"short_rim{s}",(0,s*.2565,.0345),(.185,.006,.003))
        for z in [-.029,-.007,.015]:
            box("side_rail",(s*.183,0,z),(.002,.2625,.0018))
            box("end_rail",(0,s*.2605,z),(.185,.002,.0018))
        for y in np.linspace(-.245,.245,11):box("vertical_side_rib",(s*.183,y,0),(.002,.0018,.0335))
        for x in np.linspace(-.165,.165,8):box("vertical_end_rib",(x,s*.2605,0),(.0018,.002,.0335))
    for y in np.linspace(-.218,.218,6):
        box("internal_cross_rib",(0,y,-.023),(.174,.002,.0065))
        for x in [-.145,-.043,.043,.145]:
            for dy in [-.006,.006]:
                box("retaining_clip",(x,y+dy,-.010),(.007,.0018,.0195))
    for x in [-.145,.145]:box("longitudinal_stiffener",(x,0,-.023),(.002,.25,.0065))
    visual=[]
    for name,p,h in parts:
        mesh=trimesh.creation.box(extents=2*h);mesh.apply_translation(p);visual.append(mesh)
    merged=trimesh.util.concatenate(visual)
    merged.export(OUT/"reference_rack.obj")
    volumes=np.array([8*np.prod(h) for _,_,h in parts]);masses=volumes/volumes.sum()
    center=sum(m*p for m,(_,p,_) in zip(masses,parts))
    inertia=np.zeros((3,3))
    for m,(_,p,h) in zip(masses,parts):
        d=p-center
        inertia+=m*(np.diag([(h[1]**2+h[2]**2)/3,(h[0]**2+h[2]**2)/3,(h[0]**2+h[1]**2)/3])+np.eye(3)*np.dot(d,d)-np.outer(d,d))
    body=ET.Element("body",name="tray")
    ET.SubElement(body,"freejoint",name="tray_free")
    ET.SubElement(body,"inertial",mass="1.0",pos=" ".join(map(str,center)),fullinertia=" ".join(map(str,[inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]])))
    ET.SubElement(body,"geom",type="mesh",mesh="reference_rack",rgba=".05 .052 .054 1",contype="0",conaffinity="0",group="2")
    for k,(name,p,h) in enumerate(parts):
        ET.SubElement(body,"geom",name=f"rack_{k}_{name}",type="box",pos=" ".join(map(str,p)),size=" ".join(map(str,h)),
                      friction="0.45 0.001 0.0001",solref="0.004 1",solimp="0.95 0.99 0.0005",group="3")
    # White barcode tags are visual-only ink on the two long rims.
    for s in [-1,1]:ET.SubElement(body,"geom",type="box",pos=f"{s*.1851} 0 .018",size=".00015 .012 .003",rgba=".8 .8 .75 1",contype="0",conaffinity="0",group="2")
    return body


def arena():
    source = VAB / "scenes/crate_washing.xml"
    root = ET.parse(source).getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
        if mesh.get("name","").startswith("crate_machine"):
            mesh.set("scale","0.1 0.075 0.1")
    for node in list(root):
        if node.tag in ("keyframe",):
            root.remove(node)
    world = root.find("worldbody")
    for node in list(world):
        if node.tag == "camera" or node.get("name", "").startswith("crate_box") or node.get("name") in ("robot_platform", "ceiling"):
            world.remove(node)
    # Stage the physical cart separately; original benchmark stack rests on floor.
    for material in root.findall("asset/material"):
        if material.get("name") == "crate_box_default":
            material.set("rgba", "0.025 0.027 0.028 1")
    world.find("geom[@name='floor']").set("pos","0 0 -.023")
    def block(name,p,h,rgba):
        ET.SubElement(world,"geom",name=name,type="box",pos=" ".join(map(str,p)),size=" ".join(map(str,h)),rgba=rgba,group="2")
    # Reference room details, all authored in world coordinates, not composited.
    block("window_frame",(2.0,-.95,1.45),(.72,.025,.65),".72 .74 .73 1")
    block("window_glass",(2.0,-.92,1.45),(.68,.006,.6),".59 .69 .69 1")
    for j in range(29):block(f"blind_{j}",(2.0,-.90,.89+j*.039),(.67,.008,.012),".64 .68 .65 1")
    block("washer_black_column",(.66,-.33,1.22),(.012,.045,.315),".025 .027 .03 1")
    block("washer_information_label",(.674,-.33,1.39),(.001,.039,.115),".87 .87 .84 1")
    block("control_box",(.69,-.378,1.2),(.045,.024,.095),".78 .66 .08 1")
    ET.SubElement(world,"geom",name="emergency_stop",type="cylinder",pos=".737 -.378 1.23",size=".014 .006",quat=".7071068 0 .7071068 0",rgba=".65 .02 .015 1",group="2")
    block("floor_mat",(1.98,0,.009),(.26,.62,.006),".18 .18 .17 1")
    for side in [-1,1]:
        block(f"robot_pedestal_{side}",(1.98,side*.43,.34),(.10,.10,.34),".18 .20 .21 1")
    # The source machine has a flat entry table. Reconstruct the visible wire
    # grid above it. Its transport dynamics are still unmodeled (no fake belt force).
    for j in range(35):
        x=.67+j*.0135
        ET.SubElement(world,"geom",name=f"conveyor_crosswire_{j}",type="capsule",fromto=f"{x} -.30 .91 {x} .30 .91",size=".0017",rgba=".42 .49 .43 1",group="2")
    for j in range(17):
        y=-.30+j*.0375
        ET.SubElement(world,"geom",name=f"conveyor_longwire_{j}",type="capsule",fromto=f".67 {y} .913 1.13 {y} .913",size=".0015",rgba=".42 .49 .43 1",group="2")
    tray = reconstructed_rack()
    ET.SubElement(root.find("asset"),"mesh",name="reference_rack",file=str(OUT/"reference_rack.obj"))
    tray.set("pos", "0 0 0")
    tray.set("name", "tray")
    tray.find("freejoint").set("name", "tray_free")
    # Source has unmeasured friction=1.2, torsion=.1 m. Use explicitly uncertain
    # dry polymer nominal, small physically dimensioned torsional/rolling terms.
    for geom in tray.findall("geom"):
        geom.attrib.pop("name", None)
        if geom.get("group") == "3":
            geom.set("friction", "0.45 0.001 0.0001")
            geom.set("solref", "0.004 1")
            geom.set("solimp", "0.95 0.99 0.0005")
    bodyroot = ET.Element("mujoco", model="rack")
    ET.SubElement(bodyroot, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(bodyroot, "asset")
    for node in root.find("asset"):
        if node.get("name", "").startswith("crate_box") or node.get("name")=="reference_rack":
            asset.append(copy.deepcopy(node))
    ET.SubElement(bodyroot, "worldbody").append(tray)
    save(bodyroot, OUT / "tray.xml")
    for i in range(11):
        b = copy.deepcopy(tray)
        b.set("name", f"tray_{i:02}")
        b.set("pos", f"1.38 0 {.1225+i*.075}")
        for k,g in enumerate(b.findall("geom")):g.set("name",f"tray_{i:02}_{k}")
        b.find("freejoint").set("name", f"tray_{i:02}_free")
        world.append(b)
    # Newton separates visible geoms from colliders. Keep both for authored
    # room primitives, just as the source mesh assets do.
    for g in list(world.findall("geom")):
        if g.get("contype","1")!="0" and g.get("group","2")=="2":
            visual=copy.deepcopy(g)
            visual.set("name",g.get("name")+"_visual")
            visual.attrib.update(contype="0",conaffinity="0",group="2")
            g.set("group","3")
            world.append(visual)
    save(root, OUT / "washer.xml")


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def prefix(tree, side):
    refs = {"name", "class", "childclass", "mesh", "material", "joint", "body1", "body2", "site", "tendon"}
    for node in tree.iter():
        for key in refs.intersection(node.attrib):
            node.set(key, f"{side}_{node.get(key)}")
    return tree


def robot():
    root = ET.Element("mujoco", model="bimanual_ur5e_tesollo")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", cone="elliptic")
    defaults = ET.SubElement(root, "default")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    acts = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    for side, y in (("left", .43), ("right", -.43)):
        src = ROOT / "third-party/universal_robots_ur5e/ur5e.xml"
        arm = ET.parse(src).getroot()
        for mesh in arm.findall("asset/mesh"):
            mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
            mesh.set("file", str(src.parent / "assets" / mesh.get("file")))
        prefix(arm, side)
        for item in arm.find("default"):
            defaults.append(item)
        for item in arm.find("asset"):
            asset.append(item)
        base = arm.find("worldbody/body")
        base.set("pos", f"1.98 {y} .68")
        world.append(base)
        for item in arm.find("actuator"):
            acts.append(item)
        src = ROOT / f"third-party/tesollo_model/dg5fs/dg5fs_{side}.xml"
        hand = ET.parse(src).getroot()
        for mesh in hand.findall("asset/mesh"):
            mesh.set("file", str(src.parent / "meshes" / mesh.get("file")))
        prefix(hand, side)
        for item in hand.find("asset"):
            asset.append(item)
        wrist = base.find(f".//body[@name='{side}_wrist_3_link']")
        palm = ET.SubElement(wrist, "body", name=f"{side}_palm", pos="0 0.1 0", quat="-1 1 1 1", childclass=f"{side}_hand")
        # Preserve manufacturer palm mass and full inertia from the URDF.
        inertial = ET.parse(src.with_suffix(".urdf")).find("link/inertial")
        ii = inertial.find("inertia")
        ET.SubElement(palm, "inertial", mass=inertial.find("mass").get("value"),
                      pos=inertial.find("origin").get("xyz"),
                      fullinertia=" ".join(ii.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        hd = ET.SubElement(defaults, "default", {"class": f"{side}_hand"})
        ET.SubElement(hd, "joint", damping="0.015", armature="0.00002")
        ET.SubElement(hd, "geom", friction="0.6 0.001 0.0001", rgba="0.12 0.13 0.14 1", solref="0.004 1", solimp="0.95 0.99 0.0005")
        for item in hand.find("worldbody"):
            palm.append(item)
        for parent in palm.iter("body"):
            for geom in list(parent.findall("geom")):
                visual=copy.deepcopy(geom)
                visual.attrib.update(contype="0",conaffinity="0",group="2")
                geom.set("group","3")
                parent.append(visual)
        for joint in palm.findall(".//joint"):
            joint.set("actuatorfrcrange", "-0.11 0.11")
            ET.SubElement(acts, "position", name=joint.get("name"), joint=joint.get("name"), kp="3", kv="0.06", forcerange="-0.11 0.11")
        # Only adjacent bodies are excluded. Other fingers and opposite arms collide.
        for parent in base.iter("body"):
            for child in parent.findall("body"):
                ET.SubElement(contact, "exclude", body1=parent.get("name"), body2=child.get("name"))
        # The manufacturer's thumb swivel housing lies inside the palm CAD.
        # Exclude this mechanically connected housing, not the distal fingers.
        ET.SubElement(contact,"exclude",body1=f"{side}_palm",body2=f"{side}_link_1_2")
    save(root, OUT / "bimanual_ur5e_tesollo.xml")


if __name__ == "__main__":
    arena()
    robot()
