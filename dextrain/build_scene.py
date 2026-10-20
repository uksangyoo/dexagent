"""Compose the local UR5e and Tesollo assets and initialize a nut-spin skill.

The nut is already engaged on a passive revolute fixture. This curriculum
tests contact-driven rotation, not thread acquisition or axial advancement.
"""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ARM_HOME = np.array([-1.5708, -1.2, 1.8, -2.17, -1.5708, 0.0])


def numbers(v):
    return " ".join(f"{x:.9g}" for x in v)


def compose(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arm_dir = ROOT.parent / "graph-as-policy/third_party/robosuite/robosuite/models/assets/robots/ur5e"
    hand_dir = ROOT / "third-party/tesollo_model/dg5fs"
    scene = ET.parse(arm_dir / "robot.xml").getroot()
    scene.set("model", "ur5e_dg5fs_nut_spin_curriculum")
    ET.SubElement(scene, "compiler", angle="radian", autolimits="true")
    ET.SubElement(scene, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", iterations="30", gravity="0 0 -9.81")
    for mesh in scene.findall("./asset/mesh"):
        mesh.set("file", str((arm_dir / mesh.get("file")).resolve()))
    actuator = scene.find("actuator")
    actuator.clear()
    arm_joints = scene.findall("./worldbody//joint")
    for j, home in zip(arm_joints, ARM_HOME):
        ET.SubElement(actuator, "position", name=f"drive_{j.get('name')}", joint=j.get("name"), kp="3000", kv="120", forcerange="-150 150" if "shoulder" in j.get("name") or "elbow" in j.get("name") else "-28 28")
        j.set("armature", "0.02")
    hand = ET.parse(hand_dir / "dg5fs_right_w_mount.xml").getroot()
    for mesh in hand.findall("./asset/mesh"):
        mesh.set("file", str((hand_dir / "meshes" / mesh.get("file")).resolve()))
        scene.find("asset").append(deepcopy(mesh))
    wrist = scene.find(".//body[@name='right_hand']")
    mount = ET.SubElement(wrist, "body", name="dg5fs_mount", pos="0 0 0")
    # Combine the two fixed-root inertias from the supplied URDF.
    urdf = ET.parse(hand_dir / "dg5fs_right_w_mount.urdf")
    components = []
    for name, offset in [("link_mount", np.zeros(3)), ("link_base", np.array([0, 0, .015]))]:
        i = urdf.find(f".//link[@name='{name}']/inertial")
        mass = float(i.find("mass").get("value"))
        xyz = np.fromstring(i.find("origin").get("xyz"), sep=" ") + offset
        a = i.find("inertia").attrib
        tensor = np.array([[float(a["ixx"]),float(a["ixy"]),float(a["ixz"])], [float(a["ixy"]),float(a["iyy"]),float(a["iyz"])], [float(a["ixz"]),float(a["iyz"]),float(a["izz"])]] )
        components.append((mass, xyz, tensor))
    mass = sum(x[0] for x in components)
    com = sum(m*x for m,x,_ in components) / mass
    inertia = sum(I + m*(np.dot(x-com,x-com)*np.eye(3)-np.outer(x-com,x-com)) for m,x,I in components)
    ET.SubElement(mount, "inertial", mass=str(mass), pos=numbers(com), fullinertia=numbers([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
    for element in hand.find("worldbody"):
        mount.append(deepcopy(element))
    for joint in mount.findall(".//joint"):
        joint.set("damping", "0.03")
        joint.set("armature", "0.002")
        ET.SubElement(actuator, "position", name=f"drive_{joint.get('name')}", joint=joint.get("name"), kp="6", kv="0.15", forcerange="-7.5 7.5")
    for i in range(1,6):
        tip = mount.find(f".//body[@name='link_{i}_4']")
        geom = tip.find(f"geom[@mesh='link_{i}_tip']")
        ET.SubElement(tip, "site", name=f"tip_{i}", pos=geom.get("pos"), size="0.003", rgba="1 0.2 0.2 1")
    for geom in mount.findall(".//geom"):
        geom.set("friction", "1.0 0.005 0.0001")
        geom.set("condim", "3")
        geom.set("solref", "0.006 1")
    # Keep nonadjacent self contacts, omit the mechanical base/mount overlap.
    contact = ET.SubElement(scene,"contact")
    ET.SubElement(contact,"exclude",body1="right_hand",body2="dg5fs_mount")
    for i in range(1,6):
        ET.SubElement(contact,"exclude",body1="dg5fs_mount",body2=f"link_{i}_1")
    # The convex palm hull closes the thumb-pivot cutout. Its overlap with
    # the second thumb body persists through the admissible initial grasp.
    # Filter only this mechanical pair; retain other nonadjacent contacts.
    ET.SubElement(contact,"exclude",body1="dg5fs_mount",body2="link_1_2")
    path = out / "scene.xml"
    ET.ElementTree(scene).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:6] = ARM_HOME
    mujoco.mj_forward(model,data)
    mount_id = model.body("dg5fs_mount").id
    origin = data.xpos[mount_id].copy()
    rot = data.xmat[mount_id].reshape(3,3).copy()
    tip_ids = [model.site(f"tip_{i}").id for i in range(1,6)]
    # Find an opposed thumb/index/middle configuration around a small nut.
    center = np.array([.062,.012,.125])
    tip_targets = np.array([[.063,.035,.103],[.063,.031,.148],[.063,-.009,.145],[.056,-.030,.129],[.046,-.043,.113]])
    lo, hi = model.jnt_range[6:,0].copy(), model.jnt_range[6:,1].copy()
    def residual(q):
        data.qpos[6:] = q
        mujoco.mj_forward(model,data)
        tips = (data.site_xpos[tip_ids]-origin) @ rot
        return np.concatenate(((tips-tip_targets).ravel()*100, .02*q))
    q0 = np.clip(np.array([.4,-.9,.2,.2, 0,.7,.3,.2, 0,.7,.3,.2, 0,.7,.3,.2, .2,.6,.3,.2]),lo+1e-5,hi-1e-5)
    fit = least_squares(residual,q0,bounds=(lo+1e-5,hi-1e-5),max_nfev=250)
    residual(fit.x)
    tips = ((data.site_xpos[tip_ids]-origin) @ rot).copy()
    # A world-fixed fixture oriented along the palm's X axis.
    nut_rotation = rot @ Rotation.from_euler("y", np.pi/2).as_matrix()
    quat = Rotation.from_matrix(nut_rotation).as_quat(scalar_first=True)
    nut = ET.SubElement(scene.find("worldbody"),"body",name="nut",pos=numbers(origin+rot@center),quat=numbers(quat))
    ET.SubElement(nut,"joint",name="nut_rotation",type="hinge",axis="0 0 1",limited="false",damping="0.0005",frictionloss="0.0002",armature="0.00002")
    for i in range(6):
        angle = i*np.pi/3
        ET.SubElement(nut,"geom",name=f"nut_face_{i}",type="box",pos=numbers([.020*np.cos(angle),.020*np.sin(angle),0]),euler=numbers([0,0,angle]),size="0.004 0.0116 0.010",mass="0.012",friction="1.0 0.005 0.0001",condim="3",solref="0.006 1",rgba="0.7 0.7 0.75 1")
    ET.SubElement(nut,"site",name="nut_center",size="0.002",rgba="0 1 0 1")
    ET.SubElement(scene.find("worldbody"),"light",pos="0 -1 2",dir="0 0 -1",diffuse="0.8 0.8 0.8")
    qhome = np.concatenate([ARM_HOME,fit.x,[0]])
    kf = ET.SubElement(scene,"keyframe")
    ET.SubElement(kf,"key",name="home",qpos=numbers(qhome),ctrl=numbers(qhome[:26]))
    ET.indent(scene)
    ET.ElementTree(scene).write(path)
    # Resolve mesh-surface penetration before saving a grasp initialization.
    # Tip sites are not contact surfaces; raw tip IK alone overlaps the nut.
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = qhome
    data.qpos[18:26] = 0  # Ring/little start clear; policy may recruit them.
    safe_seed = data.qpos[6:26].copy()
    nut_geoms = [model.geom(f"nut_face_{i}").id for i in range(6)]
    fingertip_geoms = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(f"link_{i}_4").id] for i in range(1,4)]
    nb = model.nbody
    def contact_residual(q):
        data.qpos[6:26] = q
        mujoco.mj_forward(model,data)
        penetration = np.zeros((nb,nb))
        for c in data.contact:
            a,b = sorted([model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]])
            penetration[a,b] = min(penetration[a,b],min(c.dist+0.00015,0))
        distances = [min(mujoco.mj_geomDistance(model,data,g,n,.1,None) for g in geoms for n in nut_geoms) for geoms in fingertip_geoms]
        # Strongly prevent interpenetration; place three fingertips near contact.
        return np.concatenate([penetration.ravel()*2000,np.array(distances)*150,(q-safe_seed)*.1])
    safe_seed = np.clip(safe_seed,lo+1e-5,hi-1e-5)
    safe = least_squares(contact_residual,safe_seed,bounds=(lo+1e-5,hi-1e-5),max_nfev=120,diff_step=1e-4)
    contact_residual(safe.x)
    qhome[6:26] = safe.x
    worst = min((float(c.dist) for c in data.contact),default=0)
    if worst < -.001:
        print('REFINEMENT',safe.message,'q',safe.x.tolist(),flush=True)
        for c in sorted(data.contact,key=lambda c:c.dist)[:8]:
            print(float(c.dist),model.body(model.geom_bodyid[c.geom1]).name,model.body(model.geom_bodyid[c.geom2]).name,flush=True)
        raise RuntimeError(f"Initial scene has excessive penetration: {worst} m")
    kf.find("key").set("qpos",numbers(qhome))
    kf.find("key").set("ctrl",numbers(qhome[:26]))
    ET.ElementTree(scene).write(path)
    meta = {"stage":"motor_skill_curriculum_not_full_study", "task":"passive_fixture_nut_rotation", "arm_qhome":ARM_HOME.tolist(),"qhome":qhome.tolist(),"nut_center_in_mount":center.tolist(),"fingertip_positions_in_mount":tips.tolist(),"ik_tip_error_m":np.linalg.norm(tips-tip_targets,axis=1).tolist(),"sources":[str(arm_dir/"robot.xml"),str(hand_dir/"dg5fs_right_w_mount.xml")],"limitations":["Simulation mount transform is provisional", "Nut has one passive rotational DOF; no thread capture or axial advance", "Arm held near fixed pose during the first motor curriculum", "Contact and torque parameters require hardware calibration"]}
    meta["initial_min_contact_distance_m"] = worst
    meta["contact_refinement"] = "Geometry distance and self-contact penalty after initial tip IK"
    meta["collision_filter_note"] = "Palm vs thumb second link excluded because the convex palm collider fills the pivot cutout; other nonadjacent self contacts remain enabled"
    meta["fingertip_positions_in_mount"] = ((data.site_xpos[[model.site(f'tip_{i}').id for i in range(1,6)]]-origin) @ rot).tolist()
    (out/"scene.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2),flush=True)
    return path


if __name__ == "__main__":
    compose(ROOT/"runs/assets")
