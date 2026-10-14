"""Unconstrained rigid industrial-object scenes with a physical workbench.

Analytic collision geometry is intentionally explicit. These are nut, closed-end
wrench and hex-head screw proxies, not threaded-fastening or deformable-cable
models. Every manipulated object has an unactuated free joint.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build_free_scene(output, *, kind="nut", scale=1.0, friction=1.0):
    if kind not in {"nut", "wrench", "screw"}:
        raise ValueError(kind)
    root = ET.parse(ROOT / "runs/assets/scene.xml").getroot()
    root.set("model", f"ur5e_tesollo_free_{kind}")
    nut = root.find(".//body[@name='nut']")
    nut.remove(nut.find("joint"))
    ET.SubElement(nut, "freejoint", name="object_free")
    old = np.fromstring(root.find("./keyframe/key").get("qpos"), sep=" ")
    q = np.r_[
        old[:26], np.fromstring(nut.get("pos"), sep=" "), np.fromstring(nut.get("quat"), sep=" ")
    ]
    root.find("./keyframe/key").set("qpos", " ".join(map(str, q)))
    if kind == "wrench":
        ET.SubElement(
            nut,
            "geom",
            name="wrench_handle",
            type="box",
            pos="0 .057 0",
            size=".007 .034 .005",
            mass=".04",
            rgba=".55 .58 .63 1",
        )
    if kind == "screw":
        ET.SubElement(
            nut,
            "geom",
            name="screw_head_core",
            type="box",
            size=".015 .015 .01",
            mass=".025",
            rgba=".65 .67 .69 1",
        )
        ET.SubElement(
            nut,
            "geom",
            name="screw_shaft",
            type="box",
            pos="0 0 .035",
            size=".006 .006 .025",
            mass=".018",
            rgba=".65 .67 .69 1",
        )
    for g in nut.findall("geom"):
        for attr in ["pos", "size"]:
            if g.get(attr):
                g.set(attr, " ".join(map(str, np.fromstring(g.get(attr), sep=" ") * scale)))
        g.set("mass", str(float(g.get("mass")) * scale**3))
        g.set("friction", f"{friction} .005 .0001")
    # MuJoCo combines sliding friction using the larger coefficient. Vary
    # both surfaces so a nominal low-friction case actually changes contact.
    for g in root.findall(".//body[@name='dg5fs_mount']//geom"):
        g.set("friction", f"{friction} .005 .0001")
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="workbench",
        type="box",
        pos="0 -.5 -.025",
        size=".4 .35 .025",
        rgba=".35 .4 .45 1",
        friction="1 .005 .0001",
    )
    for g in root.findall(".//geom"):
        g.set("solref", ".004 1")
        g.set("solimp", ".95 .99 .001")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output)
    return output
