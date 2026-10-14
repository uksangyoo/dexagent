"""Free-object nut benchmark; passive-fixture training cannot pass readiness."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.sim.object_state import matrix, pose, pose_error

from .common import TaskVerdict


def build_scene(base_scene, case, destination):
    """Make a separate randomized free nut scene without touching training assets."""
    base_scene = Path(base_scene)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scene = ET.parse(base_scene)
    root = scene.getroot()
    nut = root.find(".//body[@name='nut']")
    if nut is None:
        raise ValueError("No nut in source scene")
    for joint in list(nut.findall("joint")):
        nut.remove(joint)
    ET.SubElement(nut, "freejoint", name="object_free")
    for geom in nut.findall("geom"):
        for key in ("size", "pos"):
            if geom.get(key):
                geom.set(
                    key,
                    " ".join(
                        map(str, np.fromstring(geom.get(key), sep=" ") * case.dimension_scale)
                    ),
                )
        geom.set("mass", str(float(geom.get("mass", ".012")) * case.dimension_scale**3))
        geom.set("friction", f"{case.friction} .005 .0001")
    # Unique visible face marker makes 60/90-degree targets observable despite
    # the hex nut's geometric rotational symmetry when rendering is configured.
    nut.find("geom").set("rgba", "0.9 0.15 0.1 1")
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    dest = destination / "scene.xml"
    scene.write(dest)
    model = mujoco.MjModel.from_xml_path(str(dest))
    q = model.qpos0.copy()
    original = mujoco.MjModel.from_xml_path(str(base_scene))
    q[:26] = original.key_qpos[0, :26]
    q[6:26] += case.grasp_joint_noise_rad
    limits = model.jnt_range[:26]
    q[:26] = np.clip(q[:26], limits[:, 0], limits[:, 1])
    free = model.joint("object_free").qposadr[0]
    q[free : free + 3] += case.translation_m
    r = Rotation.from_rotvec(case.orientation_rotvec_rad) * Rotation.from_quat(
        q[free + 3 : free + 7], scalar_first=True
    )
    q[free + 3 : free + 7] = r.as_quat(scalar_first=True)
    k = ET.SubElement(root, "keyframe")
    ET.SubElement(
        k, "key", name="home", qpos=" ".join(map(str, q)), ctrl=" ".join(map(str, q[:26]))
    )
    ET.indent(root)
    scene.write(dest)
    # Record initialization quality separately; invalid cases cannot silently
    # become manipulation failures or disappear from a paired comparison.
    model = mujoco.MjModel.from_xml_path(str(dest))
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    min_dist = min((float(c.dist) for c in d.contact), default=0.0)
    meta = {
        "case": case.to_dict(),
        "object_is_free": True,
        "initial_min_distance_m": min_dist,
        "initialization_valid": min_dist >= -0.001,
        "ready_for_agent_benchmark": False,
        "pending": "Validate stable initial grasp and controller calibration in Newton",
    }
    (destination / "case.json").write_text(json.dumps(meta, indent=2) + "\n")
    return dest


class NutReorientation:
    def __init__(self, initial_state, angle_deg, *, tolerance_deg=3.0, hold_s=0.2, timeout_s=10.0):
        if not initial_state.object_state.free_body:
            raise ValueError("A passive revolute fixture is not the free-nut benchmark")
        start = matrix(initial_state.object_state.pose_palm)
        target = start.copy()
        target[:3, :3] = (
            start[:3, :3] @ Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
        )
        self.target = pose(target)
        self.start_time = initial_state.time_s
        self.hold_start = None
        self.tolerance = tolerance_deg
        self.hold_s = hold_s
        self.timeout_s = timeout_s
        self.drop_seen = False

    def update(self, state):
        p, r = pose_error(state.object_state.pose_palm, self.target)
        speed = np.linalg.norm(state.object_state.twist_palm.angular)
        self.drop_seen |= np.linalg.norm(state.object_state.pose_palm.position) > 0.25
        loaded = sum(
            sum(c.normal_force_N or 0.0 for c in f.contacts) >= 0.05 for f in state.fingers.values()
        )
        good = (
            r <= self.tolerance and p <= 0.01 and speed < 0.2 and loaded >= 2 and not self.drop_seen
        )
        self.hold_start = (
            (state.time_s if self.hold_start is None else self.hold_start) if good else None
        )
        success = bool(
            self.hold_start is not None and state.time_s - self.hold_start >= self.hold_s
        )
        terminal = success or self.drop_seen or state.time_s - self.start_time >= self.timeout_s
        return TaskVerdict(
            success,
            terminal,
            {"rotation_error_deg": r, "position_error_m": p, "dropped": bool(self.drop_seen)},
            "success"
            if success
            else "dropped"
            if self.drop_seen
            else "running"
            if not terminal
            else "timeout",
        )
