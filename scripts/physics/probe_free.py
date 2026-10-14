"""CPU MuJoCo diagnostic, not a Newton benchmark result."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/physics_revision_20261012"


def scene():
    root = ET.parse(ROOT / "runs/assets/scene.xml").getroot()
    nut = root.find(".//body[@name='nut']")
    nut.remove(nut.find("joint"))
    ET.SubElement(nut, "freejoint", name="object_free")
    old = np.fromstring(root.find("./keyframe/key").get("qpos"), sep=" ")
    q = np.r_[
        old[:26], np.fromstring(nut.get("pos"), sep=" "), np.fromstring(nut.get("quat"), sep=" ")
    ]
    root.find("./keyframe/key").set("qpos", " ".join(map(str, q)))
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
    path = OUT / "free_diagnostic.xml"
    ET.ElementTree(root).write(path)
    return path


def main():
    path = scene()
    model = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(model)
    object_id = model.body("nut").id
    og = np.flatnonzero(model.geom_bodyid == object_id)
    q = model.key_qpos[0].copy()
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    patches = []
    for i in [1, 2, 3]:
        b = model.body(f"link_{i}_4").id
        fg = np.flatnonzero(model.geom_bodyid == b)
        best = None
        for g in fg:
            for o in og:
                line = np.zeros(6)
                dist = mujoco.mj_geomDistance(model, d, int(g), int(o), 1, line)
                if best is None or dist < best[0]:
                    best = (dist, line.copy())
        dist, line = best
        normal = line[:3] - line[3:]
        normal /= np.linalg.norm(normal)
        local = d.xmat[b].reshape(3, 3).T @ (line[:3] - d.xpos[b])
        patches.append((b, local, line[:3].copy(), normal))
        print("patch", i, dist, normal, flush=True)
    results = []
    for compress in [0, 0.00025, 0.0005, 0.001, 0.002]:
        mujoco.mj_resetDataKeyframe(model, d, 0)
        control = q[:26].copy()
        samples = []
        qs = []
        for k in range(1500):
            if k % 5 == 0:
                for i, (b, local, p, n) in enumerate(patches):
                    world = d.xpos[b] + d.xmat[b].reshape(3, 3) @ local
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, d, jac, None, world, b)
                    ids = np.arange(6 + i * 4, 10 + i * 4)
                    J = jac[:, ids]
                    err = (p - n * compress) - world
                    dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), err)
                    control[ids] += np.clip(dq, -0.005, 0.005)
                    control[ids] = np.clip(control[ids], d.qpos[ids] - 0.15, d.qpos[ids] + 0.15)
                control = np.clip(control, model.jnt_range[:26, 0], model.jnt_range[:26, 1])
            d.ctrl[:] = control
            mujoco.mj_step(model, d)
            forces = np.zeros(5)
            depth = 0.0
            selfdepth = 0.0
            for cidx, c in enumerate(d.contact):
                bs = [int(model.geom_bodyid[g]) for g in [c.geom1, c.geom2]]
                if object_id in bs:
                    b = bs[1 - bs.index(object_id)]
                    name = model.body(b).name
                    if name.startswith("link_") and name.split("_")[1].isdigit():
                        v = np.zeros(6)
                        mujoco.mj_contactForce(model, d, cidx, v)
                        forces[int(name.split("_")[1]) - 1] += v[0]
                        depth = max(depth, -c.dist)
                elif all(model.body(b).name.startswith("link_") for b in bs):
                    selfdepth = max(selfdepth, -c.dist)
            if k % 10 == 0:
                samples.append(
                    dict(
                        t=float(d.time),
                        pos=d.qpos[26:29].tolist(),
                        forces=forces.tolist(),
                        depth=float(depth),
                        selfdepth=float(selfdepth),
                    )
                )
                qs.append(d.qpos.copy())
        result = dict(
            compress=compress,
            drift=float(np.linalg.norm(d.qpos[26:29] - q[26:29])),
            forces=forces.tolist(),
            max_depth=max(s["depth"] for s in samples),
            final_speed=float(np.linalg.norm(d.qvel[26:29])),
        )
        results.append(result)
        print(result, flush=True)
        np.savez_compressed(OUT / f"native_compress_{compress}.npz", qpos=qs)
        (OUT / f"native_compress_{compress}.json").write_text(json.dumps(samples))
    (OUT / "native_free_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
