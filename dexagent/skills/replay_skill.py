"""Open-loop replay of a packed skill's mean palm + hand trajectory (sanity check).

    python -m dexagent.skills.replay_skill tasks/hand_demos/cube_pinch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def make_replay_routine(scene_dir: Path):
    scene_dir = Path(scene_dir)

    def routine(sim, session=None):
        yield from _replay_on_sim(sim, scene_dir, session=session)

    return routine


def _replay_on_sim(sim, scene_dir: Path, session=None):
    from dexagent.hand import OPEN, Controller
    from dexagent.retarget.replay import _snap_robot, teleport_object
    from dexagent.scene import WORK
    from dexagent.workcell import YAW, hand_down

    skill_dir = Path(scene_dir) / "demos" / "skill"
    meta = json.loads((skill_dir / "skill.json").read_text())
    with np.load(skill_dir / "skill.npz") as z:
        # Prefer exemplar demo (coherent pinch); mean softens contact.
        if "palm_xyz_exemplar" in z.files:
            palm_xyz = z["palm_xyz_exemplar"]
            hand_q = z["hand_q_exemplar"]
        else:
            palm_xyz = z["palm_xyz_mean"]
            hand_q = z["hand_q_mean"]

    EDGE = 0.04
    START = WORK + np.array([0.0, 0.0, EDGE / 2])
    QUAT = (0.0, 0.0, np.sin(YAW / 2), np.cos(YAW / 2))
    object_name = "cube"
    c = Controller(sim, session, objects=(object_name,))
    R0 = hand_down(YAW)
    arm = c.kin.solve_arm(palm_xyz[0], R0, c.arm.copy())
    _snap_robot(sim, c.arm_idx, c.hand_idx, arm, OPEN.copy())
    c.arm, c.hand = arm.copy(), OPEN.copy()
    c.xyz, c.R = c.kin.palm_pose(arm)
    teleport_object(sim, object_name, START, QUAT)
    for _ in range(15):
        c.step()
        yield
    z0 = float(c.pose(object_name)[0][2])
    for k in range(len(palm_xyz)):
        xyz = palm_xyz[k]
        c.xyz, c.R = xyz, R0
        c.arm = c.kin.solve_arm(xyz, R0, c.arm, tol=0.006)
        c.hand = hand_q[k]
        c.phase = f"skill/{k}"
        c.step()
        yield
    z = float(c.pose(object_name)[0][2])
    metrics = {"lift_mm": round(1000 * (z - z0), 1), "n_frames": int(len(palm_xyz))}
    outcome = "success" if (z - z0) >= 0.05 else "no_lift"
    c.finish("skill_replay", outcome, metrics, Path(scene_dir) / "results")
    print(f"skill replay {outcome} {metrics}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    a = ap.parse_args(argv)
    scene_path = a.scene / "scene.py" if a.scene.is_dir() else a.scene
    scene_dir = scene_path.parent
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(scene_dir))
    from robosimstudio_real2sim_agent.scenefile import run_scene_file

    loaded = run_scene_file(scene_path)
    for _ in _replay_on_sim(loaded.sim, scene_dir):
        pass
    if hasattr(loaded.sim, "close"):
        loaded.sim.close()


if __name__ == "__main__":
    main()
