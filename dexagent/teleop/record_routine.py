"""Record demos by replaying a privileged scripted pinch-lift (no hardware).

Produces the same ``demo.npz`` schema as live teleop so ``skills.pack`` works
without Vive/Manus. Use for CI and first skill packs; switch to ``--source hw``
for real glove demos.

    python -m dexagent.teleop.record_routine tasks/hand_demos/cube_pinch --episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _manus_from_hand(controller, hand_q) -> np.ndarray:
    """Approximate Manus openness from how closed the hand is vs OPEN/pinch."""
    from dexagent.hand import OPEN

    open_q = OPEN
    closed_q = controller.pinch(0.018, helpers=(), tuck=(2, 3, 4))
    # Per finger mean |q - open| / |closed - open|
    out = np.ones(6)
    for finger, manus_i in enumerate([4, 3, 2, 1, 0]):  # thumb..little → manus channels rough
        sl = slice(4 * finger, 4 * finger + 4)
        denom = np.linalg.norm(closed_q[sl] - open_q[sl]) + 1e-6
        closedness = float(np.linalg.norm(hand_q[sl] - open_q[sl]) / denom)
        openness = float(np.clip(1.0 - closedness, 0.0, 1.0))
        if finger == 0:
            out[4] = openness
            out[5] = openness
        else:
            out[manus_i] = openness
    return out


def record_episodes(scene_dir: Path, *, episodes: int = 5, object_name: str = "cube"):
    sys.path.insert(0, str(ROOT))
    scene_path = scene_dir / "scene.py" if scene_dir.is_dir() else scene_dir
    scene_dir = scene_path.parent
    sys.path.insert(0, str(scene_dir))

    from robosimstudio_real2sim_agent.scenefile import run_scene_file

    from dexagent.hand import OPEN, Controller
    from dexagent.hom.features import HomFeatureLogger
    from dexagent.retarget.replay import _snap_robot, teleport_object
    from dexagent.skills.recorder import DemoRecorder, row_from_controller
    from dexagent.teleop._clutch_core import matrix_to_rotvec
    from dexagent.workcell import TABLE_Z, YAW, hand_down

    loaded = run_scene_file(scene_path)
    sim, ns = loaded.sim, loaded.ns
    START = np.asarray(ns["START"], float)
    QUAT = np.asarray(ns["QUAT"], float)
    EDGE = float(ns["EDGE"])
    SQUEEZE = float(ns.get("SQUEEZE", 0.018))
    HELPERS = tuple(ns.get("HELPERS", ()))
    TUCK = tuple(ns.get("TUCK", (2, 3, 4)))

    demos_dir = scene_dir / "demos"
    recorder = DemoRecorder(demos_dir, skill_id=scene_dir.name, object_name=object_name)
    c = Controller(sim, objects=(object_name,))
    saved = 0

    while saved < episodes:
        R = hand_down(YAW)
        home = c.palm_for(START + [0, 0, 0.12], R=R)
        arm = c.kin.solve_arm(home, R, c.arm.copy())
        _snap_robot(sim, c.arm_idx, c.hand_idx, arm, OPEN.copy())
        c.arm, c.hand = arm.copy(), OPEN.copy()
        c.xyz, c.R = c.kin.palm_pose(arm)
        c.integral[:] = 0.0
        teleport_object(sim, object_name, START, QUAT)
        for _ in range(15):
            c.step()

        hom = HomFeatureLogger(box_half=(EDGE / 2,) * 3)
        recorder.start(source="routine")
        z0 = float(c.pose(object_name)[0][2])
        p0, _ = c.pose(object_name)

        def log_step(engaged=True):
            tips = c.tips_world()
            obj = np.asarray(c.sim[object_name].pose, float)
            hom_row = hom.update(tips, obj[:3], 0.02)
            manus = _manus_from_hand(c, c.hand)
            tracker = np.concatenate([c.xyz, matrix_to_rotvec(c.R)])
            recorder.append(
                row_from_controller(
                    c,
                    t=float(c.sim.time),
                    engaged=engaged,
                    pose_valid=True,
                    manus_q=manus,
                    tracker_pose=tracker,
                    object_name=object_name,
                    hom=hom_row,
                )
            )

        # Approach / descend / pinch / lift (same geometry as scene pick_and_place).
        above = c.palm_for(p0 + [0, 0, 0.08], R=R)
        for _ in c.move(above, R, c.pinch(0.034, helpers=HELPERS, tuck=TUCK), 1.5, "approach"):
            log_step()
        for _ in c.move(c.palm_for(p0, R=R), R, seconds=1.0, phase="descend"):
            log_step()
        for _ in c.move(hand=c.pinch(SQUEEZE, helpers=HELPERS, tuck=TUCK), seconds=0.8, phase="pinch"):
            log_step()
        for _ in c.hold(0.3):
            log_step()
        lifted = c.palm_for(p0 + [0, 0, 0.12], R=R)
        for _ in c.move(lifted, R, seconds=1.2, phase="lift"):
            log_step()
        for _ in c.hold(0.4, "hold"):
            log_step()

        z = float(c.pose(object_name)[0][2])
        lift = z - z0
        ok = lift >= 0.08
        metrics = {"lift_mm": round(1000 * lift, 1)}
        path = recorder.stop_and_save(success=ok, metrics=metrics)
        if ok:
            saved += 1
        print(f"routine demo {path} success={ok} {metrics} ({saved}/{episodes})", flush=True)

    manifest = {"skill_id": scene_dir.name, "saved": saved, "demos_dir": str(demos_dir), "source": "routine"}
    demos_dir.mkdir(parents=True, exist_ok=True)
    (demos_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if hasattr(sim, "close"):
        sim.close()
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--object", default="cube")
    a = ap.parse_args(argv)
    record_episodes(a.scene, episodes=a.episodes, object_name=a.object)


if __name__ == "__main__":
    main()
