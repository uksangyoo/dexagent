"""Shared Vive+Manus teleop collect loop for CLI and the live-page routine.

Yields once per 50 Hz control step so the RoboSimStudio viewer updates while
you clutch-drive the hand and record demos.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dexagent.hand import OPEN, Controller
from dexagent.hom.features import HomFeatureLogger
from dexagent.retarget.replay import _snap_robot, teleport_object
from dexagent.skills.recorder import DemoRecorder, row_from_controller
from dexagent.teleop.clutch import ClutchPalmDriver
from dexagent.teleop.keys import KeyPoller
from dexagent.teleop.manus_map import ManusHandMapper
from dexagent.teleop.sources import TeleopSample, make_source
from dexagent.workcell import YAW, hand_down

DT = 0.02
LIFT_SUCCESS_M = 0.08
TRACKING_GRACE_S = 0.25
MAX_LAG_M = 0.12


def _status(session, msg: str) -> None:
    print(msg, flush=True)
    if session is not None and hasattr(session, "status"):
        try:
            session.status = msg
        except Exception:
            pass


def _reset(sim, controller, object_name, start_xyz, start_quat, home_xyz, home_R):
    arm = controller.kin.solve_arm(home_xyz, home_R, controller.arm.copy())
    _snap_robot(sim, controller.arm_idx, controller.hand_idx, arm, OPEN.copy())
    controller.arm = arm.copy()
    controller.hand = OPEN.copy()
    controller.xyz, controller.R = controller.kin.palm_pose(arm)
    controller.integral[:] = 0.0
    teleport_object(sim, object_name, start_xyz, start_quat)
    for _ in range(10):
        controller.step()


def run_teleop_collect(
    sim,
    session=None,
    *,
    scene_dir: Path,
    source_kind: str = "hw",
    object_name: str = "cube",
    start_xyz=None,
    start_quat=None,
    edge: float = 0.04,
    squeeze: float = 0.018,
    helpers=(),
    tuck=(2, 3, 4),
    yaw_cal: Path | None = None,
    manus_cal: Path | None = None,
    max_episodes: int | None = None,
    headless_keys: bool = False,
):
    """Generator: teleop + record until ``q`` or ``max_episodes`` successes.

    Keys (OS-wide via pynput): ``c`` clutch, ``n`` new episode, ``s`` save, ``q`` quit.
    """
    scene_dir = Path(scene_dir)
    from dexagent.hand import PINCH_CENTRE
    from dexagent.scene import WORK

    if start_xyz is None:
        start_xyz = WORK + np.array([0.0, 0.0, edge / 2])
    if start_quat is None:
        start_quat = (0.0, 0.0, np.sin(YAW / 2), np.cos(YAW / 2))
    start_xyz = np.asarray(start_xyz, float)
    start_quat = np.asarray(start_quat, float)

    home_R = hand_down(YAW)
    home_xyz = start_xyz + np.array([0.0, 0.0, 0.12]) - home_R @ PINCH_CENTRE

    controller = Controller(sim, session, objects=(object_name,))
    _reset(sim, controller, object_name, start_xyz, start_quat, home_xyz, home_R)

    world_frame = source_kind in ("mock", "synthetic")
    clutch = ClutchPalmDriver(
        yaw_cal_path=yaw_cal,
        world_frame_tracker=world_frame,
        max_pos_speed=2.0 if world_frame else 0.35,
        max_rot_speed=4.0 if world_frame else 1.0,
    )
    clutch.hold(controller.xyz, controller.R)
    mapper = ManusHandMapper(half_width=squeeze)
    recorder = DemoRecorder(scene_dir / "demos", skill_id=scene_dir.name, object_name=object_name)
    hom = HomFeatureLogger(box_half=(edge / 2,) * 3)

    src_kwargs = {}
    if manus_cal is not None:
        src_kwargs["manus_cal"] = manus_cal
    if world_frame:
        src_kwargs["period_s"] = 8.0
        src_kwargs["auto_keys"] = False
    source = make_source(source_kind, **src_kwargs)
    source.start()

    keys = None if headless_keys else KeyPoller()
    previous: set[str] = set()
    t_last_valid = time.time()
    episode_z0 = float(controller.pose(object_name)[0][2])
    saved = 0
    recorder.start(source=source_kind)

    _status(
        session,
        f"teleop collect — c=clutch  n=new  s=save  q=quit  |  source={source_kind}  "
        f"yaw={getattr(clutch, 'yaw_deg', 0):.1f}°",
    )

    try:
        while max_episodes is None or saved < max_episodes:
            t_loop = time.time()
            sample = source.sample()
            held = keys.held() if keys is not None else set()
            pressed = held - previous
            previous = held

            if "q" in pressed:
                _status(session, "teleop quit")
                break

            if sample.pose_valid:
                t_last_valid = t_loop

            if "c" in pressed:
                if clutch.engaged:
                    clutch.disengage()
                    _status(session, "clutch RELEASED — press c to engage")
                elif sample.pose_valid:
                    clutch.engage(sample.tracker_pose, controller.xyz, controller.R)
                    _status(session, "clutch ENGAGED — move tracker; n=reset s=save")
                else:
                    _status(session, "cannot engage: tracker pose invalid")

            if "n" in pressed:
                if recorder.recording:
                    recorder.cancel()
                clutch.disengage()
                _reset(sim, controller, object_name, start_xyz, start_quat, home_xyz, home_R)
                clutch.hold(controller.xyz, controller.R)
                hom.reset()
                episode_z0 = float(controller.pose(object_name)[0][2])
                recorder.start(source=source_kind)
                _status(session, f"new episode (saved {saved})")

            if "s" in pressed and recorder.recording:
                z = float(controller.pose(object_name)[0][2])
                lift = z - episode_z0
                ok = lift >= LIFT_SUCCESS_M
                metrics = {"lift_mm": round(1000 * lift, 1)}
                path = recorder.stop_and_save(success=ok, metrics=metrics)
                if ok:
                    saved += 1
                _status(session, f"saved {path} success={ok} {metrics} ({saved})")
                clutch.disengage()
                recorder.start(source=source_kind)

            if clutch.engaged and not sample.pose_valid and (t_loop - t_last_valid) > TRACKING_GRACE_S:
                clutch.disengage()
                _status(session, "tracking lost — clutch released")

            if clutch.engaged and sample.pose_valid:
                if world_frame:
                    from dexagent.teleop._clutch_core import rotvec_to_matrix

                    xyz = np.asarray(sample.tracker_pose[:3], float)
                    R = rotvec_to_matrix(sample.tracker_pose[3:])
                else:
                    xyz, R = clutch.step(sample.tracker_pose, DT)
                    lag = float(np.linalg.norm(xyz - controller.xyz))
                    if lag > MAX_LAG_M:
                        clutch.disengage()
                        clutch.hold(controller.xyz, controller.R)
                        _status(session, f"lag {lag:.3f} m — clutch released")
                        xyz, R = controller.xyz, controller.R
                controller.xyz, controller.R = xyz, R
                controller.arm = controller.kin.solve_arm(xyz, R, controller.arm, tol=0.005)
            else:
                clutch.hold(controller.xyz, controller.R)

            if float(sample.manus_q[3]) < 0.15 and float(sample.manus_q[4]) < 0.15:
                controller.hand = controller.pinch(squeeze, helpers=helpers, tuck=tuck)
            else:
                controller.hand = mapper.map(sample.manus_q)

            controller.phase = "engaged" if clutch.engaged else "idle"
            controller.step()

            obj_pose = np.asarray(controller.sim[object_name].pose, float)
            hom_row = hom.update(controller.tips_world(), obj_pose[:3], DT)
            if recorder.recording:
                recorder.append(
                    row_from_controller(
                        controller,
                        t=float(controller.sim.time),
                        engaged=clutch.engaged,
                        pose_valid=sample.pose_valid,
                        manus_q=sample.manus_q,
                        tracker_pose=sample.tracker_pose,
                        object_name=object_name,
                        hom=hom_row,
                    )
                )
            yield
    finally:
        source.close()
        if keys is not None:
            keys.close()
        _status(session, f"teleop done — {saved} successful demos in {scene_dir / 'demos'}")
