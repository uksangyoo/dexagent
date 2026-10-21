"""Replay a Track as a live-page routine: joint targets + object teleports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dexagent.hand import OPEN, Controller
from dexagent.retarget.retarget import NaiveRetargeter
from dexagent.retarget.tracks import CONTROL_HZ, Track, load_tracks, resample_tracks
from dexagent.scene import WORK
from dexagent.workcell import TABLE_Z, YAW, hand_down

# Reachable seed near the scripted workcell home (not the scene's possibly bad q).
_ARM_SEED = np.array([0.0, -1.2, 1.5, -1.9, -1.57, 0.0], float)


def _object_body_index(sim, name: str) -> int:
    bodies = sim[name].bodies
    if bodies.size == 0:
        raise ValueError(f"{name!r} has no body to teleport")
    return int(bodies[0])


def teleport_object(sim, name: str, xyz, quat_xyzw) -> None:
    """Write ``body_q`` / zero ``body_qd`` for one rigid object on both state buffers."""
    if name not in sim.names():
        raise KeyError(
            f"cannot teleport {name!r}; not in scene. available: {sorted(sim.names())}"
        )
    body = _object_body_index(sim, name)
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = np.asarray(xyz, float)
    pose[3:] = np.asarray(quat_xyzw, float)
    zero = np.zeros(6, dtype=np.float32)
    for state in (sim.pipeline.state_0, sim.pipeline.state_1):
        q = state.body_q.numpy()
        qd = state.body_qd.numpy()
        q[body] = pose
        qd[body] = zero
        state.body_q.assign(q)
        state.body_qd.assign(qd)


def _snap_robot(sim, arm_idx, hand_idx, arm, hand) -> None:
    """Teleport arm+hand joints (and targets) so IK starts from a known pose."""
    arm = np.asarray(arm, float).reshape(-1)
    hand = np.asarray(hand, float).reshape(-1)
    for state in (sim.pipeline.state_0, sim.pipeline.state_1):
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[arm_idx] = arm
        q[hand_idx] = hand
        # joint_qd is shorter than joint_q when the scene has free joints
        # (7 q vs 6 v); zero the whole velocity buffer instead of indexing by qpos.
        qd[:] = 0.0
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
    target = sim.pipeline.control.joint_target_q.numpy()
    target[arm_idx] = arm
    target[hand_idx] = hand
    sim.pipeline.control.joint_target_q.assign(target)


def _align_robot(controller, retargeter, frame, names, *, settle_steps: int = 15):
    """IK the first track frame from a WORK-home seed, snap the sim there, settle.

    Yields once per settle step. Leaves ``retargeter`` seeded on the aligned pose.
    """
    from dexagent.workcell import Kinematics

    kin = Kinematics()
    home_xyz = np.array([WORK[0], WORK[1] - 0.05, TABLE_Z + 0.14], float)
    home_arm = kin.solve_arm(home_xyz, hand_down(YAW), _ARM_SEED.copy())
    retargeter.seed_from_sim(home_arm, OPEN.copy())
    arm, hand, info = retargeter.solve(frame.palm_xyz, frame.palm_R, frame.tips)
    _snap_robot(controller.sim, controller.arm_idx, controller.hand_idx, arm, hand)
    retargeter.seed_from_sim(arm, hand)
    controller.arm = arm.copy()
    controller.hand = hand.copy()
    controller.xyz, controller.R = controller.kin.palm_pose(arm)
    controller.integral[:] = 0.0
    objs = {n: frame.objects[n] for n in names if n in frame.objects}
    for name, (xyz, quat) in objs.items():
        teleport_object(controller.sim, name, xyz, quat)
    for _ in range(max(0, int(settle_steps))):
        controller.apply_frame(arm, hand, objs, phase="align")
        yield
    controller.events.append(
        {
            "align_arm_ok": bool(info.get("arm_ok", True)),
            "align_finger_err_mm": info.get("finger_err_mm"),
        }
    )


class RetargetController(Controller):
    """Controller that accepts explicit joint targets and teleports objects."""

    def apply_frame(self, arm, hand, objects: dict | None = None, phase="retarget"):
        self.phase = phase
        self.arm = np.asarray(arm, float).copy()
        self.hand = np.asarray(hand, float).copy()
        self.xyz, self.R = self.kin.palm_pose(self.arm)
        if objects:
            for name, (xyz, quat) in objects.items():
                teleport_object(self.sim, name, xyz, quat)
        self.step()


def play_track(sim, track: Track, session=None, objects=None, hz=CONTROL_HZ, align: bool = True):
    """Generator: one yield per control step.

    After the generator is exhausted, read ``play_track.last`` for
    ``(controller, outcome, metrics)``.

    By default the robot is snapped to an IK solution of the first track frame
    (seeded from a reachable WORK home) before playback starts.
    """
    track = resample_tracks(track, hz=hz)
    names = tuple(objects) if objects is not None else tuple(
        sorted({n for fr in track.frames for n in fr.objects})
    )
    c = RetargetController(sim, session, objects=names)
    retargeter = NaiveRetargeter()
    outcome, metrics = "incomplete", {"frames": len(track.frames)}
    arm_fails = 0
    try:
        if align and track.frames:
            yield from _align_robot(c, retargeter, track.frames[0], names)
        else:
            q = sim.state.joint_q.numpy()
            retargeter.seed_from_sim(q[c.arm_idx], q[c.hand_idx])

        for fr in track.frames:
            arm, hand, info = retargeter.solve(fr.palm_xyz, fr.palm_R, fr.tips)
            if not info.get("arm_ok", True):
                arm_fails += 1
            objs = {n: fr.objects[n] for n in names if n in fr.objects}
            c.apply_frame(arm, hand, objs)
            yield
        metrics["arm_ik_failures"] = arm_fails
        outcome = "replayed" if arm_fails == 0 else "replayed_with_ik_failures"
    finally:
        c.events.append({"arm_ik_failures": arm_fails})
        play_track.last = (c, outcome, metrics)


def resolve_track_path(*candidates: Path | str) -> Path:
    """First existing path among ``candidates``, else the last candidate (for error messages)."""
    paths = [Path(p) for p in candidates]
    for path in paths:
        if path.is_file():
            return path
    return paths[-1]


def make_replay_routine(track_path, *, objects=None, results_dir: Path | str | None = None):
    """Build a ``routines`` entry that loads tracks and replays them.

    ``track_path`` may be a path, a sequence of candidate paths (first hit wins),
    or a zero-arg callable returning a path — so a scene can prefer
    ``from_video.json`` over ``synthetic.json`` at play time.
    """

    def _path() -> Path:
        if callable(track_path):
            return Path(track_path())
        if isinstance(track_path, (list, tuple)):
            return resolve_track_path(*track_path)
        return Path(track_path)

    def routine(sim, session=None):
        path = _path()
        if not path.is_file():
            raise FileNotFoundError(
                f"retarget track not found: {path}. "
                "Record one with `python -m dexagent.retarget.record_from_routine` "
                "or estimate from video with `python -m dexagent.retarget.from_video`."
            )
        track = load_tracks(path)
        play_track.last = None
        yield from play_track(sim, track, session=session, objects=objects)
        if play_track.last is not None and results_dir is not None:
            c, outcome, metrics = play_track.last
            c.finish("retarget_replay", outcome, metrics, Path(results_dir))

    return routine
