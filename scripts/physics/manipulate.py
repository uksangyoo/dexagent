"""Newton free-body motion/hold diagnostics from a dynamically settled grasp."""

import argparse
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.controllers.material_servo import MaterialServo
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import Constraints
from dexagent.sim.object_state import matrix
from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv
from scripts.physics.probe_free import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", choices=["rotate", "translate", "transfer", "release", "hold"], default="rotate"
    )
    p.add_argument("--angle", type=float, default=15)
    p.add_argument("--axis", type=int, default=2)
    p.add_argument("--name", default="rotation")
    p.add_argument("--repatch", action="store_true")
    p.add_argument("--passive", action="store_true")
    p.add_argument("--distance", type=float, default=0.008)
    p.add_argument("--preload", type=float, default=0.6)
    p.add_argument("--force-gain", type=float, default=0.00001)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--integral", action="store_true")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--feedback", action="store_true")
    p.add_argument("--object", choices=["nut", "wrench", "screw"], default="nut")
    args = p.parse_args()
    from dexagent.tasks.industrial_scenes import build_free_scene

    path = build_free_scene(OUT / f"free_{args.object}.xml", kind=args.object)
    env = UR5eTesolloEnv(path, device="cpu", substeps=5)
    servo = MaterialServo(env)
    init = json.loads(
        (
            OUT
            / (
                "ready_grasp_initialization.json"
                if args.object == "nut"
                else "wrench_recruit_initialization.json"
                if args.object == "wrench"
                else f"{args.object}_grasp_initialization.json"
            )
        ).read_text()
    )
    env.reset(np.array(init["qpos"]), actuator_targets=np.array(init["actuator_targets"]))
    samples = []
    qs = []
    patches = {}
    start = None
    goal = None
    violation = None
    limits = Constraints(max_normal_force_N=5, min_support_contacts=0, prevent_object_drop=False)
    quality = dict(
        samples=0,
        max_depth_m=0.0,
        max_self_depth_m=0.0,
        max_finger_force_N=0.0,
        max_external_contact_force_N=0.0,
        minimum_loaded_fingers=5,
    )

    def audit(state):
        quality["samples"] += 1
        quality["max_depth_m"] = max(
            quality["max_depth_m"],
            max((-c.separation_m for v in state.fingers.values() for c in v.contacts), default=0),
        )
        quality["max_self_depth_m"] = max(
            quality["max_self_depth_m"], state.self_collision_penetration_m
        )
        forces = [sum(c.normal_force_N or 0 for c in v.contacts) for v in state.fingers.values()]
        quality["max_finger_force_N"] = max(quality["max_finger_force_N"], max(forces))
        quality["max_external_contact_force_N"] = max(
            quality["max_external_contact_force_N"], state.object_state.external_contact_force_N
        )
        quality["minimum_loaded_fingers"] = min(
            quality["minimum_loaded_fingers"], sum(f >= 0.05 for f in forces)
        )

    env.observation_listeners.append(audit)
    wall = time.monotonic()
    rotation_bias = np.zeros(3)
    translation_bias = np.zeros(3)
    for k in range(args.steps):
        s = env.observe()
        obj = matrix(s.object_state.pose_world)
        targets = []
        phase = "settle"
        if k == (0 if args.feedback else 100):
            start = obj.copy()
            for f, v in s.fingers.items():
                if sum(c.normal_force_N or 0 for c in v.contacts) < 0.05:
                    continue
                c = max(v.contacts, key=lambda c: c.normal_force_N or 0)
                patches[f] = (
                    c.finger_link,
                    np.array(c.point_finger),
                    np.array(c.point_object),
                    start[:3, :3].T @ c.normal_world,
                    c.separation_m,
                )
                env.roles[f] = FingerRole.SUPPORT
            print("initial supports", list(patches), flush=True)
        if args.repatch and k > 0 and k % 50 == 0:
            for f, v in s.fingers.items():
                if sum(c.normal_force_N or 0 for c in v.contacts) < 0.05:
                    continue
                c = max(v.contacts, key=lambda c: c.normal_force_N or 0)
                patches[f] = (
                    c.finger_link,
                    np.array(c.point_finger),
                    np.array(c.point_object),
                    obj[:3, :3].T @ c.normal_world,
                    c.separation_m,
                )
        if k >= (0 if args.feedback else 100):
            phase = args.task if k < 400 else "hold"
            alpha = float(np.clip((k - 100) / 300, 0, 1))
            goal = start.copy()
            axis = np.eye(3)[args.axis]
            if args.task == "rotate":
                goal[:3, :3] = (
                    Rotation.from_rotvec(axis * np.deg2rad(args.angle) * alpha).as_matrix()
                    @ start[:3, :3]
                )
            if args.task == "translate":
                goal[:3, 3] += axis * args.distance * alpha
            control_goal = goal.copy()
            if args.integral and k >= 100:
                rotation_bias += (
                    0.8 * env.dt * Rotation.from_matrix(goal[:3, :3] @ obj[:3, :3].T).as_rotvec()
                )
                rotation_bias *= min(1, np.deg2rad(20) / max(np.linalg.norm(rotation_bias), 1e-12))
                translation_bias += 0.5 * env.dt * (goal[:3, 3] - obj[:3, 3])
                translation_bias = np.clip(translation_bias, -0.004, 0.004)
                control_goal[:3, :3] = (
                    Rotation.from_rotvec(rotation_bias).as_matrix() @ goal[:3, :3]
                )
                control_goal[:3, 3] += translation_bias
            for f, (b, local, op, n, sep) in patches.items():
                target = control_goal[:3, 3] + control_goal[:3, :3] @ (op + n * sep)
                if args.feedback:
                    force = sum(c.normal_force_N or 0 for c in s.fingers[f].contacts)
                    desired = 0.35 if f == Finger.THUMB else 0.65
                    target -= goal[:3, :3] @ n * 0.000015 * np.clip(desired - force, -2, 2)
                if args.task == "transfer" and f == Finger.INDEX:
                    offset = 0.008 * np.sin(np.pi * float(np.clip((k - 100) / 300, 0, 1)))
                    target += goal[:3, :3] @ n * offset
                    env.roles[f] = FingerRole.REPOSITION if offset > 0.0003 else FingerRole.SUPPORT
                if args.task == "release":
                    target += goal[:3, :3] @ n * 0.025 * alpha
                    env.roles[f] = FingerRole.RELEASE
                targets.append((f, b, local, target))
        if args.adaptive and goal is not None and args.task in ("rotate", "translate", "hold"):
            desired = obj.copy()
            shift = goal[:3, 3] - obj[:3, 3]
            shift *= min(1, 0.00008 / max(np.linalg.norm(shift), 1e-12))
            rv = Rotation.from_matrix(goal[:3, :3] @ obj[:3, :3].T).as_rotvec()
            rv *= min(1, np.deg2rad(0.12) / max(np.linalg.norm(rv), 1e-12))
            desired[:3, 3] += shift
            desired[:3, :3] = Rotation.from_rotvec(rv).as_matrix() @ obj[:3, :3]
            targets = []
            for f in [Finger.THUMB, Finger.INDEX, Finger.MIDDLE]:
                cs = [c for c in s.fingers[f].contacts if (c.normal_force_N or 0) > 0.02]
                force = sum(c.normal_force_N or 0 for c in cs)
                if cs:
                    c = max(cs, key=lambda c: c.normal_force_N or 0)
                    op = np.array(c.point_object)
                    n = obj[:3, :3].T @ c.normal_world
                    target = desired[:3, 3] + desired[:3, :3] @ (op + n * c.separation_m)
                    target -= (
                        np.array(c.normal_world)
                        * args.force_gain
                        * np.clip(args.preload - force, -2, 2)
                    )
                    targets.append((f, c.finger_link, np.array(c.point_finger), target))
                    env.roles[f] = (
                        FingerRole.DRIVE
                        if f == Finger.INDEX and k >= 100 and args.task != "hold"
                        else FingerRole.SUPPORT
                    )
                else:
                    op, n, local = env.contact_target(f)
                    target = obj[:3, 3] + obj[:3, :3] @ op - n * 0.0001
                    targets.append((f, f"link_{list(Finger).index(f) + 1}_4", local, target))
                    env.roles[f] = FingerRole.REPOSITION
        env.step_targets(
            env.last_targets if args.passive else servo.command(targets), constraints=limits
        )
        s = env.observe()
        if k % 2 == 0 or env.last_constraint_violation:
            o = matrix(s.object_state.pose_world)
            samples.append(
                dict(
                    t=s.time_s,
                    phase=phase,
                    pos=s.object_state.pose_world.position,
                    quat=s.object_state.pose_world.quaternion_wxyz,
                    forces={
                        f.value: sum(c.normal_force_N or 0 for c in v.contacts)
                        for f, v in s.fingers.items()
                    },
                    roles={f.value: r.value for f, r in env.roles.items()},
                    depth=max(
                        (-c.separation_m for v in s.fingers.values() for c in v.contacts), default=0
                    ),
                    selfdepth=s.self_collision_penetration_m,
                    vel=s.object_state.twist_world.linear,
                    omega=s.object_state.twist_world.angular,
                    external_contact_force_N=s.object_state.external_contact_force_N,
                    palm_contact_force_N=s.object_state.palm_contact_force_N,
                    contact_modes={f.value: v.contact_state.value for f, v in s.fingers.items()},
                    contacts=[
                        dict(
                            finger=f.value,
                            point=c.point_world,
                            force_on_object_N=tuple(-x for x in c.force_on_finger_world_N),
                        )
                        for f, v in s.fingers.items()
                        for c in v.contacts
                        if c.force_on_finger_world_N is not None and (c.normal_force_N or 0) > 0.02
                    ],
                    position_error_m=float(np.linalg.norm(o[:3, 3] - goal[:3, 3]))
                    if goal is not None
                    else None,
                    angle_error_deg=float(
                        np.rad2deg(Rotation.from_matrix(o[:3, :3] @ goal[:3, :3].T).magnitude())
                    )
                    if goal is not None
                    else None,
                )
            )
            qs.append(env.native_q(env.sample))
        if k % 100 == 0:
            print(k, samples[-1], flush=True)
        if env.last_constraint_violation:
            violation = env.last_constraint_violation
            print("GUARD", violation, flush=True)
            break
    np.savez_compressed(OUT / f"{args.name}.npz", qpos=qs)
    (OUT / f"{args.name}.json").write_text(json.dumps(samples))
    final = samples[-1]
    last_hold = [sample for sample in samples if sample["t"] >= samples[-1]["t"] - 0.2]
    settled = all(
        sample["position_error_m"] is not None
        and sample["position_error_m"] <= 0.002
        and sample["angle_error_deg"] <= 3
        and np.linalg.norm(sample["omega"]) < 0.2
        and np.linalg.norm(sample["vel"]) < 0.02
        and sum(f >= 0.05 for f in sample["forces"].values()) >= 2
        for sample in last_hold
    )
    success = bool(
        settled
        and not violation
        and quality["max_external_contact_force_N"] < 0.02
        and quality["minimum_loaded_fingers"] >= 2
        and len(last_hold) >= 10
    )
    result = dict(
        success=success,
        quality=quality,
        task=args.task,
        object=args.object,
        feedback=args.feedback,
        adaptive=args.adaptive,
        integral=args.integral,
        preload=args.preload,
        force_gain=args.force_gain,
        passive=args.passive,
        repatch=args.repatch,
        axis=args.axis,
        angle_deg=args.angle,
        distance_m=args.distance,
        physics="Newton / MuJoCo Warp CPU",
        free_body=env.free_body,
        object_actuated=False,
        violation=violation,
        final=final,
        max_depth_m=max(s["depth"] for s in samples),
        elapsed_s=time.monotonic() - wall,
    )
    (OUT / f"{args.name}_result.json").write_text(json.dumps(result, indent=2))
    print("DONE", result, flush=True)


if __name__ == "__main__":
    main()
