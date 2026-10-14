"""Capture genuine Newton states for a visual review; never drives hardware."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import newton
import numpy as np
import torch

from dextrain.env import NutSpinEnv
from dextrain.precision_env import PrecisionNutEnv
from dextrain.train import ActorCritic

ROOT = Path(__file__).resolve().parents[2]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def export_clip(output, name, q, frames, **metadata):
    np.savez_compressed(output / f"{name}.npz", qpos=np.asarray(q), time=np.array([f["time_s"] for f in frames]))
    result = {"id": name, "frames": frames, "physics": "Newton / SolverMuJoCo", **metadata}
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"clip": name, "frames": len(frames), "sim_seconds": frames[-1]["time_s"]}), flush=True)


def motor_clip(output, name, checkpoint, precision=False, mode="policy_mean"):
    frozen = output / f"{name}_checkpoint.pt"
    shutil.copyfile(checkpoint, frozen)
    saved = torch.load(frozen, map_location="cpu", weights_only=False)
    scene = ROOT / "runs/assets/scene.xml"
    scene_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
    if saved["metadata"]["scene_sha256"] != scene_hash:
        raise ValueError("Checkpoint scene hash mismatch")
    env = (PrecisionNutEnv if precision else NutSpinEnv)(
        scene, num_envs=1, seed=42042, disable_object_contacts=mode == "no_contacts"
    )
    env.direction.fill_(1)
    if precision:
        env.goal_angle.fill_(np.pi / 3)
    policy = ActorCritic(env.num_obs, env.num_actions).cuda().eval()
    policy.load_state_dict(saved["policy"])
    solved = newton.Contacts(
        max(env.solver.mjw_data.naconmax, env.contacts.rigid_contact_max), 0,
        device="cuda:0", requested_attributes={"force"},
    )
    shape_body = env.model.shape_body.numpy()
    names = [n.split("/")[-1] for n in env.model.body_label]
    obj = names.index("nut")
    frames, poses = [], []

    def record(time_s, velocity=0.0, initial=False):
        forces = dict.fromkeys(FINGERS, 0.0)
        if not initial:
            env.solver.update_contacts(solved)
            count = min(int(solved.rigid_contact_count.numpy()[0]), solved.rigid_contact_max)
            a, b = solved.rigid_contact_shape0.numpy()[:count], solved.rigid_contact_shape1.numpy()[:count]
            normal, force = solved.rigid_contact_normal.numpy()[:count], solved.force.numpy()[:count, :3]
            for i, (s0, s1) in enumerate(zip(a, b, strict=True)):
                b0, b1 = int(shape_body[s0]), int(shape_body[s1])
                finger_body = b1 if b0 == obj else (b0 if b1 == obj else -1)
                if finger_body < 0:
                    continue
                for j, finger in enumerate(FINGERS, 1):
                    if names[finger_body].startswith(f"link_{j}_"):
                        forces[finger] += float(abs(np.dot(normal[i], force[i])))
        q = env.q[0].cpu().numpy().copy()
        poses.append(q)
        frames.append({
            "time_s": round(time_s, 6), "angle_deg": float(np.rad2deg(q[26])),
            "speed_rad_s": float(velocity), "forces_N": forces,
            "forces_available": not initial,
            "phase": "policy" if mode == "policy_mean" else mode,
        })

    record(0, initial=True)
    obs = env.observe()
    terminal = None
    with torch.inference_mode():
        for step in range(499):
            action = torch.zeros((1, 20), device="cuda") if mode == "zero" else policy.actor(obs)
            obs, _, done, info = env.step(action)
            if bool(done[0]):
                # The training API autoresets; do not render its reset as physical motion.
                terminal = {"time_s": (step + 1) * env.dt, "unstable": bool(info["unstable"][0]),
                            "success": bool(info.get("success", torch.zeros(1))[0])}
                break
            if (step + 1) % 2 == 0:
                record((step + 1) * env.dt, info["signed_velocity"][0])
    export_clip(
        output, name, poses, frames, scene=str(scene), scene_sha256=scene_hash,
        checkpoint_update=saved["update"], checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        checkpoint_task=saved["metadata"]["task"], fps=25, intrinsic_speed=1.0,
        target_deg=60 if precision else None, reference_deg=60, reset_seed=42042,
        fixture=True, mode=mode, terminal=terminal,
        scope="Single illustrative fresh rollout; not an aggregate evaluation or free-body task.",
        force_timing="Forces from the last solved 2 ms step; poses at its end (2 ms offset).",
    )


def controller_clip(output, name="contact_sequence", hold_s=.4):
    from dexagent.controllers.contact_controller import ContactController
    from dexagent.grammar.finger_roles import Finger, FingerRole
    from dexagent.grammar.schema import Constraints, Termination, TransitionRequest
    from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv

    env = UR5eTesolloEnv()
    env.roles[Finger.INDEX] = FingerRole.SUPPORT
    ctl = ContactController(env)
    frames, poses, reports = [], [], []
    phase, last = "establish", -1.0

    def record(s):
        nonlocal last
        if s.time_s - last < .0199:
            return
        last = s.time_s
        q = env.native_q(env.sample)
        poses.append(q.copy())
        frames.append({"time_s": s.time_s, "angle_deg": float(np.rad2deg(q[26])),
                       "speed_rad_s": float(np.linalg.norm(s.object_state.twist_world.angular)),
                       "forces_N": {f.value: sum(c.normal_force_N or 0.0 for c in v.contacts)
                                    for f, v in s.fingers.items()},
                       "contact_modes": {f.value: v.contact_state.value for f, v in s.fingers.items()},
                       "phase": phase})

    point, _, _ = env.contact_target(Finger.INDEX, (-.022, .015, .010))
    with env.observe_each_step(record):
        for operation in ["establish_contact", "maintain_contact", "break_contact"]:
            phase = operation.replace("_contact", "")
            result = ctl.execute(TransitionRequest(
                operation=operation, object="hex_nut", finger=Finger.INDEX,
                object_point=tuple(point) if operation == "establish_contact" else None,
                constraints=Constraints(max_normal_force_N=5, min_support_contacts=0),
                termination=Termination(timeout_s=4, hold_s=hold_s),
            ))
            reports.append({"operation": operation, "success": result.success,
                            "message": result.message, "failure_code": result.failure_code})
            if not result.success:
                break
    export_clip(output, name, poses, frames, scene=str(env.scene), fps=25,
                intrinsic_speed=.5, target_deg=None, reference_deg=None, fixture=True,
                reports=reports, scope="Real index-finger establish / maintain / release; no multi-finger manipulation claim.",
                force_timing="Synchronized solved contacts and pre-integration poses.")
    env.close()


def free_clip(output):
    from dexagent.grammar.schema import Constraints
    from dexagent.sim.object_state import pose_error
    from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv

    source = ROOT / "runs/next_stage_smoke_20261012T020506Z/grasp"
    # Find the saved free scene without relying on a particular output nesting.
    scene = next(source.rglob("free_scene.xml"))
    previous = json.loads(next(source.rglob("result.json")).read_text())
    env = UR5eTesolloEnv(scene)
    q, targets = np.array(previous["initial_q"]), np.array(previous["actuator_targets"])
    env.reset(q, actuator_targets=targets)
    initial = env.observe().object_state.pose_world
    frames, poses = [], []

    def record(s):
        p, angle = pose_error(s.object_state.pose_world, initial)
        poses.append(env.native_q(env.sample).copy())
        frames.append({"time_s": s.time_s, "angle_deg": angle,
                       "position_error_mm": p * 1000,
                       "speed_rad_s": float(np.linalg.norm(s.object_state.twist_world.angular)),
                       "forces_N": {f.value: sum(c.normal_force_N or 0 for c in v.contacts)
                                    for f, v in s.fingers.items()},
                       "penetration_mm": max((-c.separation_m * 1000 for v in s.fingers.values() for c in v.contacts), default=0),
                       "phase": "unsupported hold"})

    # Replay the historical invalid candidate solely to visualize its failure.
    # The current 1 mm validation gate would reject it before task execution.
    limits = Constraints(max_normal_force_N=5, min_support_contacts=0, max_object_penetration_m=.01)
    with env.observe_each_step(record):
        for _ in range(100):
            env.step_targets(targets, constraints=limits)
            if env.last_constraint_violation:
                break
    export_clip(output, "free_grasp", poses, frames, scene=str(scene), fps=25,
                intrinsic_speed=.05, target_deg=None, reference_deg=5, fixture=False,
                failure=env.last_constraint_violation,
                scope="Fresh diagnostic replay of a historical invalid initialization. Current 1 mm penetration gate rejects this candidate; not a valid task trial.",
                force_timing="Synchronized solved contacts and pre-integration poses.")
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-cycle-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.contact_cycle_only:
        controller_clip(args.output, "contact_cycle", .2)
        return
    spin = ROOT / "runs/nut_motor_20261010T020245Z/seed_0/checkpoint.pt"
    precision = Path(json.loads((ROOT / "runs/latest_precision_campaign.json").read_text())["batch"]) / "seed_0/train/checkpoint.pt"
    for name, checkpoint, is_precision, mode in [
        ("learned_spin", spin, False, "policy_mean"),
        ("precision", precision, True, "policy_mean"),
        ("zero", spin, False, "zero"),
        ("no_contacts", spin, False, "no_contacts"),
    ]:
        motor_clip(args.output, name, checkpoint, is_precision, mode)
    controller_clip(args.output)
    free_clip(args.output)


if __name__ == "__main__":
    main()
