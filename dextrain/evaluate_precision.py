"""Evaluate angle-and-hold goals from fresh reset seeds, with paired controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .precision_env import PrecisionNutEnv
from .train import ActorCritic


def evaluate(checkpoint, output, num_envs=384, batches=4):
    checkpoint, output = Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene = Path(__file__).resolve().parents[1] / "runs/assets/scene.xml"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["metadata"]["task"] != "passive_fixture_precise_nut_rotation":
        raise ValueError("Expected a precision checkpoint")
    if hashlib.sha256(scene.read_bytes()).hexdigest() != saved["metadata"]["scene_sha256"]:
        raise ValueError("Checkpoint scene differs")
    if num_envs % 6:
        raise ValueError("Use a multiple of six environments for paired angle/direction targets")
    torch.set_num_threads(4)
    env = PrecisionNutEnv(scene, num_envs=num_envs, seed=40000)
    policy = ActorCritic(env.num_obs, env.num_actions).cuda()
    policy.load_state_dict(saved["policy"])
    policy.eval()
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_update": saved["update"],
        "scope": "precision motor curriculum on passive fixture; no free-body or A–F success claim",
        "cases": "fresh reset seeds 40000+, same geometry/friction",
        "regimes": {},
    }
    targets = torch.tensor([30.0, 60.0, 90.0, -30.0, -60.0, -90.0], device="cuda").repeat(
        num_envs // 6
    )
    with (output / "episodes.jsonl").open("w", buffering=1) as log, torch.inference_mode():
        for mode in ("policy_mean", "policy_sample", "zero", "policy_mean_no_contacts"):
            if mode.endswith("no_contacts"):
                env = PrecisionNutEnv(
                    scene, num_envs=num_envs, seed=40000, disable_object_contacts=True
                )
            rows = []
            for batch in range(batches):
                torch.manual_seed(40000 + batch)
                env.reset(torch.ones(num_envs, dtype=torch.bool, device="cuda"))
                env.goal_angle.copy_(torch.deg2rad(targets.abs()))
                env.direction.copy_(torch.sign(targets))
                obs = env.observe()
                alive = torch.ones(num_envs, dtype=torch.bool, device="cuda")
                success = torch.zeros_like(alive)
                unstable = torch.zeros_like(alive)
                final_error = torch.full((num_envs,), float("inf"), device="cuda")
                elapsed = torch.zeros(num_envs, device="cuda")
                for step in range(env.max_episode_steps):
                    if mode.startswith("policy_mean"):
                        action = policy.actor(obs)
                    elif mode == "policy_sample":
                        action = policy.act(obs)[0]
                    else:
                        action = torch.zeros((num_envs, 20), device="cuda")
                    obs, _, done, info = env.step(action)
                    success |= info["success"] & alive
                    unstable |= info["unstable"] & alive
                    final_error = torch.where(alive, info["angle_error_deg"], final_error)
                    elapsed = torch.where(alive, (step + 1) * env.dt, elapsed)
                    alive &= ~done
                values = [
                    t.cpu().numpy() for t in (targets, success, unstable, final_error, elapsed)
                ]
                for i in range(num_envs):
                    row = {
                        "mode": mode,
                        "case_id": f"precision_{40000 + batch}_{i}",
                        "target_deg": float(values[0][i]),
                        "success": bool(values[1][i]),
                        "unstable": bool(values[2][i]),
                        "final_error_deg": float(values[3][i]),
                        "elapsed_s": float(values[4][i]),
                    }
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
            summary["regimes"][mode] = {
                str(angle): {
                    "episodes": len(group := [r for r in rows if r["target_deg"] == angle]),
                    "successes": sum(r["success"] for r in group),
                    "success_rate": float(np.mean([r["success"] for r in group])),
                    "unstable_episodes": sum(r["unstable"] for r in group),
                    "mean_final_error_deg": float(np.mean([r["final_error_deg"] for r in group])),
                }
                for angle in (30, 60, 90, -30, -60, -90)
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "episodes": len(rows),
                        "success_rate": float(np.mean([r["success"] for r in rows])),
                    }
                ),
                flush=True,
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.num_envs, args.batches)


if __name__ == "__main__":
    main()
