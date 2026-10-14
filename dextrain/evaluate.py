"""Paired held-out motor checks, including zero/random/contact-disabled controls.

Angle crossings are reported separately from precise, settled target capture.
This fixture curriculum cannot establish free-object manipulation success.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .env import NutSpinEnv
from .train import ActorCritic


def summarize(episodes):
    return {
        "episodes": len(episodes),
        "mean_signed_velocity_rad_s": float(
            np.mean([r["mean_signed_velocity_rad_s"] for r in episodes])
        ),
        "unstable_episodes": sum(r["unstable"] for r in episodes),
        "angle_crossing_rate": {
            str(a): float(np.mean([r["crossed"][str(a)] for r in episodes])) for a in (30, 60, 90)
        },
        "settled_capture_rate": {
            str(a): float(np.mean([r["settled"][str(a)] for r in episodes])) for a in (30, 60, 90)
        },
        "mean_final_signed_angle_deg": float(
            np.mean([r["final_signed_angle_deg"] for r in episodes])
        ),
    }


def evaluate(checkpoint, output, num_envs=256, batches=2, case_seed=20000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    scene = root / "runs/assets/scene.xml"
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["metadata"]["scene_sha256"] != hashlib.sha256(scene.read_bytes()).hexdigest():
        raise ValueError("Evaluation scene differs from the trained scene")
    torch.set_num_threads(4)
    regimes = ["policy_mean", "policy_sample", "zero", "random", "policy_mean_no_contacts"]
    summary = {
        "scope": "held-out reset seeds on passive fixture; no new geometry/friction; not A–F",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_update": saved["update"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_seed": saved["metadata"]["seed"],
        "case_seed": case_seed,
        "scene_sha256": saved["metadata"]["scene_sha256"],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), root / "dextrain/env.py"]
        },
        "regimes": {},
    }
    env = NutSpinEnv(scene, num_envs=num_envs, seed=case_seed)
    policy = ActorCritic(env.num_obs, env.num_actions).cuda()
    policy.load_state_dict(saved["policy"])
    policy.eval()
    with (output / "episodes.jsonl").open("w", buffering=1) as log, torch.inference_mode():
        for regime in regimes:
            if regime == "policy_mean_no_contacts":
                # Construct once more with identical inertial properties.
                env = NutSpinEnv(
                    scene, num_envs=num_envs, seed=case_seed, disable_object_contacts=True
                )
            results = []
            for batch in range(batches):
                torch.manual_seed(case_seed + batch)
                env.reset(torch.ones(num_envs, dtype=torch.bool, device="cuda"))
                env.direction.copy_(
                    torch.where(torch.arange(num_envs, device="cuda") % 2 == 0, 1.0, -1.0)
                )
                direction = env.direction.clone()
                obs = env.observe()
                total_velocity = torch.zeros(num_envs, device="cuda")
                unstable = torch.zeros(num_envs, dtype=torch.bool, device="cuda")
                crossed = torch.zeros((num_envs, 3), dtype=torch.bool, device="cuda")
                settled = crossed.clone()
                stable = torch.zeros((num_envs, 3), dtype=torch.long, device="cuda")
                angles = torch.deg2rad(torch.tensor([30.0, 60.0, 90.0], device="cuda"))
                # Ignore any reset-and-retry after a numerical failure.
                alive = torch.ones(num_envs, dtype=torch.bool, device="cuda")
                first_final = torch.zeros(num_envs, device="cuda")
                for _step in range(env.max_episode_steps):
                    if regime.startswith("policy_mean"):
                        action = policy.actor(obs)
                    elif regime == "policy_sample":
                        action = policy.act(obs)[0]
                    elif regime == "random":
                        action = 2 * torch.rand((num_envs, 20), device="cuda") - 1
                    else:
                        action = torch.zeros((num_envs, 20), device="cuda")
                    obs, _, done, info = env.step(action)
                    total_velocity += torch.where(alive, info["signed_velocity"], 0)
                    unstable |= info["unstable"]
                    turn = info["signed_turn"]
                    crossed |= (turn[:, None] >= angles) & alive[:, None]
                    good = (
                        ((turn[:, None] - angles).abs() <= np.deg2rad(3))
                        & (info["signed_velocity"].abs()[:, None] < 0.2)
                        & alive[:, None]
                    )
                    stable = torch.where(good, stable + 1, 0)
                    settled |= stable >= int(np.ceil(0.2 / env.dt))
                    first_final = torch.where(alive, turn, first_final)
                    alive &= ~done
                    if not bool(torch.isfinite(obs).all()):
                        raise RuntimeError("Nonfinite evaluation observation")
                arrays = [
                    t.cpu().numpy()
                    for t in (
                        direction,
                        total_velocity / env.max_episode_steps,
                        unstable,
                        crossed,
                        settled,
                        first_final,
                    )
                ]
                for i in range(num_envs):
                    row = {
                        "regime": regime,
                        "case_id": f"reset_{case_seed + batch}_{i}",
                        "direction": int(arrays[0][i]),
                        "mean_signed_velocity_rad_s": float(arrays[1][i]),
                        "unstable": bool(arrays[2][i]),
                        "crossed": {
                            str(a): bool(arrays[3][i, j]) for j, a in enumerate((30, 60, 90))
                        },
                        "settled": {
                            str(a): bool(arrays[4][i, j]) for j, a in enumerate((30, 60, 90))
                        },
                        "final_signed_angle_deg": float(np.rad2deg(arrays[5][i])),
                    }
                    results.append(row)
                    log.write(json.dumps(row) + "\n")
            summary["regimes"][regime] = {
                "all": summarize(results),
                "positive": summarize([r for r in results if r["direction"] == 1]),
                "negative": summarize([r for r in results if r["direction"] == -1]),
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"regime": regime, **summary["regimes"][regime]["all"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.num_envs, args.batches)


if __name__ == "__main__":
    main()
