"""PPO training on Newton contact physics; saves local metrics and checkpoints."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .env import NutSpinEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        def network(output):
            return nn.Sequential(
                nn.Linear(obs_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, output),
            )

        self.actor = network(action_dim)
        self.critic = network(1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1)

    def act(self, obs, action=None):
        mean = self.actor(obs)
        dist = Normal(mean, self.log_std.clamp(-3, 1).exp(), validate_args=False)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(obs).squeeze(-1),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--task", choices=["spin", "precision"], default="spin")
    p.add_argument("--warm-start", type=Path)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--target-kl", type=float)
    args = p.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 3e-5 if args.task == "precision" else 3e-4
    if args.target_kl is None:
        args.target_kl = 0.03 if args.task == "precision" else 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    scene = root / "runs/assets/scene.xml"
    metadata = {
        "stage": "motor_skill_curriculum_not_agent_benchmark",
        "seed": args.seed,
        "physical_gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "source_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "newton", "warp-lang", "mujoco", "mujoco-warp"]
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.task == "precision":
        from .precision_env import PrecisionNutEnv

        env = PrecisionNutEnv(scene, args.num_envs, args.seed)
    else:
        env = NutSpinEnv(scene, args.num_envs, args.seed)
    metadata.update(env.metadata)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    policy = ActorCritic(env.num_obs, env.num_actions).cuda()
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if warm["metadata"]["scene_sha256"] != metadata["scene_sha256"]:
            raise ValueError("Warm-start scene differs")
        current = policy.state_dict()
        for name, value in warm["policy"].items():
            if name.startswith("actor."):
                if current[name].shape == value.shape:
                    current[name].copy_(value)
                elif name == "actor.0.weight" and current[name].shape[1] >= value.shape[1]:
                    current[name].zero_()
                    current[name][:, : value.shape[1]].copy_(value)
                else:
                    raise ValueError(f"Incompatible warm-start layer {name}")
        current["log_std"].copy_(warm["policy"]["log_std"].clamp(max=-1.0))
        policy.load_state_dict(current)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        if ckpt["metadata"]["scene_sha256"] != metadata["scene_sha256"]:
            raise ValueError("Checkpoint scene hash does not match")
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"]
    stop = False

    def stop_handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    def save(update):
        state = {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        tmp = args.out / "checkpoint.tmp"
        torch.save(state, tmp)
        tmp.replace(args.out / "checkpoint.pt")

    n, h = args.num_envs, args.horizon
    obs_buf = torch.empty((h, n, env.num_obs), device="cuda")
    actions = torch.empty((h, n, 20), device="cuda")
    logprobs = torch.empty((h, n), device="cuda")
    rewards = torch.empty((h, n), device="cuda")
    dones = torch.empty((h, n), device="cuda")
    values = torch.empty((h, n), device="cuda")
    advantages = torch.empty((h, n), device="cuda")
    obs = env.observe()
    start = time.monotonic()
    update = start_update
    print(json.dumps({"event": "training_started", **env.metadata}), flush=True)
    log = (args.out / "metrics.jsonl").open("a", buffering=1)
    try:
        for update in range(start_update + 1, args.updates + 1):
            tick = time.monotonic()
            vel = torch.zeros((), device="cuda")
            unstable = torch.zeros((), device="cuda")
            buffer_used = torch.zeros(2, device="cuda")
            successes = torch.zeros((), device="cuda")
            terminations = torch.zeros((), device="cuda")
            goal_error = torch.zeros((), device="cuda")
            for t in range(h):
                obs_buf[t].copy_(obs)
                with torch.no_grad():
                    a, lp, _, v = policy.act(obs)
                actions[t].copy_(a)
                logprobs[t].copy_(lp)
                values[t].copy_(v)
                obs, r, d, info = env.step(a)
                rewards[t].copy_(r)
                dones[t].copy_(d)
                vel += info["signed_velocity"].mean()
                unstable += info["unstable"].float().mean()
                if "success" in info:
                    successes += info["success"].float().sum()
                    terminations += d.float().sum()
                    goal_error += info["angle_error_deg"].mean()
                buffer_used = torch.maximum(
                    buffer_used,
                    torch.stack((info["contact_buffer_used"], info["constraint_buffer_used"])),
                )
            with torch.no_grad():
                nextv = policy.critic(obs).squeeze(-1)
                gae = torch.zeros(n, device="cuda")
                for t in reversed(range(h)):
                    alive = 1 - dones[t]
                    vnext = nextv if t == h - 1 else values[t + 1]
                    delta = rewards[t] + 0.99 * vnext * alive - values[t]
                    gae = delta + 0.99 * 0.95 * alive * gae
                    advantages[t] = gae
                returns = advantages + values
            bobs = obs_buf.reshape(-1, env.num_obs)
            ba = actions.reshape(-1, 20)
            blp = logprobs.flatten()
            badv = advantages.flatten()
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            bret = returns.flatten()
            losses = []
            kl_stopped = False
            for _epoch in range(args.epochs):
                indices = torch.randperm(n * h, device="cuda")
                for chunk in indices.split(args.minibatch):
                    _, lp, entropy, v = policy.act(bobs[chunk], ba[chunk])
                    logratio = lp - blp[chunk]
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if args.target_kl and approx_kl.detach().item() > args.target_kl:
                        kl_stopped = True
                        break
                    ploss = torch.maximum(
                        -badv[chunk] * ratio, -badv[chunk] * ratio.clamp(0.8, 1.2)
                    ).mean()
                    vloss = 0.5 * ((v - bret[chunk]) ** 2).mean()
                    loss = ploss + 0.5 * vloss - 0.005 * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    losses.append(
                        torch.stack((loss.detach(), ((ratio - 1) - logratio).mean().detach()))
                    )
                if kl_stopped:
                    break
            torch.cuda.synchronize()
            metrics = {
                "update": update,
                "env_steps": update * n * h,
                "elapsed_s": time.monotonic() - start,
                "steps_per_second": n * h / (time.monotonic() - tick),
                "reward": rewards.mean().item(),
                "signed_nut_velocity_rad_s": (vel / h).item(),
                "unstable_fraction": (unstable / h).item(),
                "loss": torch.stack(losses)[:, 0].mean().item(),
                "approx_kl": torch.stack(losses)[:, 1].mean().item(),
                "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            metrics["max_contact_buffer_fraction"] = buffer_used[0].item()
            metrics["max_constraint_buffer_fraction"] = buffer_used[1].item()
            if args.task == "precision":
                metrics["optimizer_steps"] = len(losses)
                metrics["kl_early_stopped"] = int(kl_stopped)
                metrics["successful_episodes"] = successes.item()
                metrics["terminated_episodes"] = terminations.item()
                metrics["mean_angle_error_deg"] = (goal_error / h).item()
            if not all(np.isfinite(v) for v in metrics.values()):
                raise RuntimeError(f"Nonfinite training metric: {metrics}")
            if metrics["unstable_fraction"] > 0.02:
                raise RuntimeError(
                    f"Physics health gate failed; unstable step fraction exceeds 2%: {metrics}"
                )
            if buffer_used.max().item() >= 0.95:
                raise RuntimeError(f"Physics contact/constraint buffer nearly full: {metrics}")
            log.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if update % args.save_every == 0 or update == start_update + 1:
                save(update)
            if stop or time.monotonic() - start > args.hours * 3600:
                break
        save(update)
        (args.out / "finished.json").write_text(
            json.dumps(
                {"update": update, "stop_requested": stop, "elapsed_s": time.monotonic() - start}
            )
            + "\n"
        )
    except Exception:
        save(update)
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
