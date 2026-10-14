"""Goal-conditioned 30/60/90-degree fixture rotation with a measured hold.

This stage closes the precision/termination gap in the spin curriculum. The
object remains on its passive fixture; free-body dexterity is a separate gate.
"""

import math

import torch

from .env import NutSpinEnv


class PrecisionNutEnv(NutSpinEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_angle = torch.full((self.num_envs,), math.pi / 3, device=self.device)
        self.hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.num_obs = self.observe().shape[-1]
        self.metadata.update(
            task="passive_fixture_precise_nut_rotation",
            num_obs=self.num_obs,
            goals_deg=[30, 60, 90],
            success_tolerance_deg=3,
            success_speed_rad_s=0.2,
            success_hold_s=0.2,
        )

    def reset(self, mask):
        super().reset(mask)
        if hasattr(self, "goal_angle"):
            goals = torch.randint(1, 4, (self.num_envs,), device=self.device) * math.pi / 6
            self.goal_angle.copy_(torch.where(mask, goals, self.goal_angle))
            self.hold_steps.masked_fill_(mask, 0)

    def observe(self):
        base = super().observe()
        if not hasattr(self, "goal_angle"):
            return base
        error = self.goal_angle - self.q[:, 26] * self.direction
        return torch.cat(
            (
                base,
                torch.sin(error[:, None]),
                torch.cos(error[:, None]),
                (error[:, None] / math.pi).clamp(-2, 2),
            ),
            dim=1,
        )

    @torch.no_grad()
    def step(self, action):
        target = self.goal_angle.clone()
        previous_hold = self.hold_steps.clone()
        previous_error = target - self.q[:, 26] * self.direction
        _, _, base_done, info = super().step(action)
        error = target - info["signed_turn"]
        velocity = info["signed_velocity"]
        good = (error.abs() <= math.radians(3)) & (velocity.abs() < 0.2) & ~info["unstable"]
        next_hold = torch.where(good, previous_hold + 1, 0)
        success = next_hold >= math.ceil(0.2 / self.dt)
        self.hold_steps.copy_(torch.where(base_done, 0, next_hold))
        progress = (previous_error.abs() - error.abs()).clamp(-0.2, 0.2)
        proximity = torch.exp(-((error / 0.12) ** 2))
        slow = torch.exp(-((velocity / 0.5) ** 2))
        reward = (
            30 * progress
            - 0.15 * error.abs().clamp(max=4)
            + 2 * proximity * slow
            - 0.03 * proximity * velocity.abs()
            - 0.01 * info["pose_cost"]
            + 8 * success.float()
        )
        reward = torch.where(info["unstable"], -10.0, reward)
        done = base_done | success
        info.update(
            success=success.clone(),
            target_angle_deg=torch.rad2deg(target),
            angle_error_deg=torch.rad2deg(error).abs(),
            terminal=done.clone(),
        )
        self.reset(success & ~base_done)
        return self.observe(), reward, done, info
