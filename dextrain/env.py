"""Batched Newton/MuJoCo-Warp contact-driven motor curriculum."""

import json
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp


class NutSpinEnv:
    def __init__(
        self, scene, num_envs=1024, seed=0, substeps=10, capture=True, disable_object_contacts=False
    ):
        wp.init()
        wp.set_device("cuda:0")
        self.torch_stream = torch.cuda.Stream(device="cuda:0")
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        newton.use_coord_layout_targets = True
        torch.manual_seed(seed)
        self.num_envs = num_envs
        self.substeps = substeps
        assert substeps % 2 == 0, "Even substeps keep captured state buffers stable"
        self.dt = 0.002 * substeps
        self.device = "cuda:0"
        self.num_actions = 20
        self.max_episode_steps = 500
        meta = json.loads(Path(scene).with_suffix(".json").read_text())
        robot = newton.ModelBuilder()
        robot.add_mjcf(
            str(scene), parse_visuals=False, enable_self_collisions=True, collapse_fixed_joints=True
        )
        # Match MuJoCo's rigid collision convention. Raw triangle meshes create
        # thousands of redundant contacts and are inappropriate for this hand.
        robot.approximate_meshes(method="convex_hull", keep_visual_shapes=False)
        # The Newton builder defaults to a 10 cm detection gap, larger than
        # this entire grasp. Use a 0.5 mm per-shape detection gap instead.
        robot.shape_gap[:] = [0.0005] * robot.shape_count
        # Fixed-joint collapse can leave old shape filters incomplete. Shapes
        # belonging to the same rigid body never exert contact on one another.
        for a in range(robot.shape_count):
            for b in range(a + 1, robot.shape_count):
                if robot.shape_body[a] == robot.shape_body[b]:
                    robot.shape_collision_filter_pairs.append((a, b))
        if disable_object_contacts:
            # Negative control: retain the object's geometry, inertia, joint,
            # and action mapping, but remove every object collision pair.
            object_bodies = {
                i for i, name in enumerate(robot.body_label) if name.split("/")[-1] == "nut"
            }
            if len(object_bodies) != 1:
                raise ValueError("Expected one nut body for the contact-disabled control")
            for a in range(robot.shape_count):
                for b in range(a + 1, robot.shape_count):
                    if robot.shape_body[a] in object_bodies or robot.shape_body[b] in object_bodies:
                        robot.shape_collision_filter_pairs.append((a, b))
        # Use names to detect any importer reordering before assuming the layout.
        names = [
            name.split("/")[-1]
            for i, name in enumerate(robot.joint_label)
            if robot.joint_type[i] == newton.JointType.REVOLUTE
        ]
        expected = (
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            + [f"joint_{i}_{j}" for i in range(1, 6) for j in range(1, 5)]
            + ["nut_rotation"]
        )
        if names != expected or robot.joint_dof_count != 27 or robot.joint_coord_count != 27:
            raise ValueError(f"Unexpected importer joint layout: {names}")
        robot.joint_q[:] = meta["qhome"]
        robot.joint_target_q[:] = meta["qhome"]
        builder = newton.ModelBuilder()
        builder.replicate(robot, num_envs)
        self.model = builder.finalize(device="cuda:0")
        self.home = torch.tensor(meta["qhome"], device=self.device, dtype=torch.float32)
        self.lower = wp.to_torch(self.model.joint_limit_lower).view(num_envs, 27)[0, 6:26]
        self.upper = wp.to_torch(self.model.joint_limit_upper).view(num_envs, 27)[0, 6:26]
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=30,
            ls_iterations=10,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=768,
            nconmax=256,
            use_mujoco_contacts=False,
        )
        self.state = self.model.state()
        self.other = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.contact_count = wp.to_torch(self.contacts.rigid_contact_count)
        self.constraint_count = wp.to_torch(self.solver.mjw_data.nefc)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.other)
        self.q = wp.to_torch(self.state.joint_q).view(num_envs, 27)
        self.qd = wp.to_torch(self.state.joint_qd).view(num_envs, 27)
        self.targets = wp.to_torch(self.control.joint_target_q).view(num_envs, 27)
        self.targets.copy_(self.home.expand(num_envs, -1))
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.direction = torch.ones(num_envs, device=self.device)
        self.last_action = torch.zeros((num_envs, 20), device=self.device)
        self.graph = None
        if capture:
            # Warp capture loads/compiles kernels on first use; no shared env mutation.
            with wp.ScopedCapture() as graph:
                self._simulate()
            self.graph = graph.graph
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=self.device))
        self.num_obs = self.observe().shape[-1]
        self.metadata = {
            "physics": "Newton SolverMuJoCo / mujoco_warp",
            "collision_backend": "Newton",
            "task": "passive_fixture_nut_rotation",
            "num_envs": num_envs,
            "dofs": 27,
            "controlled_dofs": 26,
            "policy_actions": 20,
            "substeps": substeps,
            "physics_dt": 0.002,
            "self_collisions": True,
            "num_obs": self.num_obs,
            "curriculum_only": True,
        }
        self.metadata["object_contacts_disabled"] = disable_object_contacts

    def _simulate(self):
        for _ in range(self.substeps):
            self.state.clear_forces()
            self.model.collide(self.state, self.contacts)
            self.solver.step(self.state, self.other, self.control, self.contacts, 0.002)
            self.state, self.other = self.other, self.state

    def observe(self):
        return torch.cat(
            (
                (self.q[:, :26] - self.home[:26]).clamp(-4, 4),
                self.qd[:, :26].clamp(-20, 20) * 0.05,
                self.last_action,
                self.direction[:, None],
                torch.sin(self.q[:, 26:27]),
                torch.cos(self.q[:, 26:27]),
                self.qd[:, 26:27].clamp(-20, 20) * 0.1,
                (self.targets[:, 6:26] - self.home[6:26]).clamp(-4, 4),
                (self.steps.float() / self.max_episode_steps)[:, None],
            ),
            dim=1,
        )

    def reset(self, mask):
        # Fixed-size masked writes support capture later and avoid CPU sync.
        noise = 0.002 * torch.randn_like(self.q)
        noise[:, :6] = 0
        noise[:, 26] = 0
        q = self.home[None, :] + noise
        q[:, 6:26] = torch.maximum(torch.minimum(q[:, 6:26], self.upper), self.lower)
        self.q.copy_(torch.where(mask[:, None], q, self.q))
        self.qd.masked_fill_(mask[:, None], 0)
        self.targets.copy_(torch.where(mask[:, None], q, self.targets))
        self.steps.masked_fill_(mask, 0)
        self.last_action.masked_fill_(mask[:, None], 0)
        dirs = torch.randint(0, 2, (self.num_envs,), device=self.device) * 2 - 1
        self.direction.copy_(torch.where(mask, dirs, self.direction))
        # Clear solver warm-start history for reset worlds.
        for name in ("qacc_warmstart", "qacc", "qfrc_applied"):
            arr = getattr(self.solver.mjw_data, name, None)
            if arr is not None:
                tensor = wp.to_torch(arr)
                tensor.masked_fill_(mask[:, None], 0)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    @torch.no_grad()
    def step(self, action):
        old = self.q[:, 26].clone()
        applied = action.clamp(-1, 1)
        self.targets[:, 6:26].add_(applied * (np.pi * self.dt))
        self.targets[:, 6:26].copy_(
            torch.maximum(torch.minimum(self.targets[:, 6:26], self.upper), self.lower)
        )
        self.targets[:, :6] = self.home[:6]
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.steps.add_(1)
        velocity = ((self.q[:, 26] - old) / self.dt) * self.direction
        pose_cost = ((self.q[:, 6:26] - self.home[6:26]) ** 2).mean(dim=1)
        action_cost = ((applied - self.last_action) ** 2).mean(dim=1)
        reward = (
            2.0 * velocity.clamp(-2, 2)
            + 0.5 * torch.exp(-((velocity - 1.0) ** 2))
            - 0.03 * pose_cost
            - 0.01 * action_cost
        )
        finite = torch.isfinite(self.q).all(dim=1) & torch.isfinite(self.qd).all(dim=1)
        unstable = (
            ~finite
            | (self.qd.abs().amax(dim=1) > 150)
            | ((self.q[:, :6] - self.home[:6]).abs().amax(dim=1) > 0.25)
        )
        reward = torch.where(unstable, torch.full_like(reward, -10), reward)
        done = unstable | (self.steps >= self.max_episode_steps)
        self.last_action.copy_(applied)
        info = {
            "signed_velocity": torch.nan_to_num(velocity).clamp(-50, 50),
            "unstable": unstable,
            "signed_turn": torch.nan_to_num(self.q[:, 26] * self.direction),
            "pose_cost": torch.nan_to_num(pose_cost),
        }
        info["contact_buffer_used"] = (
            self.contact_count.float().amax() / self.solver.mjw_data.naconmax
        )
        info["constraint_buffer_used"] = (
            self.constraint_count.float().amax() / self.solver.mjw_data.njmax
        )
        self.reset(done)
        return self.observe(), reward, done, info


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--no-capture", action="store_true")
    args = ap.parse_args()
    env = NutSpinEnv(
        Path(__file__).resolve().parents[1] / "runs/assets/scene.xml",
        args.num_envs,
        capture=not args.no_capture,
    )
    torch.cuda.synchronize()
    t = time.monotonic()
    for i in range(args.steps):
        obs, r, d, info = env.step(torch.zeros((args.num_envs, 20), device="cuda"))
        if i % 20 == 0:
            print(i, {k: float(v.float().mean()) for k, v in info.items()}, flush=True)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                **env.metadata,
                "elapsed_s": time.monotonic() - t,
                "finite": bool(torch.isfinite(obs).all()),
            }
        ),
        flush=True,
    )
