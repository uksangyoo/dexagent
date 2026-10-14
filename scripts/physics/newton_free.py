"""Free-object Newton CPU calibration with no pose resets during execution."""

import argparse
import json
import time

import mujoco
import numpy as np

from dexagent.grammar.finger_roles import Finger
from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv
from scripts.physics.probe_free import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=3)
    p.add_argument("--capture", default="newton_free")
    p.add_argument("--mode", choices=["hold", "close", "thumb", "recruit"], default="hold")
    p.add_argument("--object", choices=["nut", "wrench", "screw"], default="nut")
    args = p.parse_args()
    from dexagent.tasks.industrial_scenes import build_free_scene

    scene_path = build_free_scene(OUT / f"free_{args.object}.xml", kind=args.object)
    env = UR5eTesolloEnv(scene_path, device="cpu", substeps=5)
    model = env.kin_model
    d = mujoco.MjData(model)
    qs = []
    samples = []
    start = time.monotonic()
    control = env.last_targets.copy()
    for k in range(int(args.seconds / env.dt)):
        s = env.observe()
        q = env.native_q()
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        if args.mode in ("close", "thumb", "recruit") and k > 50:
            for f in (
                [Finger.MIDDLE]
                if args.mode == "recruit" and k < 200
                else [Finger.THUMB]
                if args.mode in ("thumb", "recruit")
                else [Finger.THUMB, Finger.INDEX, Finger.MIDDLE]
            ):
                op, normal, local = env.contact_target(f)
                bid = model.body(f"link_{list(Finger).index(f) + 1}_4").id
                world = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
                oid = model.body("nut").id
                target = d.xpos[oid] + d.xmat[oid].reshape(3, 3) @ op
                force = sum(c.normal_force_N or 0 for c in s.fingers[f].contacts)
                error = target - world - normal * 0.0003 * np.clip(0.6 - force, -1, 1)
                error *= min(1, 0.0002 / max(np.linalg.norm(error), 1e-9))
                ids = env.finger_q[f]
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, d, jac, None, world, bid)
                J = jac[:, ids]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), error)
                control[ids] += np.clip(dq, -0.004, 0.004)
                control[ids] = np.clip(control[ids], q[ids] - 0.1, q[ids] + 0.1)
                control[ids] = np.clip(
                    control[ids], model.jnt_range[ids, 0], model.jnt_range[ids, 1]
                )
        env.step_targets(control)
        s = env.observe()
        if k % 2 == 0:
            samples.append(
                dict(
                    t=s.time_s,
                    pos=s.object_state.pose_world.position,
                    quat=s.object_state.pose_world.quaternion_wxyz,
                    forces={
                        f.value: sum(c.normal_force_N or 0 for c in v.contacts)
                        for f, v in s.fingers.items()
                    },
                    depth=max(
                        (-c.separation_m for v in s.fingers.values() for c in v.contacts), default=0
                    ),
                    selfdepth=s.self_collision_penetration_m,
                    vel=s.object_state.twist_world.linear,
                )
            )
            qs.append(env.native_q(env.sample))
        if k % 50 == 0:
            print(k, samples[-1], "elapsed", time.monotonic() - start, flush=True)
    np.savez_compressed(OUT / f"{args.capture}.npz", qpos=qs)
    (OUT / f"{args.capture}_initialization.json").write_text(
        json.dumps(
            dict(
                qpos=env.native_q().tolist(),
                actuator_targets=env.last_targets.tolist(),
                source="Dynamically settled free-body Newton CPU trajectory; no fixture",
                time_s=env.time_s,
            )
        )
    )
    (OUT / f"{args.capture}.json").write_text(json.dumps(samples))
    print("DONE", time.monotonic() - start, flush=True)


if __name__ == "__main__":
    main()
