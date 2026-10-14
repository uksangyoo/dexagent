"""Execute a real GaP support-transfer graph on a free Newton object."""

import argparse
import json

import gap
import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.gap.connector import DexterousConnector
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import Constraints, Termination, TransitionRequest
from dexagent.sim.object_state import matrix
from dexagent.sim.ur5e_tesollo_env import UR5eTesolloEnv
from dexagent.skills.regrasp import regrasp
from scripts.physics.probe_free import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait", action="store_true")
    parser.add_argument("--object", choices=["nut", "screw"], default="nut")
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--case")
    args = parser.parse_args()
    name = f"gap_{args.object}_gait" if args.gait else "gap_repaired_transfer"
    name = args.case or name
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    from dexagent.tasks.industrial_scenes import build_free_scene
    scene = build_free_scene(OUT / f"{name}_scene.xml", kind=args.object, friction=args.friction)
    env = UR5eTesolloEnv(scene, device="cpu", substeps=5,
                        object_name="hex_nut" if args.object == "nut" else "hex_screw")
    init = json.loads(
        (
            OUT
            / (
                "ready_grasp_initialization.json"
                if args.object == "nut"
                else "screw_grasp_initialization.json"
            )
        ).read_text()
    )
    env.reset(np.array(init["qpos"]), actuator_targets=np.array(init["actuator_targets"]))
    env.roles = {
        f: FingerRole.SUPPORT
        if f in (Finger.THUMB, Finger.INDEX, Finger.MIDDLE)
        else FingerRole.FREE
        for f in Finger
    }
    conn = DexterousConnector(env)
    conn.hand.contacts.force_gain = 0.00001
    s = env.observe()
    initial = matrix(s.object_state.pose_world)
    c = max(s.fingers[Finger.MIDDLE].contacts, key=lambda c: c.normal_force_N or 0)
    request = TransitionRequest(
        operation="transfer_support",
        object=env.object_name,
        finger=Finger.THUMB,
        to_finger=Finger.MIDDLE,
        object_point=c.point_object,
        constraints=Constraints(max_normal_force_N=5, min_support_contacts=2),
        termination=Termination(timeout_s=4),
    )
    if args.gait:
        from dexagent.skills.composition import HandOperation, workflow

        thumb = max(s.fingers[Finger.THUMB].contacts, key=lambda c: c.normal_force_N or 0)
        recontact = TransitionRequest(
            operation="establish_contact",
            object=env.object_name,
            finger=Finger.THUMB,
            object_point=thumb.point_object,
            constraints=request.constraints,
            termination=Termination(timeout_s=4),
        )
        operations = [
            HandOperation("hand.transition_contact", {"request": r.model_dump(mode="json")})
            for r in [request, recontact]
        ]
        operations.append(
            HandOperation(
                "hand.assign_roles", {"roles": {f.value: r.value for f, r in env.roles.items()}}
            )
        )
        workflow("finger_gait", operations).save(folder / "workflow.json")
    else:
        regrasp([request]).save(folder / "workflow.json")
    certificate = conn.hand.check_support_wrench(
        dict(friction=min(1.,args.friction), max_force_per_finger_N=5, exclude_fingers=["thumb"])
    )
    (folder / "support_wrench.json").write_text(json.dumps(certificate, indent=2))
    samples = []
    qs = []

    def record(s):
        if samples and s.time_s - samples[-1]["t"] < 0.0199:
            return
        obj = matrix(s.object_state.pose_world)
        samples.append(
            dict(
                t=s.time_s,
                phase="recontact"
                if args.gait and env.roles[Finger.THUMB] == FingerRole.FREE
                else "support_transfer",
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
                position_error_m=float(np.linalg.norm(obj[:3, 3] - initial[:3, 3])),
                angle_error_deg=float(
                    np.rad2deg(Rotation.from_matrix(obj[:3, :3] @ initial[:3, :3].T).magnitude())
                ),
            )
        )
        qs.append(env.native_q(env.sample))

    with env.observe_each_step(record):
        run = gap.execute(
            folder, connector=conn, skills=[], trace_dir=folder / "trace", max_node_workers=1
        )
        if run.success:
            for _ in range(100):
                env.step_targets(
                    env.last_targets, constraints=request.constraints, require_support=True
                )
                if env.last_constraint_violation:
                    break
    np.savez_compressed(OUT / f"{name}.npz", qpos=qs)
    (OUT / f"{name}.json").write_text(json.dumps(samples))
    result = dict(
        success=run.success and not env.last_constraint_violation,
        graph_success=run.success,
        task="gait" if args.gait else "transfer",
        object=args.object,
        physics="Newton / MuJoCo Warp CPU",
        free_body=True,
        object_actuated=False,
        angle_deg=0,
        violation=env.last_constraint_violation,
        final=samples[-1],
        friction=args.friction,
        observed_contact_friction=env.solver.mjw_data.contact.friction.numpy()[:int(env.solver.mjw_data.nacon.numpy()[0]),0].tolist(),
        reports=[r.model_dump(mode="json") for r in conn.hand.history],
        certificate=certificate,
    )
    (OUT / f"{name}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
