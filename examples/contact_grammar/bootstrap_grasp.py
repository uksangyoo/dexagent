"""Measured fixture contact bootstrap, followed by a free-body nut hold.

The fixture is used only to construct a candidate initialization. The hold test
uses a separate scene with a free joint and no object support constraint.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexagent.controllers.contact_controller import ContactController
from dexagent.grammar.finger_roles import Finger, FingerRole
from dexagent.grammar.schema import Constraints, Termination, TransitionRequest
from dexagent.sim.object_state import pose_error
from dexagent.sim.ur5e_tesollo_env import ROOT, UR5eTesolloEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-force", type=float, default=0.3)
    parser.add_argument("--lock-fixture", action="store_true")
    parser.add_argument("--contact-stiffness", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", type=Path, default=ROOT / "runs/assets/scene.xml")
    parser.add_argument("--force-gain", type=float)
    parser.add_argument(
        "--fingers",
        nargs="+",
        default=["index", "thumb", "middle"],
        choices=[f.value for f in Finger],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_scene = args.scene
    if args.lock_fixture:
        setup_tree = ET.parse(setup_scene)
        joint = setup_tree.getroot().find(".//joint[@name='nut_rotation']")
        joint.set("limited", "true")
        joint.set("range", "-0.000001 0.000001")
        setup_scene = args.output / "fixture_scene.xml"
        setup_tree.write(setup_scene)
    env = UR5eTesolloEnv(
        setup_scene, device=args.device, contact_stiffness_N_m=args.contact_stiffness
    )
    (args.output / "contact_profile.json").write_text(
        json.dumps(env.contact_profile, indent=2) + "\n"
    )
    controller = ContactController(env)
    if args.force_gain is not None:
        controller.force_gain = args.force_gain
    constraints = Constraints(
        max_normal_force_N=5.0,
        min_support_contacts=0,
        min_support_force_N=args.minimum_force,
        calibration_id="provisional_fixture_bootstrap",
    )
    records = []
    for name in args.fingers:
        finger = Finger(name)
        env.roles[finger] = FingerRole.SUPPORT
        point = (-0.022, 0.015, 0.010) if name == "index" and not records else None
        point, _, _ = env.contact_target(finger, point)
        result = controller.execute(
            TransitionRequest(
                operation="establish_contact",
                object="hex_nut",
                finger=finger,
                object_point=tuple(point),
                constraints=constraints,
                termination=Termination(timeout_s=4),
            )
        )
        records.append(result.model_dump(mode="json"))
        (args.output / "fixture_reports.json").write_text(json.dumps(records, indent=2) + "\n")
        (args.output / f"{name}_trace.json").write_text(
            json.dumps(controller.last_trace, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "finger": name,
                    "success": result.success,
                    "reason": result.message,
                    "forces": {
                        f.value: sum(c.normal_force_N or 0.0 for c in s.contacts)
                        for f, s in env.observe().fingers.items()
                    },
                }
            ),
            flush=True,
        )
    # Freeze a settled, synchronized physical snapshot for initialization.
    for _ in range(25):
        env.step_targets(env.last_targets, constraints=constraints)
        if env.last_constraint_violation:
            break
    state = env.observe()
    if sum(bool(s.contacts) for s in state.fingers.values()) < 2:
        (args.output / "result.json").write_text(
            json.dumps(
                {"ready": False, "reason": "Fewer than two physical contacts after bootstrap"}
            )
            + "\n"
        )
        return
    # Retain actual hand posture, object pose, and loaded actuator targets.
    q = env.native_q(env.sample)
    targets = env.last_targets.copy()
    obj = state.object_state.pose_world
    tree = ET.parse(args.scene)
    root = tree.getroot()
    nut = root.find(".//body[@name='nut']")
    for joint in list(nut.findall("joint")):
        nut.remove(joint)
    ET.SubElement(nut, "freejoint", name="object_free")
    for key in list(root.findall("keyframe")):
        root.remove(key)
    freeq = np.r_[q[:26], obj.position, obj.quaternion_wxyz]
    key = ET.SubElement(root, "keyframe")
    ET.SubElement(
        key,
        "key",
        name="home",
        qpos=" ".join(map(str, freeq)),
        ctrl=" ".join(map(str, targets[:26])),
    )
    scene = args.output / "free_scene.xml"
    tree.write(scene)
    free = UR5eTesolloEnv(scene, device=args.device, contact_stiffness_N_m=args.contact_stiffness)
    freetargets = freeq.copy()
    freetargets[:26] = targets[:26]
    free.reset(freeq, actuator_targets=freetargets)
    initial = free.observe()
    samples = []

    def record(s):
        p, r = pose_error(s.object_state.pose_world, initial.object_state.pose_world)
        samples.append(
            {
                "time_s": s.time_s,
                "position_error_m": p,
                "rotation_error_deg": r,
                "forces": {
                    f.value: sum(c.normal_force_N or 0 for c in v.contacts)
                    for f, v in s.fingers.items()
                },
                "max_object_penetration_m": max(
                    (-c.separation_m for v in s.fingers.values() for c in v.contacts), default=0.0
                ),
                "object_pose_palm": s.object_state.pose_palm.model_dump(mode="json"),
                "linear_speed_m_s": float(np.linalg.norm(s.object_state.twist_world.linear)),
            }
        )

    with free.observe_each_step(record):
        for _ in range(100):
            free.step_targets(freetargets, constraints=constraints)
            if free.last_constraint_violation:
                break
    final = free.observe()
    stable = (
        len(samples) >= 990
        and all(s["position_error_m"] <= 0.01 and s["rotation_error_deg"] <= 5 for s in samples)
        and min(sum(f >= 0.05 for f in s["forces"].values()) for s in samples) >= 2
        and samples[-1]["linear_speed_m_s"] < 0.02
        and np.linalg.norm(final.object_state.twist_world.angular) < 0.2
    )
    out = {
        "ready": stable,
        "contact_profile": free.contact_profile,
        "scope": "2-second unsupported hold; no rotation benchmark",
        "externally_supported": final.object_state.externally_supported,
        "free_body": final.object_state.free_body,
        "max_position_drift_m": max(s["position_error_m"] for s in samples),
        "constraint_violation": free.last_constraint_violation,
        "final": final.model_dump(mode="json"),
        "initial_q": freeq.tolist(),
        "actuator_targets": freetargets.tolist(),
    }
    (args.output / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.output / "hold_trace.json").write_text(json.dumps(samples) + "\n")
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ["final", "initial_q", "actuator_targets"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
