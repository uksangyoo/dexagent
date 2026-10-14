"""Translate recorded physics diagnostics into review clips with provenance."""

import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/physics_revision_20261012"
CASES = {
    "gap_repaired_transfer": ("GaP: transfer support and release", "nut", "transfer"),
    "nut_hold_validity": ("Free nut: supported by fingers", "nut", "hold"),
    "screw_hold_validity": ("Free screw: grasp stability", "screw", "hold"),
    "wrench_hold_passive": ("Free wrench: grasp stability", "wrench", "hold"),
    "release_negative": ("Open the hand: the nut falls", "nut", "release"),
    "rotate_x_integral": ("30° nut turn: residual error", "nut", "rotate"),
    "screw_rotate": ("30° screw alignment: incomplete", "screw", "rotate"),
    "wrench_rotate_adaptive": ("Wrench turn: force guard stops", "wrench", "rotate"),
    "nut_translate_out": ("8 mm in-hand translation: incomplete", "nut", "translate"),
}


def main():
    for name, (title, kind, task) in CASES.items():
        path = RUN / f"{name}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        result = json.loads((RUN / f"{name}_result.json").read_text())
        start = Rotation.from_quat(raw[0]["quat"], scalar_first=True)
        frames = []
        for row in raw:
            angle = np.rad2deg(
                (Rotation.from_quat(row["quat"], scalar_first=True) * start.inv()).magnitude()
            )
            frames.append(
                dict(
                    time_s=row["t"],
                    angle_deg=float(angle),
                    speed_rad_s=float(np.linalg.norm(row["omega"])),
                    forces_N=row["forces"],
                    contact_modes=row.get("contact_modes", {}),
                    phase=row["phase"],
                    position_error_mm=1000 * (row.get("position_error_m") or 0),
                    rotation_error_deg=row.get("angle_error_deg"),
                    penetration_mm=1000 * row["depth"],
                    finger_roles=row["roles"],
                    external_force_N=row.get("external_contact_force_N", 0),
                    contacts=row.get("contacts", []),
                )
            )
        passed = result.get("success", False)
        scene = RUN / f"free_{kind}.xml"
        scope = (
            "Three-second free-object hold. At least two loaded fingers at every 2 ms sample; final pose within 2 mm / 3°, low speed, no fixture/table contact. Single diagnostic case, not an industrial success rate."
            if task == "hold"
            else "Controller development diagnostic. Object has six free DOFs, gravity and collision forces. No object actuation or pose edits during execution. Precise industrial manipulation is not yet solved."
        )
        if task == "transfer":
            scope = "Executed by the real GaP graph runtime: confirm the already loaded middle finger, release the thumb, then hold for one second. This is a measured primitive demonstration, not an LLM-generated policy or agent ablation."
        if task == "release":
            scope = "Negative control: deliberate release. The object falls onto the physical workbench; this verifies that the free-body scene has no hidden supporting joint."
        meta = dict(
            id=name,
            title=title,
            tag="FREE BODY · HOLD PASSED"
            if passed
            else "FREE BODY · NEGATIVE CONTROL"
            if task == "release"
            else "FREE BODY · TASK FAILED",
            summary=scope,
            scope=scope,
            fixture=False,
            physical_note="Unactuated free object. All support comes from simulated contacts. The workbench is physical; its contact force is recorded separately.",
            fps=50,
            intrinsic_speed=1,
            sim_seconds_per_video_second=1,
            scene=str(scene),
            scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
            states_sha256=hashlib.sha256((RUN / f"{name}.npz").read_bytes()).hexdigest(),
            physics="Newton collision detection + MuJoCo Warp solver, CPU",
            force_timing="Contact forces and rendered poses share the same pre-integration snapshot (2 ms physics step). White points and red arrows show recorded contacts and force on the object (1 cm/N).",
            reference_deg=result["angle_deg"] if task == "rotate" else None,
            target_deg=result["angle_deg"] if task == "rotate" else None,
            frames=frames,
            result=result,
            captured_at=datetime.datetime.fromtimestamp(
                path.stat().st_mtime, datetime.UTC
            ).isoformat(),
            failure=result.get("violation"),
            study_stage="controller_diagnostic_not_agent_ablation",
        )
        (RUN / "review_input" / f"{name}.json").write_text(json.dumps(meta))
        shutil.copyfile(RUN / f"{name}.npz", RUN / "review_input" / f"{name}.npz")
    print("Prepared", len(CASES), "diagnostic clips")


if __name__ == "__main__":
    main()
