"""Paired physical cases and explicit benchmark readiness contracts."""

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np


class TaskKind(StrEnum):
    NUT = "nut_reorientation"
    WRENCH = "wrench_fastening"
    CABLE = "cable_wrapping"
    SCREW = "screw_handling"


REQUIRED_CAPABILITIES = {
    TaskKind.NUT: {"free_rigid_object", "solved_contact_forces"},
    TaskKind.WRENCH: {
        "free_rigid_object",
        "wrench_nut_contact",
        "thread_advance",
        "solved_contact_forces",
    },
    TaskKind.CABLE: {"deformable_cable", "material_coordinate_tracking", "calibrated_tension"},
    TaskKind.SCREW: {
        "free_rigid_object",
        "thread_advance",
        "screw_fixture_alignment",
        "solved_contact_forces",
    },
}


@dataclass(frozen=True)
class PhysicalCase:
    case_id: str
    task: str
    split: str
    seed: int
    angle_deg: float
    friction: float
    dimension_scale: float
    translation_m: tuple
    orientation_rotvec_rad: tuple
    grasp_joint_noise_rad: tuple

    def to_dict(self):
        return asdict(self)


def sample_case(task, split, index, seed=0):
    if split not in {"train", "test_id", "test_ood"}:
        raise ValueError(split)
    key = f"{task}:{split}:{index}:{seed}"
    case_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(case_seed)
    ood = split == "test_ood"
    # OOD geometry and friction are disjoint from training support.
    scale = rng.uniform(1.15, 1.3) if ood else rng.uniform(0.9, 1.1)
    friction = rng.uniform(0.25, 0.45) if ood else rng.uniform(0.6, 1.2)
    return PhysicalCase(
        hashlib.sha256(key.encode()).hexdigest()[:16],
        str(task),
        split,
        case_seed,
        [30.0, 60.0, 90.0][index % 3],
        float(friction),
        float(scale),
        tuple(rng.uniform(-0.001, 0.001, 3)),
        tuple(rng.uniform(-0.05, 0.05, 3)),
        tuple(rng.normal(0, 0.002, 20)),
    )


def require_capabilities(task, available):
    missing = REQUIRED_CAPABILITIES[TaskKind(task)] - set(available)
    if missing:
        raise ValueError(f"Benchmark {task} is not runnable: missing {sorted(missing)}")


@dataclass(frozen=True)
class TaskVerdict:
    success: bool
    terminal: bool
    metrics: dict
    reason: str
