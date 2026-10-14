"""Static gravity/disturbance support certificate with a friction-pyramid LP.

Necessary mechanical check under the stated point-contact/friction assumptions;
not a proof of dynamic stability, reachability, or hardware force capacity.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class WrenchCertificate:
    feasible: bool
    required_normal_force_N: float | None
    residual: float | None
    reason: str


def certify_support(
    contacts,
    *,
    center_of_mass_world,
    mass_kg,
    friction,
    max_force_per_finger_N,
    gravity=(0, 0, -9.81),
    disturbance_force=(0, 0, 0),
    disturbance_torque=(0, 0, 0),
):
    if mass_kg <= 0 or friction < 0 or max_force_per_finger_N <= 0:
        raise ValueError("Invalid mass, friction, or force bound")
    columns = []
    owners = []
    fingers = list(contacts)
    for index, finger in enumerate(fingers):
        for contact in contacts[finger]:
            if (contact.normal_force_N or 0) < 0.02 or contact.separation_m > 0.00015:
                continue
            n = -np.asarray(contact.normal_world, dtype=float)
            n /= np.linalg.norm(n)
            tangent = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(n, tangent)
            lever = np.asarray(contact.point_world) - center_of_mass_world
            for angle in np.arange(8) * np.pi / 4:
                direction = n + friction * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
                columns.append(np.r_[direction, np.cross(lever, direction) / 0.03])
                owners.append(index)
    if not columns:
        return WrenchCertificate(False, None, None, "No load-bearing object contacts")
    A = np.array(columns).T
    b = -np.r_[
        mass_kg * np.asarray(gravity) + disturbance_force, np.asarray(disturbance_torque) / 0.03
    ]
    cap = np.array([[int(owner == i) for owner in owners] for i in range(len(fingers))])
    fit = linprog(
        np.ones(len(columns)),
        A_ub=cap,
        b_ub=np.full(len(fingers), max_force_per_finger_N),
        A_eq=A,
        b_eq=b,
        bounds=(0, None),
        method="highs",
    )
    return WrenchCertificate(
        bool(fit.success),
        float(fit.fun) if fit.success else None,
        float(np.linalg.norm(A @ fit.x - b)) if fit.success else None,
        "ok" if fit.success else "Cannot balance the requested wrench within friction/force limits",
    )
