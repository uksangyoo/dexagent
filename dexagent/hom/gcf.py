"""Generalized contact frames from fingertip pads vs an object AABB.

Minimal stand-in for the paper's mesh SDF / hydroelastic COP queries: the
finger witness is the tip site; the object witness is the closest point on the
object's axis-aligned bounding box (world frame). Normal = unit vector from
object witness to tip (separation direction).
"""
from __future__ import annotations

import numpy as np


def closest_point_aabb(point, center, half) -> np.ndarray:
    """Closest point on an AABB ``[center ± half]`` to ``point``."""
    lo = np.asarray(center, float) - np.asarray(half, float)
    hi = np.asarray(center, float) + np.asarray(half, float)
    return np.minimum(np.maximum(np.asarray(point, float), lo), hi)


def gcf_from_tip_and_box(tip_xyz, box_center, box_half) -> dict:
    """Return GCF origins, normal, signed distance for one finger vs a box."""
    tip = np.asarray(tip_xyz, float)
    p_obj = closest_point_aabb(tip, box_center, box_half)
    delta = tip - p_obj
    dist = float(np.linalg.norm(delta))
    if dist < 1e-8:
        # Degenerate: tip inside / on surface — use vector from box center.
        delta = tip - np.asarray(box_center, float)
        dist = float(np.linalg.norm(delta))
        if dist < 1e-8:
            normal = np.array([0.0, 0.0, 1.0])
            dist = 0.0
        else:
            normal = delta / dist
            dist = 0.0  # penetrating / coincident
    else:
        normal = delta / dist
    # Tangent basis: pick a vector not parallel to normal.
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(normal, helper)
    y /= max(float(np.linalg.norm(y)), 1e-8)
    z = np.cross(normal, y)
    R = np.column_stack([normal, y, z])  # columns = x(normal), y, z
    return {
        "p_finger": tip.copy(),
        "p_object": p_obj.copy(),
        "normal": normal.copy(),
        "R": R,
        "dist": dist,
    }


def relative_contact_twist(gcf_prev: dict | None, gcf: dict, dt: float) -> dict:
    """Finite-difference relative contact velocities in the current GCF.

    Returns ``v_sep`` (scalar), ``v_slide`` (2,), ``v_roll`` (2,), ``v_spin`` (scalar).
    """
    zeros = {
        "v_sep": 0.0,
        "v_slide": np.zeros(2),
        "v_roll": np.zeros(2),
        "v_spin": 0.0,
    }
    if gcf_prev is None or dt <= 1e-6:
        return zeros
    # Linear: finger origin motion relative to object origin, in GCF axes.
    dp = (gcf["p_finger"] - gcf_prev["p_finger"]) - (gcf["p_object"] - gcf_prev["p_object"])
    v_lin = dp / dt
    R = gcf["R"]
    v_local = R.T @ v_lin  # x=sep, yz=slide
    # Angular: change of normal expressed in GCF (roll about y/z, spin about x).
    n0 = gcf_prev["normal"]
    n1 = gcf["normal"]
    # Small-angle ω ≈ n0 × n1 / dt projected into GCF.
    w_world = np.cross(n0, n1) / dt
    w_local = R.T @ w_world
    return {
        "v_sep": float(v_local[0]),
        "v_slide": np.asarray(v_local[1:], float).copy(),
        "v_roll": np.asarray(w_local[1:], float).copy(),
        "v_spin": float(w_local[0]),
    }
