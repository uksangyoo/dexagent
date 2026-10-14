"""Validated surface proposals for unions of authored box/sphere object geoms."""

import mujoco
import numpy as np


def primitive_distance(model, data, geom, point):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local = rotation.T @ (np.asarray(point) - data.geom_xpos[geom])
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        delta = np.abs(local) - model.geom_size[geom]
        return float(np.linalg.norm(np.maximum(delta, 0)) + min(np.max(delta), 0))
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(np.linalg.norm(local) - model.geom_size[geom, 0])
    raise ValueError("Surface projection supports box/sphere task geoms only")


def project_object_surface(
    model, data, body, point, *, approach_from=None, feature_tolerance_m=0.0001
):
    """Project onto exterior features and choose an accessible edge normal.

    Within 0.1 mm of an edge, equivalent face candidates are disambiguated by
    the approach point. Otherwise tiny numerical changes can switch the normal
    by 90 degrees. The returned point is on the selected face, not the original
    potentially off-surface proposal. This is geometry, not a reachability test.
    """
    geoms = np.flatnonzero(
        (model.geom_bodyid == body) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    candidates = []
    point = np.asarray(point)
    for geom in geoms:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        origin = data.geom_xpos[geom]
        local = rotation.T @ (point - origin)
        size = model.geom_size[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            for axis in range(3):
                for sign in (-1, 1):
                    surface = np.clip(local, -size, size)
                    surface[axis] = sign * size[axis]
                    normal = np.zeros(3)
                    normal[axis] = sign
                    candidates.append((origin + rotation @ surface, rotation @ normal))
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
            normal = local / max(np.linalg.norm(local), 1e-12)
            if np.linalg.norm(normal) < 0.5:
                normal = np.array([1.0, 0.0, 0.0])
            candidates.append((origin + rotation @ (normal * size[0]), rotation @ normal))
        else:
            raise ValueError("Surface projection supports box/sphere task geoms only")
    # Exclude internal faces buried inside another component of the object.
    candidates = [
        (p, n)
        for p, n in candidates
        if min(primitive_distance(model, data, g, p) for g in geoms) >= -1e-7
    ]
    if not candidates:
        raise ValueError("No exterior surface candidate")
    closest = min(candidates, key=lambda item: np.linalg.norm(item[0] - point))
    if approach_from is None:
        return closest
    distance = np.linalg.norm(closest[0] - point)
    candidates = [
        (p, n) for p, n in candidates if np.linalg.norm(p - point) <= distance + feature_tolerance_m
    ]

    def alignment(item):
        p, n = item
        direction = np.asarray(approach_from) - p
        return float(np.dot(n, direction) / max(np.linalg.norm(direction), 1e-12))

    return max(candidates, key=alignment)
