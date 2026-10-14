"""Rigid transforms and twists with explicit reference points and frames."""

import numpy as np
from scipy.spatial.transform import Rotation

from dexagent.grammar.schema import ObjectState, Pose, Twist


def matrix(pose: Pose):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(pose.quaternion_wxyz, scalar_first=True).as_matrix()
    out[:3, 3] = pose.position
    return out


def pose(transform):
    return Pose(
        position=tuple(transform[:3, 3]),
        quaternion_wxyz=tuple(Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)),
    )


def transform_point(transform, point):
    return transform[:3, :3] @ np.asarray(point) + transform[:3, 3]


def origin_velocity(transform, com_local, spatial_velocity):
    """Newton body_qd is [v_COM, omega], both in world coordinates."""
    r = transform[:3, :3] @ np.asarray(com_local)
    return np.asarray(spatial_velocity[:3]) - np.cross(spatial_velocity[3:], r)


def point_velocity(transform, com_local, spatial_velocity, point_world):
    com = transform_point(transform, com_local)
    return np.asarray(spatial_velocity[:3]) + np.cross(
        spatial_velocity[3:], np.asarray(point_world) - com
    )


def relative_object_state(
    name, palm, obj, palm_twist, object_twist, *, externally_supported=False, free_body=False
):
    # Supplied linear velocities are at frame origins, not at centers of mass.
    r = obj[:3, 3] - palm[:3, 3]
    v = np.asarray(object_twist.linear) - palm_twist.linear - np.cross(palm_twist.angular, r)
    w = np.asarray(object_twist.angular) - palm_twist.angular
    return ObjectState(
        name=name,
        pose_world=pose(obj),
        pose_palm=pose(np.linalg.inv(palm) @ obj),
        twist_world=object_twist,
        twist_palm=Twist(linear=tuple(palm[:3, :3].T @ v), angular=tuple(palm[:3, :3].T @ w)),
        externally_supported=externally_supported,
        free_body=free_body,
    )


def pose_error(actual: Pose, target: Pose):
    a, b = matrix(actual), matrix(target)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(
        np.rad2deg(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude())
    )
