"""Velocity decomposition for a hand mounted on a moving arm."""
import numpy as np

def finger_velocity_target(desired_world_velocity, point_jacobian, arm_dofs, commanded_arm_velocity):
    return desired_world_velocity - point_jacobian[:,arm_dofs] @ commanded_arm_velocity

def tracked_contact_velocity(world_translation, relative_correction, correction_limit=.04):
    """Bound corrective motion without clipping the carried object's motion."""
    return world_translation + np.clip(relative_correction,-correction_limit,correction_limit)


def palm_velocity_for_tip(tip_velocity, angular_velocity, tip_position, palm_position):
    """Rigid carried-point velocity mapping; this sets an arm command only."""
    return tip_velocity-np.cross(angular_velocity,tip_position-palm_position)

def palm_velocity_for_slipping_tool(tip_velocity, arm_omega, measured_tool_omega, tip, center, palm):
    """Separate the carried grasp-center lever from the freely pivoting shaft."""
    return tip_velocity-np.cross(measured_tool_omega,tip-center)-np.cross(arm_omega,center-palm)
