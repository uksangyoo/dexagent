from dexagent.grammar.finger_roles import ContactMode, Finger, FingerRole
from dexagent.grammar.schema import FingerState, HandState, ObjectState, Pose, SurfaceContact, Twist


def contact(
    *,
    force=1.0,
    slip=0.0,
    point=(0.01, 0, 0),
    time=0.0,
    omega=0.0,
    pair=(1, 2),
    normal=(1.0, 0.0, 0.0),
):
    return SurfaceContact(
        object="hex_nut",
        finger_link="link_1_4",
        shape_pair=pair,
        point_world=point,
        point_palm=point,
        point_object=point,
        point_finger=point,
        normal_world=normal,
        force_on_finger_world_N=None if force is None else (force, 0.0, 0.0),
        normal_force_N=force,
        tangential_speed_m_s=slip,
        normal_speed_m_s=0.0,
        rolling_angular_speed_rad_s=omega,
        separation_m=-0.00001,
        sample_time_s=time,
        force_source="unavailable" if force is None else "mujoco_warp_constraint",
    )


def state(contacts=None, roles=None, time=0.0):
    contacts = contacts or {}
    return HandState(
        time_s=time,
        contact_time_s=time,
        palm_pose_world=Pose(position=(0, 0, 0)),
        fingers={
            f: FingerState(
                joint_positions=(0, 0, 0, 0),
                joint_velocities=(0, 0, 0, 0),
                fingertip_pose_palm=Pose(position=(0.01, 0, 0)),
                contacts=contacts.get(f, []),
                contact_state=ContactMode.STICKING if contacts.get(f) else ContactMode.FREE,
                classification_confidence=1.0,
            )
            for f in Finger
        },
        finger_roles=roles or dict.fromkeys(Finger, FingerRole.FREE),
        object_state=ObjectState(
            name="hex_nut",
            pose_world=Pose(position=(0, 0, 0.1)),
            pose_palm=Pose(position=(0, 0, 0.1)),
            twist_palm=Twist(),
            twist_world=Twist(),
            free_body=True,
        ),
        force_estimates_available=True,
    )


def action_dict():
    return dict(
        mode="rotate",
        object="hex_nut",
        target={"axis": [0, 0, 1], "angle_deg": 30, "frame": "object"},
        finger_roles={
            "thumb": "SUPPORT",
            "index": "DRIVE",
            "middle": "SUPPORT",
            "ring": "REPOSITION",
            "pinky": "FREE",
        },
        constraints={
            "min_support_contacts": 2,
            "max_normal_force_N": 5.0,
            "prevent_object_drop": True,
        },
    )
