"""Intended finger functions, independent of measured physical contact modes."""

from enum import StrEnum


class Finger(StrEnum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


class FingerRole(StrEnum):
    SUPPORT = "SUPPORT"
    DRIVE = "DRIVE"
    REPOSITION = "REPOSITION"
    RELEASE = "RELEASE"
    FREE = "FREE"


class ContactMode(StrEnum):
    FREE = "free"
    STICKING = "sticking"
    SLIDING = "sliding"
    ROLLING = "rolling"


# Explicit asset adapter: standard 20-DOF right DG5F-S, not the 15-DOF asset.
FINGER_NUMBERS = {finger: i for i, finger in enumerate(Finger, 1)}
JOINT_NAMES = {f: tuple(f"joint_{i}_{j}" for j in range(1, 5)) for f, i in FINGER_NUMBERS.items()}
LINK_NAMES = {f: tuple(f"link_{i}_{j}" for j in range(1, 5)) for f, i in FINGER_NUMBERS.items()}
