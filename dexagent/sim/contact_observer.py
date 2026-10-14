"""Object-specific contact estimates, slip hysteresis, and conservative rolling labels."""

from dataclasses import dataclass, field

import numpy as np

from dexagent.grammar.finger_roles import ContactMode, Finger
from dexagent.grammar.schema import SurfaceContact

from .object_state import point_velocity, transform_point


@dataclass(frozen=True)
class ObserverThresholds:
    touch_distance_m: float = 0.00015
    force_on_N: float = 0.02
    slip_on_m_s: float = 0.005
    slip_off_m_s: float = 0.002
    rolling_angular_rad_s: float = 0.15
    patch_migration_m_s: float = 0.0005
    confirmation_samples: int = 3
    maximum_gap_s: float = 0.05


@dataclass
class ModeMemory:
    mode: ContactMode = ContactMode.FREE
    pending: ContactMode = ContactMode.FREE
    count: int = 0
    time: float = -1.0
    # Matched by finger link AND shape pair, never by collision-buffer index.
    patches: dict = field(default_factory=dict)
    result: tuple | None = None


class ContactObserver:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or ObserverThresholds()
        self.memory = {f: ModeMemory() for f in Finger}

    def reset(self):
        self.memory = {f: ModeMemory() for f in Finger}

    def classify(self, finger, contacts, time_s):
        t, memory = self.thresholds, self.memory[finger]
        if time_s == memory.time and memory.result is not None:
            mode, confidence, evidence = memory.result
            return mode, confidence, list(evidence)
        if time_s < memory.time or time_s - memory.time > t.maximum_gap_s:
            memory = self.memory[finger] = ModeMemory()
        contacts = [c for c in contacts if self.is_touching(c)]
        if not contacts:
            result = (ContactMode.FREE, 1.0, ("no load-bearing or penetrating object contact",))
            self.memory[finger] = ModeMemory(time=time_s, result=result)
            return result[0], result[1], list(result[2])
        slip = max(c.tangential_speed_m_s for c in contacts)
        limit = t.slip_off_m_s if memory.mode == ContactMode.SLIDING else t.slip_on_m_s
        candidate = ContactMode.SLIDING if slip > limit else ContactMode.STICKING
        evidence = [f"max relative tangential surface speed {slip:.6g} m/s"]
        dt = time_s - memory.time
        patches = {}
        migrating = False
        groups = {}
        for c in contacts:
            groups.setdefault((c.finger_link, c.shape_pair), []).append(c)
        for key, group in groups.items():
            p_f = np.mean([c.point_finger for c in group], axis=0)
            p_o = np.mean([c.point_object for c in group], axis=0)
            patches[key] = (p_f, p_o)
            if 0 < dt <= t.maximum_gap_s and key in memory.patches:
                prev_f, prev_o = memory.patches[key]
                migration = min(np.linalg.norm(p_f - prev_f), np.linalg.norm(p_o - prev_o)) / dt
                omega = max(c.rolling_angular_speed_rad_s for c in group)
                migrating |= migration > t.patch_migration_m_s and omega > t.rolling_angular_rad_s
        if candidate == ContactMode.STICKING and migrating:
            candidate = ContactMode.ROLLING
            evidence.append(
                "low slip, relative tangential rotation, and material contact migration on both bodies"
            )
        # Re-reading the same snapshot does not manufacture temporal evidence.
        if time_s > memory.time:
            memory.count = memory.count + 1 if candidate == memory.pending else 1
            memory.pending = candidate
            memory.patches = patches
            memory.time = time_s
        if candidate == ContactMode.SLIDING or memory.count >= t.confirmation_samples:
            memory.mode = candidate
            confidence = 0.85
        else:
            # Rolling requires a temporal history; a static contact cannot prove it.
            memory.mode = ContactMode.STICKING
            confidence = 0.35
            evidence.append("provisional label; insufficient persistent mode evidence")
        if any(c.normal_force_N is None for c in contacts):
            confidence = min(confidence, 0.3)
            evidence.append("force unavailable; geometric contact only")
        memory.result = (memory.mode, confidence, tuple(evidence))
        return memory.mode, confidence, evidence

    def is_touching(self, contact):
        if contact.normal_force_N is not None:
            return (
                contact.normal_force_N >= self.thresholds.force_on_N
                and contact.separation_m <= self.thresholds.touch_distance_m
            )
        return contact.separation_m <= 0

    def from_arrays(
        self,
        *,
        points0,
        points1,
        normals,
        shapes0,
        shapes1,
        forces,
        shape_body,
        body_transforms,
        body_velocities,
        body_com,
        body_fingers,
        body_names,
        object_body,
        object_name,
        palm,
        sample_time_s,
        force_source,
        robot_bodies=None,
    ):
        """Decode Newton contacts; forces are world forces ON shape0.

        points0/points1 are body-local surface points. Only hand–target
        object pairs are returned; self/fixture contacts cannot count as support.
        """
        result = {f: [] for f in Finger}
        inv_palm = np.linalg.inv(palm)
        inv_object = np.linalg.inv(body_transforms[object_body])
        self_penetration = 0.0
        for i, (s0, s1) in enumerate(zip(shapes0, shapes1, strict=True)):
            if s0 < 0 or s1 < 0:
                continue
            b0, b1 = int(shape_body[s0]), int(shape_body[s1])
            x0 = body_transforms[b0] if b0 >= 0 else np.eye(4)
            x1 = body_transforms[b1] if b1 >= 0 else np.eye(4)
            p0, p1 = transform_point(x0, points0[i]), transform_point(x1, points1[i])
            n = np.asarray(normals[i], dtype=float)
            n /= max(np.linalg.norm(n), 1e-12)
            distance = float(np.dot(p1 - p0, n))
            robot_bodies = set(body_fingers) if robot_bodies is None else robot_bodies
            if b0 != b1 and b0 in robot_bodies and b1 in robot_bodies:
                self_penetration = max(self_penetration, -distance)
            if b0 in body_fingers and b1 == object_body:
                finger_body, finger_point, outward, sign = b0, points0[i], -n, 1
                object_point_world = p1
            elif b1 in body_fingers and b0 == object_body:
                finger_body, finger_point, outward, sign = b1, points1[i], n, -1
                object_point_world = p0
            else:
                continue
            point = 0.5 * (p0 + p1)
            vf = point_velocity(
                body_transforms[finger_body],
                body_com[finger_body],
                body_velocities[finger_body],
                point,
            )
            vo = point_velocity(
                body_transforms[object_body],
                body_com[object_body],
                body_velocities[object_body],
                point,
            )
            velocity = vf - vo
            normal_speed = float(np.dot(velocity, outward))
            omega = body_velocities[finger_body, 3:] - body_velocities[object_body, 3:]
            omega -= np.dot(omega, outward) * outward  # Spin about the normal is not rolling.
            force = sign * np.asarray(forces[i, :3]) if forces is not None else None
            contact = SurfaceContact(
                object=object_name,
                finger_link=body_names[finger_body],
                shape_pair=(int(s0), int(s1)),
                point_world=tuple(point),
                point_palm=tuple(transform_point(inv_palm, point)),
                point_object=tuple(transform_point(inv_object, object_point_world)),
                point_finger=tuple(finger_point),
                normal_world=tuple(outward),
                force_on_finger_world_N=None if force is None else tuple(force),
                normal_force_N=None if force is None else max(0.0, float(np.dot(force, outward))),
                tangential_speed_m_s=float(np.linalg.norm(velocity - normal_speed * outward)),
                normal_speed_m_s=normal_speed,
                rolling_angular_speed_rad_s=float(np.linalg.norm(omega)),
                separation_m=distance,
                sample_time_s=sample_time_s,
                force_source=force_source,
            )
            # Keep geometric penetration even when its solved force is tiny or
            # zero. Classification still filters non-load-bearing candidates;
            # safety must not silently discard an interpenetrating pair.
            if self.is_touching(contact) or contact.separation_m < 0:
                result[body_fingers[finger_body]].append(contact)
        return result, max(0.0, self_penetration)
