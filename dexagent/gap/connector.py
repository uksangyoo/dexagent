"""GaP's connector protocol, tool registry, and privileged World snapshots."""

import mujoco
import numpy as np
from gap.runtime.verify.world import Body, World
from gap_core.tools import ToolRegistry

from dexagent.sim.object_state import matrix

from .tools import HandTools


class DexterousConnector:
    def __init__(self, env):
        self.env = env
        self.tool_registry = ToolRegistry()
        self.hand = HandTools(env)
        self.hand.register(self.tool_registry)

    def get_observation(self):
        return {
            "cameras": [],
            "arms": [],
            "dexterous_hand": self.env.observe().model_dump(mode="json"),
        }

    def world_snapshot(self):
        with self.env.transaction():
            state = self.env.observe()
            obj = state.object_state
            T = matrix(obj.pose_world)
            # Exact conservative AABB from MuJoCo's local geometry AABBs,
            # transformed by current object pose; no fabricated cube geometry.
            m = self.env.kin_model
            d = mujoco.MjData(m)
            d.qpos[:] = self.env.native_q(self.env.sample)
            mujoco.mj_forward(m, d)
            body_id = m.body(self.env.object_body_name).id
            corners = []
            for g in np.flatnonzero(m.geom_bodyid == body_id):
                center = m.geom_aabb[g, :3]
                half = m.geom_aabb[g, 3:]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            corners.append(
                                d.geom_xpos[g]
                                + d.geom_xmat[g].reshape(3, 3) @ (center + half * [sx, sy, sz])
                            )
            contacts = frozenset(
                c.finger_link for finger in state.fingers.values() for c in finger.contacts
            )
            body = Body(
                name=obj.name,
                position=T[:3, 3],
                quaternion_wxyz=np.asarray(obj.pose_world.quaternion_wxyz),
                aabb_lower=np.min(corners, axis=0),
                aabb_upper=np.max(corners, axis=0),
                linear_velocity=np.asarray(obj.twist_world.linear),
                angular_velocity=np.asarray(obj.twist_world.angular),
                contacts=contacts,
            )
            return World(
                env_id=0,
                bodies={obj.name: body},
                time_s=state.time_s,
                robot_link_prefixes=("link_",),
                raw_contact_diagnostics={
                    f.value: [c.model_dump(mode="json") for c in s.contacts]
                    for f, s in state.fingers.items()
                },
                sim_state={"dexterous_hand": state.model_dump(mode="json")},
                scene_spec_ids=(obj.name,),
            )

    def close(self):
        self.env.close()
