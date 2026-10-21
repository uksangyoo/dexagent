"""Offline geometric tests; synthetic poses here are not task rollouts."""
import unittest
import mujoco
import numpy as np
from dexagent.sim.seating import geometry,contact_reached,flush


class Seating(unittest.TestCase):
    def setUp(self):
        boxes=''.join(f'<geom name="workpiece_{i}_{j}" type="box" pos="{.02*(1 if i else -1)} {.02*j} .24" size=".01 .01 .005"/>'for i in [0,1]for j in [-1,1])
        self.m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+boxes+'<body name="screw" pos="0 0 .24794365855856177"><freejoint/><geom name="slot_floor" type="cylinder" pos="0 0 -.0005" size=".0065 .0005" mass=".008"/></body></worldbody></mujoco>')
        self.d=mujoco.MjData(self.m)

    def observe(self):
        mujoco.mj_kinematics(self.m,self.d)
        return geometry(self.m,self.d)

    def test_remaining_travel_comes_from_initial_geometry(self):
        obs=self.observe()
        self.assertAlmostEqual(obs['flat_seat_remaining_mm'],1.94365855856177)
        self.assertFalse(contact_reached(obs,1.))

    def test_flush_geometry_requires_actual_contact_force(self):
        self.d.qpos[2]=.246;obs=self.observe()
        self.assertTrue(flush(obs));self.assertFalse(contact_reached(obs,0.))
        self.assertTrue(contact_reached(obs,.3))

    def test_tilted_edge_contact_is_not_flush_seating(self):
        angle=np.radians(.5);self.d.qpos[3:]=[np.cos(angle/2),np.sin(angle/2),0,0]
        self.d.qpos[2]=.246+.0065*np.sin(angle)
        obs=self.observe()
        self.assertGreater(obs['head_gap_max_m'],100e-6)
        self.assertFalse(flush(obs));self.assertFalse(contact_reached(obs,.3))

    def test_deep_penetration_is_not_accepted_as_seating(self):
        self.d.qpos[2]=.24598;obs=self.observe()
        self.assertFalse(flush(obs));self.assertFalse(contact_reached(obs,.3))


if __name__=='__main__':unittest.main()
