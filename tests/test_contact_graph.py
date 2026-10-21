"""Policy guards that must hold before low-level motion is authorized."""
import importlib
import unittest
from types import SimpleNamespace
import numpy as np
from dexagent.sim.contact_graph import specification, advance


def context():
    c=SimpleNamespace(cycle='alternating', phase='acquire', phase_start=0., ready=0.,
        filtered=np.array([.4,.6,.4]), drive_load=.4, return_load=.06,
        amplitude=np.radians(24), speed=np.radians(60), strokes=np.zeros(2),
        driver=0, cycles=0, normal_targets=np.full(2,.4), h=SimpleNamespace(d=SimpleNamespace(time=0.)))
    c.graph=specification(c.cycle,'hom')
    def transition(name):c.phase=name;c.phase_start=c.h.d.time
    c.transition=transition
    return c


class Guards(unittest.TestCase):
    def test_all_graph_references_are_executable(self):
        for cycle in ['alternating','paired']:
            g=specification(cycle,'hom')
            for node in g['nodes'].values():
                module, name=node['ref'].rsplit('.',1)
                self.assertTrue(callable(getattr(importlib.import_module(module),name)))
            for edge in g['conditional_edges'].values():
                self.assertTrue(set(edge['mapping'].values())<=set(g['nodes']))

    def test_no_turn_before_acquisition(self):
        c=context();c.filtered[0]=0
        for _ in range(200):advance(c,.002)
        self.assertEqual(c.phase,'acquire');np.testing.assert_array_equal(c.strokes,[0,0])

    def test_no_return_stroke_without_contact_or_cap_support(self):
        for bad in [np.array([.4,.6,0]),np.array([.4,0,.06]),np.array([.4,.6,.4])]:
            c=context();c.phase='thumb_drive';c.filtered=bad
            c.strokes[:]=[0,c.amplitude];before=c.strokes.copy()
            advance(c,.002);np.testing.assert_array_equal(c.strokes,before)

    def test_loaded_return_and_driver_advance_together(self):
        c=context();c.phase='middle_drive';c.strokes[:]=[c.amplitude,0];c.filtered[:]=[.06,.6,.4]
        advance(c,.002)
        self.assertLess(c.strokes[0],c.amplitude);self.assertGreater(c.strokes[1],0)
        self.assertEqual(c.cycles,0)

    def test_paired_pressure_release_is_gradual_and_reset_stays_guarded(self):
        c=context();c.cycle='paired';c.graph=specification('paired','hom');c.phase='unload';c.stroke=c.amplitude
        load,omega=advance(c,.002)
        self.assertGreater(load,.35);self.assertLess(load,.4)
        self.assertEqual(omega,0);self.assertEqual(c.phase,'unload')
        c.filtered[:]=[.04,.6,.04];c.h.d.time=.3
        _,omega=advance(c,.002)
        self.assertEqual(c.phase,'reset');self.assertLess(omega,0)


if __name__=='__main__':unittest.main()
