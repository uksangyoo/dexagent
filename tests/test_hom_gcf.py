"""Unit tests for minimal HOM GCF math (no simulator)."""
import numpy as np

from dexagent.hom.gcf import closest_point_aabb, gcf_from_tip_and_box, relative_contact_twist
from dexagent.hom.features import HomFeatureLogger, recompute_hom_arrays


def test_closest_point_outside():
    p = closest_point_aabb([1, 0, 0], [0, 0, 0], [0.02, 0.02, 0.02])
    assert np.allclose(p, [0.02, 0, 0])


def test_gcf_separation_along_x():
    g = gcf_from_tip_and_box([0.05, 0, 0], [0, 0, 0], [0.02, 0.02, 0.02])
    assert abs(g["dist"] - 0.03) < 1e-9
    assert abs(g["normal"][0] - 1.0) < 1e-6


def test_relative_twist_sep():
    g0 = gcf_from_tip_and_box([0.05, 0, 0], [0, 0, 0], [0.02] * 3)
    g1 = gcf_from_tip_and_box([0.06, 0, 0], [0, 0, 0], [0.02] * 3)
    v = relative_contact_twist(g0, g1, 0.1)
    assert abs(v["v_sep"] - 0.1) < 1e-6


def test_logger_shapes():
    log = HomFeatureLogger(box_half=(0.02,) * 3)
    tips = np.zeros((5, 3))
    tips[0] = [0.05, 0, 0]
    tips[1] = [-0.05, 0, 0]
    row = log.update(tips, [0, 0, 0], 0.02)
    assert row["hom_sep"].shape == (5,)
    assert row["hom_slide"].shape == (5, 2)


def test_recompute():
    n = 10
    t = np.linspace(0, 1, n)
    tips = np.zeros((n, 5, 3))
    tips[:, 0, 0] = 0.05 + 0.01 * t
    obj = np.zeros((n, 7))
    obj[:, 6] = 1.0
    out = recompute_hom_arrays(tips, obj, t)
    assert out["hom_sep"].shape == (n, 5)
