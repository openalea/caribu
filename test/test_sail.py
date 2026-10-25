"""The turbid medium (sail): SAIL's physics on simple canopies, s2v's structure on simple triangles."""
import math

import numpy as np
import pytest

from openalea.caribu import sail


def uniform(layers=4, lai=0.5, angle_class=None, rho=10., tau=5.):
    """A canopy of *layers* equal layers (from the soil up), leaves of one zenith class (spherical-like when None)."""
    frequencies = np.full((layers, 18), 1 / 18.) if angle_class is None else np.eye(18)[[angle_class] * layers]
    return frequencies, np.full(layers, lai), np.full(layers, rho), np.full(layers, tau)


@pytest.mark.parametrize("angle_class, zenith", [(None, 0.), (0, 30.), (17, 60.), (9, 75.)])
def test_energy_balance(angle_class, zenith):
    """A direct beam of unit energy: absorbed by the layers, reflected at the top, absorbed by the soil."""
    soil = 0.2
    frequencies, lai, rho, tau = uniform(angle_class=angle_class)
    direction = (math.sin(math.radians(zenith)), 0., -math.cos(math.radians(zenith)))
    profile = sail.profiles(frequencies, lai, rho, tau, soil, 0.25, [(1., direction)])
    soil_absorbed = (profile["direct"][0] + profile["downward"][0]) * (1 - soil)
    total = profile["absorbed"].sum() + profile["upward"][-1] + soil_absorbed
    assert total == pytest.approx(1., abs=1e-12)
    assert profile["downward"][-1] == 0.  # no diffuse sky


def test_horizontal_leaves_extinction():
    """Nearly horizontal leaves (the class's mid angle 2.5°): the uncollided beam at the soil is exp(−l cos 2.5°)
    whatever the sun's zenith (the path's length and the leaves' projection cancel; ks = l for flat leaves)."""
    frequencies, lai, rho, tau = uniform(layers=1, lai=0.8, angle_class=0)
    for zenith in (0., 40., 80.):
        direction = (math.sin(math.radians(zenith)), 0., -math.cos(math.radians(zenith)))
        profile = sail.profiles(frequencies, lai, rho, tau, 0., 1., [(1., direction)])
        if zenith + 2.5 <= 90:
            assert profile["direct"][0] == pytest.approx(math.exp(-0.8 * math.cos(math.radians(2.5))), rel=1e-12)


def test_sources_add_by_energy():
    frequencies, lai, rho, tau = uniform()
    a, b = (0., 0., -1.), (0.5, 0., -0.8)
    both = sail.profiles(frequencies, lai, rho, tau, 0.1, 0.25, [(1., a), (2., b)])
    one = sail.profiles(frequencies, lai, rho, tau, 0.1, 0.25, [(1., a)])
    two = sail.profiles(frequencies, lai, rho, tau, 0.1, 0.25, [(1., b)])
    for name in ("downward", "upward", "direct"):
        np.testing.assert_allclose(both[name], one[name] + 2 * two[name], rtol=1e-12)


def test_beta_law_sums_nearly_to_100():
    """distrib renormalises to first order only (f += (1 − Σf) f: a total of 100 s (2 − s), s the raw sum)."""
    for mu, nu in ((2.77, 1.172), (1.101, 1.93), (1.5, 1.5)):
        assert sum(sail.beta_frequencies(mu, nu)) == pytest.approx(100., abs=0.05)


def test_view_radiance_and_diffuse_sky():
    frequencies, lai, rho, tau = uniform()
    profile = sail.profiles(frequencies, lai, rho, tau, 0.1, 0.25, [(1., (0., 0., -1.))], view_zenith=30.,
                            view_azimuth=90.)
    assert np.all(profile["view"] >= 0)
    sky = sail.profiles(frequencies, lai, rho, tau, 0.1, 0.25, [], diffuse=(60., 1.))
    assert sky["downward"][-1] == pytest.approx(1.)  # the whole sky diffuse at the top
    assert sky["direct"][-1] == 0.


def test_layers_of_a_horizontal_and_a_vertical_triangle():
    """s2v: a horizontal leaf in the top layer (class 0), a vertical stem across both (class 17, half its area)."""
    leaf = [[0., 0., 1.5], [1., 0., 1.5], [0., 1., 1.5]]
    stem = [[0.2, 0.2, 0.1], [0.4, 0.2, 0.1], [0.2, 0.2, 1.9]]
    labels = [10 ** 11 + 10 ** 6 + 1000, 10 ** 11 + 10 ** 6]
    optics = (0.1, {1: (0.2, 0.1, 0.05, 0.1, 0.05)})
    structure = sail.layers(np.array([leaf, stem]), labels, 2, 2., (0., 0., 2., 2.), [optics])
    top, bottom = structure["frequencies"]
    assert top[0] > 0 and top[17] > 0 and bottom[17] == pytest.approx(1.)
    total_lai = structure["leaf_area_index"].sum()
    assert total_lai == pytest.approx((0.5 + 0.25 * 0.2 * 1.8) / 4.)
