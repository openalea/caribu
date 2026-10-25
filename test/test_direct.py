"""Direct light on cases with a known answer."""
import numpy as np
import pytest

from openalea.caribu import direct
from openalea.caribu.labels import encode_labels

SQUARE = np.array([[(0., 0., 1.), (1., 0., 1.), (1., 1., 1.)], [(0., 0., 1.), (1., 1., 1.), (0., 1., 1.)]])
OPTICS = (0.15, {1: (0.2, 0.1, 0.05, 0.08, 0.04)})


@pytest.mark.parametrize("direction", [(0., 0., -1.), (0.5, 0.2, -0.6)])
def test_horizontal_leaf_receives_the_horizontal_irradiance(direction):
    """A lone horizontal leaf facing up receives E on its upper face, nothing below, and absorbs E (1 − ρ − τ)."""
    result = direct.run(SQUARE, encode_labels(1, leaf=1), [(2., direction)], OPTICS)
    assert result["accepted"].all()
    np.testing.assert_allclose(result["area"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(result["Ei_sup"], 2., rtol=5e-3)
    np.testing.assert_allclose(result["Ei_inf"], 0., atol=1e-6)
    np.testing.assert_allclose(result["Eabs"], 2. * (1 - 0.1 - 0.05), rtol=5e-3)


def test_leaf_lit_from_below_uses_its_lower_face():
    flipped = SQUARE[:, ::-1]
    result = direct.run(flipped, encode_labels(1, leaf=1), [(1., (0., 0., -1.))], OPTICS)
    np.testing.assert_allclose(result["Ei_inf"], 1., rtol=5e-3)
    np.testing.assert_allclose(result["Ei_sup"], 0., atol=1e-6)
    np.testing.assert_allclose(result["Eabs"], 1 - 0.08 - 0.04, rtol=5e-3)


def test_opaque_and_soil():
    labels = np.array([encode_labels(1), encode_labels(0)])
    result = direct.run(SQUARE, labels, [(1., (0., 0., -1.))], OPTICS)
    np.testing.assert_allclose(result["Ei"], 1., rtol=5e-3)
    np.testing.assert_array_equal(result["Ei_inf"], -1.)
    np.testing.assert_allclose(result["Eabs"], [0.8, 0.85], rtol=5e-3)


def test_shading():
    """A leaf right under another one is in its shade."""
    below = SQUARE - (0., 0., 0.5)
    labels = encode_labels(1, leaf=1, element=[0, 1, 2, 3])
    result = direct.run(np.concatenate([SQUARE, below]), labels, [(1., (0., 0., -1.))], OPTICS)
    np.testing.assert_allclose(result["Ei_sup"][:2], 1., rtol=5e-3)
    np.testing.assert_allclose(result["Ei_sup"][2:], 0., atol=1e-6)


def test_rejected_triangles_keep_their_place():
    degenerate = np.array([[(0., 0., 0.), (0., 0., 0.), (1., 0., 0.)]])
    result = direct.run(np.concatenate([degenerate, SQUARE]), encode_labels(1, leaf=1, element=[0, 1, 2]),
                        [(1., (0., 0., -1.))], OPTICS)
    assert result["accepted"].tolist() == [False, True, True]
    assert result["area"][0] == 0 and np.isnan(result["Eabs"][0])
