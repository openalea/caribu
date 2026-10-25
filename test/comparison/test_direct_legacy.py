"""Direct light (canestrad -1): the port's Etri.vec0 against canestrad's, row by row as text."""
import numpy as np
import pytest

from openalea.caribu import formats
from openalea.caribu.comparison import legacy_run, port_run
from openalea.caribu.legacy.data_samples import data_path

pytestmark = pytest.mark.skipif(not legacy_run.available(), reason="libcaribu's canestrad is not installed")

# species 1 symmetric, 2 asymmetric, 3 singular (rho_sup tau_sup = rho_inf tau_inf with rho_sup = tau_inf)
MATERIALS = {1: (0.10, 0.05), 2: (0.12, 0.07, 0.03, 0.02), 3: (0.2, 0.3, 0.3, 0.2)}
SOIL = 0.15
DIRECTIONS = [(0., 0., -1.), (-0.35, 0.82, -0.33), (1.3, -0.91, -0.45), (0.29, 0.03, -0.55)]


def mixed_scene(name="f331s1_100plantes.can"):
    """A sample scene relabelled to every kind of triangle: leaves of species 1, 2 and 3, stems of species 1, soil
    (species 0), and degenerate triangles canestrad rejects, first, inside and last."""
    vertices, labels = formats.read_can_arrays(data_path(name))
    kinds = np.arange(len(labels)) % 5
    species = np.array([1, 2, 3, 1, 0])[kinds]
    leaf = np.where(kinds < 3, 1, 0)
    labels = species * 10 ** 11 + 10 ** 6 + leaf * 10 ** 3 + kinds
    point = vertices[0, 0]
    degenerate = np.array([[point, point, point + 1.],                       # coincident vertices
                           [point, point + 1., point + 2.],                  # collinear
                           [point, point + (1., 0., 0.), point + (1., 0., 0.)]])
    vertices = np.concatenate([degenerate[:1], vertices[:7], degenerate[1:2], vertices[7:], degenerate[2:]])
    labels = np.concatenate([[10 ** 11 + 10 ** 6 + 1000], labels[:7], [10 ** 11 + 10 ** 6 + 1001], labels[7:],
                             [10 ** 11 + 10 ** 6 + 1002]])
    return vertices, labels


def compare(vertices, labels, lights, size=1536):
    legacy = legacy_run.direct_etri(vertices, labels, lights, MATERIALS, SOIL, size).splitlines()
    port = port_run.direct_etri(vertices, labels, lights, MATERIALS, SOIL, size).splitlines()
    assert len(port) == len(legacy) == len(labels)
    differing = [(number, a, b) for number, (a, b) in enumerate(zip(legacy, port)) if a != b]
    assert not differing, differing[:3]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_filter_scene(direction):
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, [(1., direction)])


@pytest.mark.parametrize("direction", DIRECTIONS[:2])
def test_adel_scene(direction):
    vertices, labels = mixed_scene()
    compare(vertices, labels, [(1., direction)])


def test_adel_scene_several_lights():
    vertices, labels = mixed_scene()
    compare(vertices, labels, [(0.3 + 0.1 * k, direction) for k, direction in enumerate(DIRECTIONS)])


def test_small_image():
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, [(1., DIRECTIONS[1])], size=128)
