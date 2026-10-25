"""Direct light (canestrad -1): the port's Etri.vec0 against canestrad's, row by row as text."""
import numpy as np
import pytest

from openalea.caribu import formats, geometry, periodic
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


def sensor_grid(vertices, count=6):
    """Horizontal sensor triangles at several heights through the scene, some facing down (unseen), one beyond it."""
    low, high = vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0)
    sensors = []
    for k, height in enumerate(np.linspace(low[2], high[2] + 0.1 * (high[2] - low[2]), count)):
        x0, y0 = low[0] + (k / count) * (high[0] - low[0]), low[1] + 0.2 * (high[1] - low[1])
        side = 0.3 * (high[0] - low[0])
        triangle = [(x0, y0, height), (x0 + side, y0, height), (x0, y0 + side, height)]
        sensors.append(triangle if k % 3 else triangle[::-1])
    sensors.append([(high[0] + 1., high[1] + 1., high[2]), (high[0] + 2., high[1] + 1., high[2]),
                    (high[0] + 1., high[1] + 2., high[2])])
    return np.array(sensors)


@pytest.mark.parametrize("direction", DIRECTIONS[:3])
def test_sensors(direction):
    vertices, labels = mixed_scene()
    sensors = sensor_grid(vertices)
    lights = [(1., direction), (0.5, DIRECTIONS[3])]
    legacy = legacy_run.direct(vertices, labels, lights, MATERIALS, SOIL, sensors=sensors)
    port = port_run.direct(vertices, labels, lights, MATERIALS, SOIL, sensors=sensors)
    assert port["solem.dat"] == legacy["solem.dat"]
    assert port["Etri.vec0"] == legacy["Etri.vec0"]


def adel_pattern(vertices):
    """A pattern narrower than the Adel scene, so that periodise moves triangles and many cross its borders."""
    low, high = vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0)
    extent = high - low
    return (float(low[0] + 0.1 * extent[0]), float(low[1] + 0.05 * extent[1]), float(high[0] - 0.1 * extent[0]),
            float(high[1] - 0.05 * extent[1]))


def compare_periodic(vertices, labels, lights, pattern, sensors=None):
    legacy = legacy_run.direct(vertices, labels, lights, MATERIALS, SOIL, pattern=pattern, sensors=sensors)
    port = port_run.direct(vertices, labels, lights, MATERIALS, SOIL, pattern=pattern, sensors=sensors)
    rows = [line.split() for line in legacy["scene_8.can"].splitlines() if line.strip()]
    periodised = np.array([[np.float32(value) for value in row[-9:]] for row in rows]).reshape(-1, 3, 3)
    np.testing.assert_array_equal(periodic.periodise(geometry.as_legacy_vertices(vertices), pattern), periodised)
    legacy_rows, port_rows = legacy["Etri.vec0"].splitlines(), port["Etri.vec0"].splitlines()
    assert len(port_rows) == len(legacy_rows) == len(labels)
    differing = [(number, a, b) for number, (a, b) in enumerate(zip(legacy_rows, port_rows)) if a != b]
    assert not differing, differing[:3]
    if sensors is not None:
        assert port["solem.dat"] == legacy["solem.dat"]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_periodic_filter_scene(direction):
    vertices, labels = mixed_scene("filterT.can")
    compare_periodic(vertices, labels, [(1., direction)], formats.read_pattern(data_path("filter.8")))


@pytest.mark.parametrize("direction", DIRECTIONS[1:3])
def test_periodic_adel_scene(direction):
    vertices, labels = mixed_scene()
    compare_periodic(vertices, labels, [(1., direction)], adel_pattern(vertices))


def test_periodic_adel_scene_several_lights_and_sensors():
    vertices, labels = mixed_scene()
    compare_periodic(vertices, labels, [(0.3 + 0.1 * k, direction) for k, direction in enumerate(DIRECTIONS)],
                     adel_pattern(vertices), sensor_grid(vertices))
