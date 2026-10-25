"""Radiosity in the whole scene (canestrad -d -1): the port's Etri.vec0, first-order and solved radiosities (E0.dat,
B.dat) and sensors (solem.dat) against canestrad's, row by row as text."""
import pytest

from openalea.caribu.comparison import legacy_run, port_run

from test_direct_legacy import DIRECTIONS, MATERIALS, SOIL, mixed_scene, sensor_grid

pytestmark = pytest.mark.skipif(not legacy_run.available(), reason="libcaribu's canestrad is not installed")


def compare(vertices, labels, lights, sensors=None, size=1536):
    legacy = legacy_run.radiosity(vertices, labels, lights, MATERIALS, SOIL, size, sensors)
    port = port_run.radiosity(vertices, labels, lights, MATERIALS, SOIL, size, sensors)
    assert set(legacy) <= set(port)
    for name, text in legacy.items():
        rows, port_rows = text.splitlines(), port[name].splitlines()
        assert len(port_rows) == len(rows), name
        differing = [(number, a, b) for number, (a, b) in enumerate(zip(rows, port_rows)) if a != b]
        assert not differing, (name, differing[:3])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_filter_scene(direction):
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, [(1., direction)])


def test_filter_scene_sensors_several_lights():
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, [(0.3 + 0.1 * k, direction) for k, direction in enumerate(DIRECTIONS)],
            sensor_grid(vertices))


def test_adel_plants():
    """The first plants of the Adel scene (a few thousand triangles, every kind), sensors within the scene's box
    (canestrad's radiosity crashes on a sensor beyond it, out of its voxel grid)."""
    vertices, labels = mixed_scene()
    vertices, labels = vertices[:3000], labels[:3000]
    compare(vertices, labels, [(1., DIRECTIONS[1])], sensor_grid(vertices)[:-1], size=512)
