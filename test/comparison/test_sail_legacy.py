"""The far field and pure SAIL (periodise, s2v, mcsail, canestrad -8 -d 0): the port's leafarea, band.spec,
cropchar, mlsail.env, Etri.vec0 and solem.dat against the legacy's, row by row as text."""
import pytest

from openalea.caribu.comparison import legacy_run, port_run

from test_direct_legacy import DIRECTIONS, MATERIALS, SOIL, mixed_scene, sensor_grid

pytestmark = pytest.mark.skipif(not legacy_run.available(), reason="libcaribu's executables are not installed")

LIGHTS = [(1., DIRECTIONS[1]), (0.5, DIRECTIONS[0]), (0.25, DIRECTIONS[2])]


def narrow_pattern(vertices):
    """A pattern narrower than the scene (triangles moved into it, some across its borders)."""
    flat = vertices.reshape(-1, 3)
    return (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()) * 0.7,
            float(flat[:, 1].max()) * 0.8)


def compare(vertices, labels, lights, layers, height=None, sensors=None, soil=SOIL):
    pattern = narrow_pattern(vertices)
    height = float(vertices.reshape(-1, 3)[:, 2].max()) if height is None else height
    legacy = legacy_run.layered(vertices, labels, lights, MATERIALS, pattern, layers, height, soil, sensors=sensors)
    port = port_run.layered(vertices, labels, lights, MATERIALS, pattern, layers, height, soil, sensors=sensors)
    assert set(port) == set(legacy) - {"scene_8.can"}
    for name, text in port.items():
        rows, legacy_rows = text.splitlines(), legacy[name].splitlines()
        assert len(rows) == len(legacy_rows), name
        differing = [(number, a, b) for number, (a, b) in enumerate(zip(legacy_rows, rows)) if a != b]
        assert not differing, (name, differing[:3])


@pytest.mark.parametrize("layers", [1, 4, 7])
def test_filter_scene(layers):
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, LIGHTS, layers)


def test_filter_scene_sensors_one_light_low_height():
    """Sensors through the canopy and above it (canestrad's profile table read beyond its top level: the next
    column's first levels); a height below the scene's top, its upper triangles above the top level."""
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, [(1., DIRECTIONS[3])], 5, sensors=sensor_grid(vertices))
    compare(vertices, labels, LIGHTS, 3, height=12.5)


def test_no_soil_reflectance():
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, LIGHTS, 4, soil=-1)


def test_adel_scene():
    vertices, labels = mixed_scene()
    compare(vertices, labels, LIGHTS, 10)
