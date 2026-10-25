"""Nested radiosity (periodise, s2v, mcsail, canestrad -8 -d D): the port's Etri.vec0, E0.dat, B.dat, solem.dat
against canestrad's, row by row as text, and its near field against canestrad's matrix files (the counts, the
far field's weights)."""
import numpy as np
import pytest

from openalea.caribu import nested
from openalea.caribu.comparison import legacy_run, port_run
from openalea.caribu.labels import is_translucent

from test_direct_legacy import DIRECTIONS, MATERIALS, SOIL, mixed_scene, sensor_grid
from test_sail_legacy import LIGHTS, narrow_pattern

pytestmark = pytest.mark.skipif(not legacy_run.available(), reason="libcaribu's executables are not installed")


def compare(vertices, labels, layers, diameter, sensors=None, bias=True, matrices=False):
    pattern = narrow_pattern(vertices)
    height = float(vertices.reshape(-1, 3)[:, 2].max())
    legacy = legacy_run.layered(vertices, labels, LIGHTS, MATERIALS, pattern, layers, height, SOIL, sensors=sensors,
                                diameter=diameter, bias=bias)
    port = port_run.layered(vertices, labels, LIGHTS, MATERIALS, pattern, layers, height, SOIL, sensors=sensors,
                            diameter=diameter, bias=bias)
    for name in ("Etri.vec0", "E0.dat", "B.dat") + (() if sensors is None else ("solem.dat",)):
        rows, legacy_rows = port[name].splitlines(), legacy[name].splitlines()
        assert len(rows) == len(legacy_rows), name
        differing = [(number, a, b) for number, (a, b) in enumerate(zip(legacy_rows, rows)) if a != b]
        assert not differing, (name, differing[:3])
    if matrices:
        result = nested.run(vertices, labels, LIGHTS, port_run.optics(MATERIALS, SOIL), pattern, layers, height,
                            diameter, sensors=sensors, bias=bias)
        diagonal = np.frombuffer(legacy["hd/diag_ff"][16:], dtype=np.int32)
        pairs = np.frombuffer(legacy["hd/nz_ff"], dtype=np.int32).reshape(-1, 2)
        indptr, faces, values = result["counts"]
        np.testing.assert_array_equal(np.abs(diagonal) - 1, indptr)
        np.testing.assert_array_equal(pairs[:, 0], faces)
        np.testing.assert_array_equal(pairs[:, 1], values)
        weights = np.frombuffer(legacy["hd/Bfar_ff"][8:], dtype=np.float32)
        translucent = np.concatenate([is_translucent(np.asarray(labels)[result["accepted"]]),
                                      np.zeros(0 if sensors is None else len(sensors), dtype=bool)])
        port_weights = np.concatenate([np.concatenate([w[0:2].T.reshape(-1)] + ([w[2:4].T.reshape(-1)] if upper else []))
                                       for w, upper in zip(result["weights"], translucent)])
        np.testing.assert_array_equal(weights, port_weights)


@pytest.mark.parametrize("diameter, bias", [(6., True), (12., True), (12., False), (40., True)])
def test_filter_scene(diameter, bias):
    """Spheres of one and several voxels, wider than the pattern (its translated images), canestrad's -B."""
    vertices, labels = mixed_scene("filterT.can")
    compare(vertices, labels, 4, diameter, bias=bias, matrices=True)


def test_adel_plants_sensors():
    """The first plants (3,000 triangles), sensors within the grid (canestrad crashes on one beyond it)."""
    vertices, labels = mixed_scene()
    vertices, labels = vertices[:3000], labels[:3000]
    compare(vertices, labels, 6, 1.5, sensors=sensor_grid(vertices)[:-1], matrices=True)
    compare(vertices, labels, 6, 1.5, bias=False)


def test_adel_scene():
    vertices, labels = mixed_scene()
    compare(vertices, labels, 10, 0.5)
