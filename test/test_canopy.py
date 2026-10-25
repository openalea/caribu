"""Canopy: repeated lighting of a static scene from its caches gives the results of fresh runs."""
import numpy as np
import pytest

from openalea.caribu import direct, formats, nested, radiosity
from openalea.caribu.canopy import Canopy
from openalea.caribu.comparison import port_run
from openalea.caribu.legacy.data_samples import data_path

OPTICS = port_run.optics({1: (0.10, 0.05)}, 0.15)
OTHER_OPTICS = port_run.optics({1: (0.40, 0.45)}, 0.3)
SKY = formats.read_light(data_path("Turtle16soc.light"))


@pytest.fixture(scope="module")
def plants():
    vertices, labels = formats.read_can_arrays(data_path("f331s1_100plantes.can"))
    flat = vertices[:2000].reshape(-1, 3)
    pattern = (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))
    return vertices[:2000], labels[:2000], pattern


def same(a, b):
    for name in ("Eabs", "Ei", "Ei_sup", "Ei_inf", "area"):
        np.testing.assert_array_equal(a[name], b[name])


def test_direct_from_cached_directions(plants):
    vertices, labels, pattern = plants
    for canopy_pattern in (None, pattern):
        canopy = Canopy(vertices, labels, pattern=canopy_pattern)
        canopy.direct(SKY, OPTICS)
        brighter = [(2 * energy, direction) for energy, direction in SKY]
        same(canopy.direct(brighter, OTHER_OPTICS),
             direct.run(vertices, labels, brighter, OTHER_OPTICS, pattern=canopy_pattern))


def test_radiosity_from_cached_form_factors(plants):
    vertices, labels, _ = plants
    canopy = Canopy(vertices[:600], labels[:600], size=512)
    canopy.radiosity(SKY[:3], OPTICS)
    same(canopy.radiosity(SKY[3:6], OTHER_OPTICS),
         radiosity.run(vertices[:600], labels[:600], SKY[3:6], OTHER_OPTICS, size=512))


def test_nested(plants):
    vertices, labels, pattern = plants
    canopy = Canopy(vertices, labels, pattern=pattern)
    arguments = dict(layers=5, height=1.2, diameter=0.4)
    canopy.nested(SKY, OPTICS, **arguments)
    fresh = nested.run(vertices, labels, SKY[:4], OTHER_OPTICS, pattern, **arguments)
    same(canopy.nested(SKY[:4], OTHER_OPTICS, **arguments), fresh)  # the legacy's precision: recomputed
    canopy.nested(SKY, OPTICS, precision="full", **arguments)
    fresh = nested.run(vertices, labels, SKY[:4], OTHER_OPTICS, pattern, precision="full", **arguments)
    reused = canopy.nested(SKY[:4], OTHER_OPTICS, precision="full", **arguments)  # the far field from the weights
    for name in ("Eabs", "Ei_sup"):
        np.testing.assert_allclose(reused[name], fresh[name], rtol=1e-5)
