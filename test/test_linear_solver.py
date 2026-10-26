"""The radiosity system solved by scipy's GMRES: the radiosities of meschach's MGCR (canestrad's), full and nested
radiosity; refused with the legacy's precision."""
import numpy as np
import pytest

from openalea.caribu import lights, nested, radiosity
from test_corrections import LEAF, OPTICS, random_canopy


@pytest.mark.parametrize("model", ["radiosity", "nested"])
def test_gmres_as_mgcr(model):
    vertices = random_canopy(200)
    sky = lights.turtle_lights("16", "soc", 1.)

    def run(linear_solver):
        if model == "radiosity":
            return radiosity.run(vertices, LEAF, sky, OPTICS, size=512, linear_solver=linear_solver)
        return nested.run(vertices, LEAF, sky, OPTICS, (0, 0, 0.4, 0.4), 5, 0.65, 0.1, size=512, precision="full",
                          linear_solver=linear_solver)
    mgcr, gmres = run("mgcr"), run("gmres")
    np.testing.assert_allclose(gmres["radiosity"], mgcr["radiosity"], rtol=1e-6, atol=1e-9)
    assert 0 < gmres["steps"] <= 2 * mgcr["steps"]


def test_gmres_needs_the_full_precision():
    with pytest.raises(ValueError):
        nested.run(random_canopy(20), LEAF, lights.turtle_lights("16", "soc", 1.), OPTICS, (0, 0, 0.4, 0.4), 5, 0.65,
                   0.1, linear_solver="gmres")
