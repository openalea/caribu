"""The port's API beyond the legacy's: full precision, bands with a soil (a TypeError in the legacy)."""
import numpy as np

from openalea.caribu import api

LEAF = [(0., 0., 1.), (1., 0., 1.), (0., 1., 1.)]
STEM = [(0.5, 0.5, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.9)]


def test_full_precision_is_the_legacy_one_unrounded():
    triangles, materials = [LEAF, STEM], [(0.06, 0.07), (0.13,)]
    lights = [(1., (0.3, 0.1, -0.9))]
    legacy = api.raycasting(triangles, materials, lights)
    full = api.raycasting(triangles, materials, lights, precision="full")
    for name in ("area", "Eabs", "Ei_sup", "Ei_inf"):
        np.testing.assert_allclose(full[name], legacy[name], atol=5e-7)
        assert any(a != b for a, b in zip(full[name], legacy[name])) or name == "Ei_inf"


def test_bands_with_a_soil():
    scene = {"leaf": [LEAF], "stem": [STEM]}
    opt = {"par": {"leaf": (0.06, 0.07), "stem": (0.13,)}, "nir": {"leaf": (0.4, 0.45), "stem": (0.3,)}}
    caribu_scene = api.CaribuScene(scene=scene, opt=opt, soil_reflectance={"par": 0.15, "nir": 0.3},
                                   pattern=(-1., -1., 2., 2.), soil_mesh=1)
    raw, aggregated = caribu_scene.run(infinite=True)
    assert set(aggregated) == {"par", "nir"}
    soil = caribu_scene.soil_aggregated
    np.testing.assert_allclose(soil["par"]["Eabs"] / soil["nir"]["Eabs"], 0.85 / 0.7, rtol=1e-6)
    np.testing.assert_allclose(soil["par"]["Ei"], soil["nir"]["Ei"])
