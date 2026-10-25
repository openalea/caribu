"""The port's CaribuScene and raycasting against the legacy's, on the same inputs: their outputs equal."""
import numpy as np
import pytest

from openalea.caribu import api, formats
from openalea.caribu.comparison import legacy_run
from openalea.caribu.legacy import caribu as legacy_caribu
from openalea.caribu.legacy.CaribuScene import CaribuScene as LegacyScene
from openalea.caribu.legacy.data_samples import data_path

pytestmark = pytest.mark.skipif(not legacy_run.available(), reason="libcaribu's executables are not installed")

SKY = formats.read_light(data_path("Turtle16soc.light"))


def same(a, b):
    """Equal, NaN equal to NaN, through dicts and lists."""
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


def plants():
    """The Adel scene as 100 plants: the primitives by plant label, leaves and stems of two materials."""
    scene = formats.read_can(data_path("f331s1_100plantes.can"))
    triangles = [triangle for values in scene.values() for triangle in values]
    by_plant = {}
    for index, triangle in enumerate(triangles):
        by_plant.setdefault("plant%d" % (index % 37) if index % 11 else "stem%d" % (index % 3), []).append(triangle)
    material = {key: ((0.13,) if key.startswith("stem") else (0.06, 0.07)) for key in by_plant}
    return by_plant, material


def test_raycasting_function():
    scene, material = plants()
    triangles = [t for values in scene.values() for t in values][:3000]
    materials = [(0.06, 0.07) if k % 3 else (0.1, 0.05, 0.08, 0.02) for k in range(len(triangles))]
    lights = SKY[:4]
    assert same(api.raycasting(triangles, materials, lights), legacy_caribu.raycasting(triangles, materials, lights))


@pytest.mark.parametrize("options", [dict(), dict(infinite=True), dict(split_face=True, simplify=True)])
def test_scene_single_band(options):
    scene, material = plants()
    arguments = dict(scene=scene, light=SKY, pattern=(-2.6, -3.3, 0.2, 0.3), opt={"par": material})
    port = api.CaribuScene(**arguments).run(**options)
    legacy = LegacyScene(**arguments).run(**options)
    assert same(port, legacy)


def test_scene_bands_units_soil_sensors():
    scene, material = plants()
    centimetres = {key: [[tuple(100 * c for c in point) for point in t] for t in values]
                   for key, values in scene.items()}
    opt = {"par": material, "nir": {key: ((0.3,) if key.startswith("stem") else (0.4, 0.45)) for key in material}}
    soil = {"par": 0.15, "nir": 0.3}
    sensors = {"low": [[(-100., -100., 20.), (0., -100., 20.), (-100., 0., 20.)]],
               "top": [[(-50., -50., 150.), (50., -50., 150.), (-50., 50., 150.)],
                       [(-150., -50., 150.), (-50., -50., 150.), (-150., 50., 150.)]]}
    arguments = dict(scene=centimetres, light=SKY, pattern=(-260., -330., 20., 30.), opt=opt, soil_reflectance=soil,
                     scene_unit="cm")
    port = api.CaribuScene(**arguments).run(infinite=True, sensors=sensors, split_face=True)
    legacy = LegacyScene(**arguments).run(infinite=True, sensors=sensors, split_face=True)
    # the legacy's bug, not reproduced: its bands share one sensors dict, converted to m² once per band in place,
    # so the last band's sensors are converted twice
    for port_part, legacy_part in zip(port, legacy):
        doubled = legacy_part["nir"].pop("sensors")
        measured = port_part["nir"].pop("sensors")
        assert same(measured, port_part["par"]["sensors"])
        for name in ("Ei", "Ei0", "area"):
            factor = 1e-4 if name == "area" else 1e4
            for key in measured[name]:
                expected = measured[name][key]
                values = doubled[name][key]
                if isinstance(expected, list):
                    np.testing.assert_allclose(values, np.array(expected) * factor, rtol=1e-12)
                else:
                    np.testing.assert_allclose(values, expected * factor, rtol=1e-12)
    assert same(port, legacy)


def test_soil_single_band():
    scene, material = plants()
    arguments = dict(scene=scene, light=SKY, pattern=(-2.6, -3.3, 0.2, 0.3), opt={"par": material}, soil_mesh=1)
    port_scene, legacy_scene = api.CaribuScene(**arguments), LegacyScene(**arguments)
    port, legacy = port_scene.run(infinite=True), legacy_scene.run(infinite=True)
    assert same(port, legacy)
    assert same(port_scene.soil_aggregated, legacy_scene.soil_aggregated)
    assert same(port_scene.getSoilEnergy(), legacy_scene.getSoilEnergy())
    assert same(port_scene.getIncidentEnergy(), legacy_scene.getIncidentEnergy())


def test_can_and_opt_files():
    arguments = dict(scene=str(data_path("filterT.can")), light=str(data_path("zenith.light")),
                     pattern=str(data_path("filter.8")), opt=[str(data_path("par.opt")), str(data_path("nir.opt"))])
    port = api.CaribuScene(**arguments).run(infinite=True)
    legacy = LegacyScene(**arguments).run(infinite=True)
    assert same(port, legacy)
