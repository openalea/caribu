"""CaribuLight on Adel's port (ShootArchitecture's triangles): a stand grown in a metafspm Scene, lit every 4 hours,
against the port's CaribuScene and against the legacy run as Wheat-BRIDGES runs it (its sky from GenSky and
GetLight, its optics per element, CaribuScene(...).run(direct=True, infinite=True), Erel its aggregated Eabs)."""
import numpy as np
import pytest

pytest.importorskip("openalea.adel")

from openalea.adel.development import ScenarioShoot, random_delays, sown_stand
from openalea.caribu import api
from openalea.caribu.component import CaribuLightModel
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.scene.population import planting
from openalea.metafspm.scene.scene import Scene

N_PLANTS, N_ROWS, ROW_SPACING, PLANT_SPACING = 8, 2, 0.15, 0.04
XRANGE, YRANGE = N_PLANTS // N_ROWS * PLANT_SPACING, N_ROWS * ROW_SPACING
HOUR = 3600.
WHEAT_BRIDGES_OPTICS = {"LeafElement": (0.10, 0.05), "StemElement": (0.10,)}


@pytest.fixture(scope="module")
def stand():
    """The stand after 61 steps of 10 °Cd, lit every 4 steps: at its last step, before the plants grow in it (the
    Scene runs the environment first), on the triangles of the 60 steps before, kept."""
    Choregrapher().reset()
    plants = sown_stand(N_PLANTS, n_rows=N_ROWS, row_spacing=ROW_SPACING, plant_spacing=PLANT_SPACING)
    delays = random_delays(N_PLANTS)
    table = planting(plants, ScenarioShoot, {"n_sect": 2, "thermal_time_step": 10.},
                     [{"delay": float(delay)} for delay in delays], xrange=XRANGE, yrange=YRANGE)
    scene = Scene(table, environment=[CaribuLightModel], time_step=HOUR, forcings={"PARi": lambda time: 400.})
    for _ in range(60):
        scene.run()
    triangles = scene.populations[0].data_structure.triangles()
    scene.run()
    return scene, triangles


def _elements(stand):
    stand, triangles = stand
    ds = stand.populations[0].data_structure
    light = stand.environment[0].light
    primitives = {}
    for vertices, owner in zip(triangles.vertices.tolist(), triangles.owner.tolist()):
        primitives.setdefault(owner, []).append([tuple(point) for point in vertices])
    labels = light.element_labels()
    opt = {"par": {owner: WHEAT_BRIDGES_OPTICS[labels[owner]] for owner in primitives}}
    return ds, primitives, opt


def test_the_stand_is_lit_as_by_caribu_scene(stand):
    ds, primitives, opt = _elements(stand)
    assert len(primitives) > 50
    sky = stand[0].environment[0].light.sources(400., 1, 0, 48.85, 4, 5, 1.)
    _, aggregated = api.CaribuScene(scene=primitives, light=sky, pattern=(0, 0, XRANGE, YRANGE), opt=opt,
                                    precision="full", corrections="recommended").run(direct=True, infinite=True)
    erel = np.asarray(ds.get("Erel"))
    for element, value in aggregated["par"]["Eabs"].items():
        assert erel[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(ds.get("PARa"), 400. * erel)
    lit = np.isin(np.arange(erel.size), list(primitives))
    assert (erel[~lit] == 0).all() and erel[lit].max() > 0.5


def test_wheat_bridges_use_reproduced(stand):
    """The legacy (libcaribu) as Wheat-BRIDGES' LightModel calls it: equal to its printed digits."""
    from openalea.caribu.comparison import legacy_run
    if not legacy_run.available():
        pytest.skip("libcaribu is not installed")
    from openalea.caribu.legacy.CaribuScene import CaribuScene as LegacyScene
    from openalea.caribu.legacy.sky_tools import GenSky, GetLight
    ds, primitives, opt = _elements(stand)
    sky = []
    for line in GetLight.GetLight(GenSky.GenSky()(1, "soc", 4, 5)).split("\n"):
        if line:
            values = [float(value) for value in line.split(" ")]
            sky.append((values[0], tuple(values[1:4])))
    _, aggregated = LegacyScene(scene=primitives, light=sky, pattern=(0, 0, XRANGE, YRANGE),
                                opt=opt).run(direct=True, infinite=True)
    light = stand[0].environment[0].light
    full = np.asarray(ds.get("Erel"))
    light.precision, light.corrections = "legacy", ()  # the legacy's numbers: its printed digits, no correction
    nan = np.full(full.size, np.nan)
    erel, _, _ = light.lit(stand[1], nan, nan, light.sources(400., 1, 0, 48.85, 4, 5, 1.), 0.5, 10, 0.15, 1536)
    light.precision, light.corrections = "full", "recommended"
    for element, value in aggregated["par"]["Eabs"].items():
        assert erel[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
    # the printed areas of small triangles (m²) and the corrections move Erel
    assert 1e-3 < np.abs(erel - full).max() < 0.1
