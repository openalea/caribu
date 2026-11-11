"""CaribuLight, the metafspm component: on a population's DataStructure and on a union of two in a Scene, against
CaribuScene on the same triangles and optics; the lighting's cadence, the optics, the sky and the sun."""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.caribu import api, lights
from openalea.caribu.component import CaribuLight, CaribuLightModel
from openalea.caribu.optics import PlantOptics
from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.geometry import HalfSpace, Triangles
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.scene import Scene

PANEL = dict(unit="m", description="The panel's geometry", default=0., scale=scales.SubOrgan,
             kind="descriptor", on_grow="inherit")
SKY = lights.Sky(4, 5).diffuse(1., "soc").lights()
XRANGE, YRANGE = 0.3, 0.2
HOUR = 3600.


@dataclass
class Panels(StructuralComponent):
    """A plant of a vertical stem panel and leaf panels above each other, tilted: each element a square of two
    triangles (centre, side, tilt about x)."""
    centre_x: float = state_variable(**PANEL)
    centre_y: float = state_variable(**PANEL)
    centre_z: float = state_variable(**PANEL)
    side: float = state_variable(**PANEL)
    tilt: float = state_variable(**PANEL)

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        s, labels = mtg.scales, mtg.labels
        axis = mtg.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
        unit = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
        phytomer = mtg.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/', label=labels.Phytomer.Shoot))
        stem = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.StemInternode))
        blade = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.Blade))
        x, y = mtg.property("x")[plant], mtg.property("y")[plant]
        elements = [mtg.add_component(stem, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                        label=labels.SubOrgan.StemElement))]
        values = [(x, y, 0.1, 0.05, np.pi / 2)]
        for rank in range(int(parameters.get("n_leaves", 3))):
            leaf = PropsConfig(scale=s.SubOrgan, edge_type='/' if rank == 0 else '<', label=labels.SubOrgan.LeafElement)
            elements.append(mtg.add_component(blade, **leaf) if rank == 0 else mtg.add_child(elements[-1], **leaf))
            values.append((x + 0.02 * rank, y, 0.12 + 0.06 * rank, 0.08, 0.3 * rank))
        for v, value in zip(elements, values):
            for name, number in zip(("centre_x", "centre_y", "centre_z", "side", "tilt"), value):
                mtg.set_values(name, [v], [float(number)])

    def triangles(self, where=None):
        data_structure = self.data_structure
        centre = np.stack([np.asarray(data_structure.get(name), dtype=float) for name in ("centre_x", "centre_y", "centre_z")], 1)
        half, tilt = 0.5 * np.asarray(data_structure.get("side")), np.asarray(data_structure.get("tilt"))
        nodes = np.arange(centre.shape[0]) if where is None else np.flatnonzero(where)
        corners = []
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(centre[nodes] + np.stack([u * half[nodes], v * half[nodes] * np.cos(tilt[nodes]),
                                                     v * half[nodes] * np.sin(tilt[nodes])], 1))
        a, b, c, d = corners
        vertices = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3, 3)
        return Triangles(vertices, np.repeat(nodes, 2))


@dataclass
class LightUse(FunctionalComponent):
    """A plant component reading the PAR its elements absorb."""
    PARa: float = input_variable(unit="µmol m-2 s-1", description="The PAR the element absorbs", by="CaribuLight",
                                 default=-1., scale=scales.SubOrgan)


def plant_model(name):
    """A plant model of its own classes (a Scene's component classes are one per model): its panels, their optics
    (PlantOptics: Wheat-BRIDGES' by default) and a component reading the PAR they absorb."""
    panels = dataclass(type(f"{name}Panels", (Panels,), {}))
    optics = dataclass(type(f"{name}Optics", (PlantOptics,), {}))
    use = dataclass(type(f"{name}LightUse", (LightUse,), {}))

    class Model(CompositeModel):
        initiators = (panels,)

        parts = {"panels": panels, "optics": optics, "use": use}
        schedule = ("optics",)
        def spin_up(self, scene):
            self.optics.run_stage("state")
    Model.__name__ = name
    return Model


Wheat, Pea = plant_model("Wheat"), plant_model("Pea")


def planted(models, n_leaves=3):
    """One plant of each model in a row (metafspm's planting), *n_leaves* leaves each."""
    return planting([Planting(x=0.07 + 0.12 * i, y=0.1) for i in range(len(models))], list(models),
                    {"n_leaves": n_leaves}, x_range=XRANGE, y_range=YRANGE)


def scene(models, settings=None, translator=None, PARi=500., soil_surface=0.):
    scenario = dict(settings or {})
    return Scene(planted(models), environment=[CaribuLightModel], environment_scenarios=[scenario],
                 translator=translator, time_step=HOUR, forcings={"PARi": lambda time: PARi},
                 soil_surface=soil_surface)


def caribu_scene(triangles, labels, mode="direct", **run):
    """CaribuScene (full precision, the corrections) on the same triangles, a primitive per element, Wheat-BRIDGES'
    optics."""
    materials = {"LeafElement": (0.10, 0.05), "StemElement": (0.10,)}
    primitives = {}
    for vertices, owner in zip(triangles.vertices.tolist(), triangles.owner.tolist()):
        primitives.setdefault(owner, []).append(vertices)
    opt = {"par": {owner: materials[labels[owner]] for owner in primitives}}
    caribu = api.CaribuScene(scene=primitives, light=SKY, pattern=(0, 0, XRANGE, YRANGE), opt=opt, precision="full",
                             soil_reflectance={"par": 0.15}, corrections="recommended")
    _, aggregated = caribu.run(direct=mode == "direct", infinite=True, **run)
    return aggregated["par"]


def test_one_population_as_caribu_scene():
    """One population: CaribuLight on its union (of one), each element's Erel and lit area as CaribuScene's
    Eabs, PARa = Erel · PARi, PARa reaching the plant model's reader by the default identity link."""
    built = scene([Wheat, Wheat])
    built.run()
    union, plants = built.environment[0].data_structure, built.populations[0].data_structure
    light = built.environment[0].light
    expected = caribu_scene(union.triangles(), light.element_labels().tolist())
    for element, value in expected["Eabs"].items():
        assert union.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert union.get("lit_area")[element] == pytest.approx(expected["area"][element], rel=1e-12)
    np.testing.assert_allclose(union.get("PARa"), 500. * union.get("Erel"))
    assert 0 < union.get("Erel").max() < 1 and (union.get("Erel_incident") >= union.get("Erel")).all()
    np.testing.assert_array_equal(plants.get("PARa"), union.get("PARa"))         # into the plants, at their step
    assert not plants.has("Erel")                                                 # the light's stays the light's

def test_the_canopy_is_lit_at_its_cadence_PARa_following_PARi_every_step():
    PARi = {"value": 500.}
    built = Scene(planted([Wheat]), environment=[CaribuLightModel], environment_scenarios=[{"cadence": {"light": 2 * HOUR}}],
                  time_step=HOUR, forcings={"PARi": lambda time: PARi["value"]})
    plants, union = built.populations[0].data_structure, built.environment[0].data_structure
    built.run()
    first = union.get("Erel").copy()
    plants.set("tilt", plants.get("tilt") + 0.5)            # a new geometry, seen at the next lighting
    PARi["value"] = 200.
    built.run()
    np.testing.assert_array_equal(union.get("Erel"), first)
    np.testing.assert_allclose(union.get("PARa"), 200. * first)
    built.run()
    assert not np.allclose(union.get("Erel"), first)

def test_two_populations_in_a_union():
    """Two models: CaribuLight on their union, its outputs to each population by the default identity links (one per
    population); the canopy is lit as one, equal to CaribuScene on the triangles of both."""
    built = scene([Wheat, Pea, Wheat])
    union = built.environment[0].data_structure
    assert len(union.parts) == 2
    built.run()
    light = built.environment[0].light
    expected = caribu_scene(union.triangles(), light.element_labels().tolist())
    for element, value in expected["Eabs"].items():
        assert union.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
    built.run()                                           # the exchanges into the plants, at their next step
    wheat, pea = (population.data_structure for population in built.populations)
    for part, data_structure in enumerate((wheat, pea)):          # the union in plant order: each part's positions
        np.testing.assert_array_equal(data_structure.get("PARa"), union.get("PARa")[union.part_positions(part)])



def test_a_renamed_input_by_the_scene_translator():
    """A plant input under another name (a photosynthesis' absorbed PAR in mol): the scene translator's link, a
    rename with its unit's factor."""
    @dataclass
    class Photosynthesis(FunctionalComponent):
        absorbed_PAR_density: float = input_variable(unit="mol m-2 s-1", description="The PAR the element absorbs",
                                                     by="CaribuLight", default=-1., scale=scales.SubOrgan)

    class Model(CompositeModel):
        initiators = (Panels,)

        parts = {"panels": Panels, "optics": PlantOptics, "photosynthesis": Photosynthesis}
        schedule = ("optics",)
        def spin_up(self, scene):
            self.optics.run_stage("state")

    translator = Translator().link("Photosynthesis", "absorbed_PAR_density", "CaribuLight", {"PARa": 1e-6})
    built = scene([Model], translator=translator)
    built.run()
    built.run()                                           # the exchanges into the plants, at their next step
    union, plants = built.environment[0].data_structure, built.populations[0].data_structure
    np.testing.assert_allclose(plants.get("absorbed_PAR_density"), 1e-6 * union.get("PARa"), rtol=1e-12)
    assert union.get("PARa").max() > 0 and not plants.has("PARa")


def test_optics_from_the_plants():
    """The light's optics are the plants' (PAR_reflectance, PAR_transmittance): an element's own, an opaque one
    (NaN transmittance); an element with triangles but no optics raises."""
    built = scene([Wheat])
    union = built.environment[0].data_structure
    light = built.environment[0].light
    labels = light.element_labels()
    leaves = np.flatnonzero(labels == "LeafElement")
    reflectance = np.where(labels == "LeafElement", 0.10, 0.10)
    transmittance = np.where(labels == "LeafElement", 0.05, np.nan)
    reflectance[leaves[0]], transmittance[leaves[0]] = 0.2, 0.1
    reflectance[leaves[1]], transmittance[leaves[1]] = 0.3, np.nan                 # an opaque leaf
    expected = [(0.10,) if label == "StemElement" else (0.10, 0.05) for label in labels]
    expected[leaves[0]], expected[leaves[1]] = (0.2, 0.1), (0.3,)
    assert sorted(set(labels)) == ["LeafElement", "StemElement"]
    union.set("PAR_reflectance", reflectance)
    union.set("PAR_transmittance", transmittance)
    light.run_stage("rate")
    assert light.materials(union.get("reflectance"), union.get("transmittance")) == expected
    union.set("PAR_reflectance", np.where(labels == "LeafElement", np.nan, 0.10))
    with pytest.raises(ValueError, match="no optics"):                              # the leaves without optics
        light.run_stage("rate")

def test_nested_mode_as_caribu_scene():
    built = scene([Wheat, Wheat], {"mode": "nested", "sphere_diameter": 0.1, "layers": 5})
    built.run()
    data_structure = built.environment[0].data_structure
    light = built.environment[0].light
    triangles = data_structure.triangles()
    expected = caribu_scene(triangles, light.element_labels().tolist(), mode="nested", d_sphere=0.1, layers=5,
                            height=float(triangles.vertices[..., 2].max()))
    for element, value in expected["Eabs"].items():
        assert data_structure.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_the_sun_shares_the_unit_energy_and_sets():
    built = scene([Wheat], {"sun": True, "latitude": 48.85})
    light = built.environment[0].light
    day = light.sources(2000., 172, 12.)                  # a clear noon: a diffuse fraction of 0.2
    assert len(day) == 21 and sum(energy for energy, _ in day) == pytest.approx(1.)
    assert day[-1][1][2] < 0 and 0 < day[-1][0] < 1
    night = light.sources(0., 172, 1.)
    assert len(night) == 20 and sum(energy for energy, _ in night) == pytest.approx(1.)


def test_above_soil():
    """A triangle across the soil cut at its plane (one or two triangles, the area above kept), one below dropped
    (metafspm's Triangles.clip, vectorised, which CaribuLight lights: data_structure.geometry.clipped("air"))."""
    vertices = np.array([[[0, 0, -1], [2, 0, 1], [0, 2, 1]],         # one vertex below: two triangles
                         [[0, 0, 1], [2, 0, -1], [0, 2, -1]],         # two below: one triangle
                         [[0, 0, -1], [1, 0, -1], [0, 1, -2]],        # all below: none
                         [[0, 0, 0], [1, 0, 1], [0, 1, 0]]], dtype=float)  # on the soil: kept whole
    kept = Triangles(vertices, [0, 1, 2, 3]).clip(HalfSpace(0.))
    assert kept.owner.tolist() == [0, 0, 1, 3]
    assert (kept.vertices[..., 2] >= 0).all()
    areas = kept.per_owner(kept.areas(), 4)
    whole = Triangles(vertices, [0, 1, 2, 3]).areas()
    np.testing.assert_allclose(areas, [whole[0] * 3 / 4, whole[1] / 4, 0., whole[3]])


def test_the_parts_below_the_soil_receive_no_light():
    """The soil 10 cm up (the Scene's soil_surface): the stems' lower halves left out (their lit area halved), the
    light as CaribuScene's on the parts above."""
    built = scene([Wheat], soil_surface=0.1)
    built.run()
    data_structure = built.environment[0].data_structure
    light = built.environment[0].light
    assert light.soil_surface == 0.1
    stems = light.element_labels() == "StemElement"
    np.testing.assert_allclose(data_structure.get("lit_area")[stems], 0.05 * 0.05 / 2, rtol=1e-6)
    expected = caribu_scene(data_structure.triangles().clip(HalfSpace(0.1)), light.element_labels().tolist())
    for element, value in expected["Eabs"].items():
        assert data_structure.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_the_sun_over_the_step_is_consistent_with_its_mean_PAR():
    """The sun of a lit step (step_sources): an hourly step's at its middle (PARi its mean over the step); a daily
    step's course, a sun per hour weighted by that hour's PAR, the energies summing to 1 with the diffuse skies
    summed into one."""
    from openalea.caribu.simulation import clear_day_PAR
    from openalea.metafspm.coupling.forcings import as_forcing
    def by_hour():                                                     # (a forcing is set at its scene's calendar)
        return as_forcing(clear_day_PAR(172, 48.85), time_unit="h", origin="calendar")
    built = Scene(planted([Wheat]), environment=[CaribuLightModel], environment_scenarios=[{"sun": True}],
                  time_step=HOUR, forcings={"PARi": by_hour()}, start="2026-06-21 11:00")
    light = built.environment[0].light
    light.__dict__["_clock"] = 0.                                      # the step [11 h, 12 h]
    hourly = light.step_sources()
    middle = light.sources(light.forcing("PARi"), 172, 11.5)
    assert len(hourly) == 21 and sum(energy for energy, _ in hourly) == pytest.approx(1.)
    np.testing.assert_allclose(hourly[-1][1], middle[-1][1])                           # the sun at 11:30
    assert hourly[-1][1] != pytest.approx(light.sources(light.forcing("PARi"), 172, 12.)[-1][1])
    daily = Scene(planted([Wheat]), environment=[CaribuLightModel], environment_scenarios=[{"sun": True}],
                  time_step=24 * HOUR, forcings={"PARi": by_hour()}, start="2026-06-21 00:00")
    light = daily.environment[0].light
    sources = light.step_sources()
    suns = sources[20:]
    assert 12 <= len(suns) <= 18                                       # the sun's daylight hours
    assert sum(energy for energy, _ in sources) == pytest.approx(1.)
    noon = max(range(len(suns)), key=lambda k: suns[k][0])
    assert -suns[noon][1][2] == pytest.approx(max(-vector[2] for _, vector in suns))   # the most at the highest sun


def test_optics_per_plant():
    """The optics are plant parameters (PlantOptics): two genotypes' leaves reflect differently."""
    built = scene([Wheat, Wheat])
    plants, union = built.populations[0].data_structure, built.environment[0].data_structure
    plants.set("leaf_reflectance", np.array([0.08, 0.2]))
    built.populations[0].instance.optics.run_stage("state")      # as at the plants' next step
    built.run()
    first = np.asarray(plants.owner("Plant")) == 0
    labels = built.environment[0].light.element_labels()
    leaf, stem = labels == "LeafElement", labels == "StemElement"
    np.testing.assert_allclose(union.get("reflectance")[leaf & first], 0.08)
    np.testing.assert_allclose(union.get("reflectance")[leaf & ~first], 0.2)
    np.testing.assert_allclose(union.get("reflectance")[stem], 0.10)
    assert np.isnan(union.get("transmittance")[stem]).all()

def test_the_forcings_are_declared():
    """PARi is a declared forcing, required: a scene without it fails when built; the diffuse fraction is optional."""
    assert CaribuLight.declared_forcings() == {"PARi": True, "diffuse_fraction": False}
    with pytest.raises(ValueError, match="CaribuLight.PARi"):
        Scene(planted([Wheat]), environment=[CaribuLightModel], time_step=HOUR)



def test_the_pattern_is_the_scenes():
    """The infinite canopy's pattern is the environment model's (the planting's ranges): not a scenario entry."""
    built = scene([Wheat])
    assert built.environment[0].light.pattern == built.environment[0].pattern == (0., 0., XRANGE, YRANGE)
    with pytest.raises(ValueError, match="pattern"):
        scene([Wheat], {"pattern": (0., 0., 1., 1.)})

def test_a_subclass_lights_with_its_component():
    """CaribuLightModel.light_component: a subclass lights with a subclass of CaribuLight, its added steps run."""
    from openalea.metafspm.solve.decorator import state as state_step

    @dataclass
    class FluxLight(CaribuLight):
        PAR_flux: float = state_variable(unit="µmol s-1", description="PARa times the lit area",
                                         kind="extensive", default=0., scale=scales.SubOrgan)

        @state_step(include_inactive=True)
        def _PAR_flux(self, PARa, lit_area) -> dict:
            return {"PAR_flux": PARa * lit_area}

    class FluxLightModel(CaribuLightModel):
        light_component = FluxLight

    built = Scene(planted([Wheat]), environment=[FluxLightModel], time_step=HOUR, forcings={"PARi": lambda time: 500.})
    built.run()
    data_structure = built.environment[0].data_structure
    assert isinstance(built.environment[0].light, FluxLight)
    np.testing.assert_allclose(data_structure.get("PAR_flux"), data_structure.get("PARa") * data_structure.get("lit_area"))
    assert np.sum(data_structure.get("PAR_flux")) > 0


def test_the_threads_bound_the_projection_threads(monkeypatch):
    """threads: the threads projecting the sources ('auto' half the processors), the Scene's Compute's unless the
    scenario gives it; a bad value raises."""
    from concurrent import futures
    from openalea.caribu import direct
    from openalea.metafspm.compute import Compute
    bounds = []

    class Recording(futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, **options):
            bounds.append(max_workers)
            super().__init__(max_workers=max_workers, **options)

    monkeypatch.setattr(direct, "ThreadPoolExecutor", Recording)
    built = scene([Wheat], {"threads": 2})
    built.run()
    assert bounds == [2]
    built = Scene(planted([Wheat]), environment=[CaribuLightModel], time_step=HOUR,
                  forcings={"PARi": lambda time: 500.}, compute=Compute(threads=3))
    assert built.environment[0].light.threads == 3
    assert scene([Wheat]).environment[0].light.projection_threads() == max(1, (__import__("os").cpu_count() or 1) // 2)
    for threads in (0, "all", 1.5):
        with pytest.raises(ValueError, match="threads"):
            CaribuLight(data_structure=built.environment[0].data_structure, threads=threads)


def test_the_screen_follows_the_pixel_size():
    """pixel_size (m): the screen's side the scene's horizontal extent over it (the pattern's here), so that a larger
    pattern keeps its resolution; None: 1536 pixels. The light at a pixel size giving 1536 pixels is the default's."""
    from openalea.caribu.component import SCREEN_SIZE
    reference = scene([Wheat])
    reference.run()
    light = reference.environment[0].light
    assert light.screen_size(None, light.pattern) == SCREEN_SIZE == 1536
    fine = scene([Wheat], {"pixel_size": XRANGE / 1536})
    fine.run()
    assert fine.environment[0].light.screen_size(None, light.pattern) == 1536
    np.testing.assert_array_equal(fine.environment[0].data_structure.get("Erel"),
                                  reference.environment[0].data_structure.get("Erel"))
    assert scene([Wheat], {"pixel_size": 1e-3}).environment[0].light.screen_size(None, light.pattern) == 300
    with pytest.raises(ValueError, match="pixel_size"):
        scene([Wheat], {"pixel_size": 0.})
