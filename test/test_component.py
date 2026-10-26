"""CaribuLight, the metafspm component: on a population's DataStructure and on a union of two in a Scene, against
CaribuScene on the same triangles and optics; the lighting interval, the optics, the sky and the sun."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.caribu import api, lights
from openalea.caribu.component import CaribuLight, CaribuLightModel
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   state_variable)
from openalea.metafspm.coupling.geometry import Triangles
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene

PANEL = dict(unit="m", description="The panel's geometry", initialize=0., scale=scales.SubOrgan,
             state_variable_type="descriptor", on_grow="inherit")
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
    def initiate_plant(cls, g, plant, parameters):
        s, labels = g.scales, g.labels
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
        unit = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
        phytomer = g.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/', label=labels.Phytomer.Shoot))
        stem = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.StemInternode))
        blade = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.Blade))
        x, y = g.property("x")[plant], g.property("y")[plant]
        elements = [g.add_component(stem, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                        label=labels.SubOrgan.StemElement))]
        values = [(x, y, 0.1, 0.05, np.pi / 2)]
        for rank in range(int(parameters.get("n_leaves", 3))):
            leaf = PropsConfig(scale=s.SubOrgan, edge_type='/' if rank == 0 else '<', label=labels.SubOrgan.LeafElement)
            elements.append(g.add_component(blade, **leaf) if rank == 0 else g.add_child(elements[-1], **leaf))
            values.append((x + 0.02 * rank, y, 0.12 + 0.06 * rank, 0.08, 0.3 * rank))
        for v, value in zip(elements, values):
            for name, number in zip(("centre_x", "centre_y", "centre_z", "side", "tilt"), value):
                g.property(name)[v] = float(number)

    def triangles(self, where=None):
        ds = self.data_structure
        centre = np.stack([np.asarray(ds.get(name), dtype=float) for name in ("centre_x", "centre_y", "centre_z")], 1)
        half, tilt = 0.5 * np.asarray(ds.get("side")), np.asarray(ds.get("tilt"))
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
                                 initialize=-1., scale=scales.SubOrgan)


def plant_model(name):
    """A plant model of its own classes (a Scene's component classes are one per model)."""
    panels = dataclass(type(f"{name}Panels", (Panels,), {}))
    use = dataclass(type(f"{name}LightUse", (LightUse,), {}))

    class Model:
        initiators = (panels,)

        def __init__(self, data_structure, time_step):
            self.panels = panels(data_structure=data_structure)
            self.use = use(data_structure=data_structure)
            self.components = [self.panels, self.use]

        def run(self):
            pass
    Model.__name__ = name
    return Model


Wheat, Pea = plant_model("Wheat"), plant_model("Pea")


def planting(models, n_leaves=3):
    table = pd.DataFrame([dict(plant=f"p{i}", model=model, x=0.07 + 0.12 * i, y=0.1, z=0., rotation=0.,
                               scenario={"parameters": {"n_leaves": n_leaves}}) for i, model in enumerate(models)])
    table.attrs.update(xrange=XRANGE, yrange=YRANGE)
    return table


def scene(models, environment_scenario=None, translator=None, PARi=500.):
    Choregrapher().reset()
    return Scene(planting(models), environment=[CaribuLightModel], environment_scenarios=[environment_scenario or {}],
                 translator=translator, time_step=HOUR, forcings={"PARi": lambda time: PARi})


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
    """Wheat-BRIDGES' use: an infinite canopy under the 4 × 5 standard overcast sky, Erel the elements' area-weighted
    Eabs, PARa = Erel · PARi."""
    built = scene([Wheat, Wheat])
    built.run()
    ds = built.populations[0].data_structure
    light = built.environment[0].light
    expected = caribu_scene(ds.triangles(), light.element_labels().tolist())
    for element, value in expected["Eabs"].items():
        assert ds.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
        assert ds.get("lit_area")[element] == pytest.approx(expected["area"][element], rel=1e-12)
    np.testing.assert_allclose(ds.get("PARa"), 500. * ds.get("Erel"))
    assert 0 < ds.get("Erel").max() < 1 and (ds.get("Erel_incident") >= ds.get("Erel")).all()


def test_the_canopy_is_lit_every_interval_PARa_following_PARi():
    PARi = {"value": 500.}
    Choregrapher().reset()
    built = Scene(planting([Wheat]), environment=[CaribuLightModel],
                  environment_scenarios=[{"lighting_interval": 2 * HOUR}], time_step=HOUR,
                  forcings={"PARi": lambda time: PARi["value"]})
    ds = built.populations[0].data_structure
    built.run()
    first = ds.get("Erel").copy()
    ds.set("tilt", ds.get("tilt") + 0.5)                 # a new geometry, seen at the next lighting
    PARi["value"] = 200.
    built.run()
    np.testing.assert_array_equal(ds.get("Erel"), first)
    np.testing.assert_allclose(ds.get("PARa"), 200. * first)
    built.run()
    assert not np.allclose(ds.get("Erel"), first)


def test_two_populations_in_a_union():
    """Two models: CaribuLight on their union, its outputs to each population by the translator; the canopy is lit
    as one, equal to CaribuScene on the triangles of both."""
    translator = Translator()
    for model in ("Wheat", "Pea"):
        translator.link(f"{model}LightUse", "PARa", "CaribuLight", {"PARa": 1.})
    built = scene([Wheat, Pea, Wheat], translator=translator)
    union = built.environment[0].data_structure
    assert len(union.parts) == 2
    built.run()
    light = built.environment[0].light
    expected = caribu_scene(union.triangles(), light.element_labels().tolist())
    for element, value in expected["Eabs"].items():
        assert union.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)
    built.run()                                           # the exchanges into the plants, at their next step
    wheat, pea = (population.data_structure for population in built.populations)
    np.testing.assert_array_equal(np.r_[wheat.get("PARa"), pea.get("PARa")], union.get("PARa"))


def test_optics_from_inputs_and_labels():
    built = scene([Wheat])
    ds = built.populations[0].data_structure
    light = built.environment[0].light
    labels = light.element_labels()
    leaves = np.flatnonzero(labels == "LeafElement")
    reflectance = np.full(ds.n_nodes(), np.nan)
    transmittance = np.full(ds.n_nodes(), np.nan)
    reflectance[leaves[0]], transmittance[leaves[0]] = 0.2, 0.1
    reflectance[leaves[1]] = 0.3                          # an opaque leaf
    expected = [(0.10,) if label == "StemElement" else (0.10, 0.05) for label in labels]
    expected[leaves[0]], expected[leaves[1]] = (0.2, 0.1), (0.3,)
    assert sorted(set(labels)) == ["LeafElement", "StemElement"]
    assert light.materials(reflectance, transmittance) == expected
    light.optical_properties = (("StemElement", (0.1,)),)
    with pytest.raises(ValueError):                       # the leaves without optics
        nan = np.full(ds.n_nodes(), np.nan)
        light.lit(ds.triangles(), nan, nan, SKY, 0.5, 10, 0.15, 1536)


def test_nested_mode_as_caribu_scene():
    built = scene([Wheat, Wheat], environment_scenario={"mode": "nested", "sphere_diameter": 0.1, "layers": 5})
    built.run()
    ds = built.populations[0].data_structure
    light = built.environment[0].light
    triangles = ds.triangles()
    expected = caribu_scene(triangles, light.element_labels().tolist(), mode="nested", d_sphere=0.1, layers=5,
                            height=float(triangles.vertices[..., 2].max()))
    for element, value in expected["Eabs"].items():
        assert ds.get("Erel")[element] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_the_sun_shares_the_unit_energy_and_sets():
    built = scene([Wheat], environment_scenario={"sun": True, "start_day_of_year": 172, "latitude": 48.85})
    light = built.environment[0].light
    day = light.sources(2000., 172, 12., 48.85, 4, 5, 1 / (0.48 * 4.57))    # a clear noon: a diffuse fraction of 0.2
    assert len(day) == 21 and sum(energy for energy, _ in day) == pytest.approx(1.)
    assert day[-1][1][2] < 0 and 0 < day[-1][0] < 1
    night = light.sources(0., 172, 1., 48.85, 4, 5, 1.)
    assert len(night) == 20 and sum(energy for energy, _ in night) == pytest.approx(1.)
