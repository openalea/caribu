"""
Caribu for users: a small crop stand lit in a few lines, its results as tables.

    from openalea.caribu.simulation import simulate, parameters

    result = simulate()                         a stand of simple plants lit over a clear summer day, hour by hour
    result.summary                              per hour: the incident PAR, the PAR the stand absorbs, its fractions
    result.elements()                           per leaf and stem: its plant, height, area, relative absorbed light
    result.plants()                             per plant: the PAR it absorbed over the run
    result.draw()                               the stand in 3D, coloured by the light its leaves absorb
    parameters()                                every parameter and setting: its value, unit and description
    simulate({"leaf_inclination": 70.})         more erect leaves
    simulate(PARi=series)                       a measured incident PAR (µmol m-2 s-1), by hour

The stand: rows of identical plants (SimplePlant), each a vertical stem and leaves as flat rectangles leaving it
on alternate sides, repeated around the stand (an infinite canopy). Lengths in m, PAR in µmol m-2 s-1.
"""
from dataclasses import dataclass, fields
from math import radians

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, parameter, state_variable
from openalea.metafspm.coupling.geometry import Triangles
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.scene import Scene

from . import lights
from .component import CaribuLight, CaribuLightModel

HOUR = 3600.
GEOMETRY = dict(initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
PLANT = dict(by="the user")

# The parameters a beginner changes first: the plants' leaves, the light model, the site and the optics
MAIN_PARAMETERS = ("n_leaves", "leaf_length", "leaf_width", "leaf_inclination", "plant_height", "leaf_reflectance",
                   "leaf_transmittance", "mode", "sun", "latitude", "soil_reflectance")


@dataclass
class SimplePlant(StructuralComponent):
    """A simple plant for lighting: a vertical stem (a thin opaque strip) and leaves (flat rectangles), evenly spaced
    up the stem, the top one at the plant's height, leaving it on alternate sides, inclined above the horizontal.
    Each leaf and the stem is one element of the plant graph."""
    n_leaves: int = parameter(unit="adim", description="The number of leaves", default=4, **PLANT)
    leaf_length: float = parameter(unit="m", description="A leaf's length", default=0.2, **PLANT)
    leaf_width: float = parameter(unit="m", description="A leaf's width", default=0.015, **PLANT)
    leaf_inclination: float = parameter(unit="°", description="A leaf's angle above the horizontal (0: flat, 90: "
                                                               "erect)", default=30., **PLANT)
    plant_height: float = parameter(unit="m", description="The height of the stem, where the top leaf leaves it",
                                    default=0.4, **PLANT)
    stem_width: float = parameter(unit="m", description="The stem's width", default=0.004, **PLANT)

    start_x: float = state_variable(unit="m", description="Where the element starts, x", **GEOMETRY)
    start_y: float = state_variable(unit="m", description="Where the element starts, y", **GEOMETRY)
    start_z: float = state_variable(unit="m", description="Where the element starts, z (its height)", **GEOMETRY)
    azimuth: float = state_variable(unit="°", description="The element's direction, from x towards y", **GEOMETRY)
    inclination: float = state_variable(unit="°", description="The element's angle above the horizontal",
                                        **GEOMETRY)
    element_length: float = state_variable(unit="m", description="The element's length", **GEOMETRY)
    element_width: float = state_variable(unit="m", description="The element's width", **GEOMETRY)
    leaf_rank: int = state_variable(unit="adim", description="The leaf's rank from the bottom (0: the stem)",
                                    **GEOMETRY)

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        default = {field.name: field.default for field in fields(cls)}
        value = {name: parameters.get(name, default[name]) for name in (
            "n_leaves", "leaf_length", "leaf_width", "leaf_inclination", "plant_height", "stem_width")}
        s, labels = g.scales, g.labels
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
        unit = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
        phytomer = g.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/', label=labels.Phytomer.Shoot))
        stem = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.StemInternode))
        blade = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.Blade))
        x, y, rotation = (g.property(name)[plant] for name in ("x", "y", "rotation"))
        elements = [g.add_component(stem, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                        label=labels.SubOrgan.StemElement))]
        geometry = [(x, y, 0., rotation, 90., value["plant_height"], value["stem_width"], 0)]
        n_leaves = int(value["n_leaves"])
        for rank in range(1, n_leaves + 1):
            leaf = PropsConfig(scale=s.SubOrgan, edge_type='/' if rank == 1 else '<', label=labels.SubOrgan.LeafElement)
            elements.append(g.add_component(blade, **leaf) if rank == 1 else g.add_child(elements[-1], **leaf))
            geometry.append((x, y, value["plant_height"] * rank / n_leaves, rotation + 180. * (rank % 2),
                             value["leaf_inclination"], value["leaf_length"], value["leaf_width"], rank))
        for element, values in zip(elements, geometry):
            for name, number in zip(("start_x", "start_y", "start_z", "azimuth", "inclination", "element_length",
                                     "element_width", "leaf_rank"), values):
                g.property(name)[element] = float(number)

    def triangles(self, where=None):
        """Each element a rectangle of two triangles: from its start along its direction, its width horizontal."""
        ds = self.data_structure
        nodes = np.arange(ds.n_nodes()) if where is None else np.flatnonzero(where)
        start = np.stack([np.asarray(ds.get(name), dtype=float)[nodes] for name in ("start_x", "start_y", "start_z")],
                         1)
        azimuth = np.radians(np.asarray(ds.get("azimuth"), dtype=float)[nodes])
        inclination = np.radians(np.asarray(ds.get("inclination"), dtype=float)[nodes])
        length = np.asarray(ds.get("element_length"), dtype=float)[nodes][:, None]
        half_width = 0.5 * np.asarray(ds.get("element_width"), dtype=float)[nodes][:, None]
        along = np.stack([np.cos(inclination) * np.cos(azimuth), np.cos(inclination) * np.sin(azimuth),
                          np.sin(inclination)], 1)
        across = np.stack([-np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], 1)
        a, b = start - half_width * across, start + half_width * across
        c, d = b + length * along, a + length * along
        vertices = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3, 3)
        return Triangles(vertices, np.repeat(nodes, 2))


class SimplePlantModel:
    """The plant model: SimplePlant alone (the light is the scene's environment model)."""
    initiators = (SimplePlant,)

    def __init__(self, data_structure, time_step):
        self.plant = SimplePlant(data_structure=data_structure)
        self.components = [self.plant]

    def run(self):
        pass


def parameters(values: dict = None, main_only: bool = False) -> pd.DataFrame:
    """The parameters and settings, one row each: its name, its value (the default, or the one in *values*), its
    unit, its description, the model it belongs to (SimplePlant, the plants; CaribuLight, the light), and whether it
    is one of the main ones (those first)."""
    tables = []
    for model in (SimplePlant, CaribuLight):
        declarations = model.declarations()
        table = declarations[declarations["kind"].isin(("parameter", "setting"))][
            ["name", "default", "unit", "description"]].copy()
        tables.append(table.assign(model=model.__name__))
    table = pd.concat(tables).rename(columns={"default": "value"}).reset_index(drop=True)
    for name, value in (values or {}).items():
        table.loc[table["name"] == name, "value"] = value
    table["main"] = table["name"].isin(MAIN_PARAMETERS)
    order = {name: rank for rank, name in enumerate(MAIN_PARAMETERS)}
    table = table.assign(rank=table["name"].map(order).fillna(len(order))).sort_values("rank", kind="stable")
    table = table.drop(columns="rank").reset_index(drop=True)
    return table[table["main"]].reset_index(drop=True) if main_only else table


def clear_day_PAR(day_of_year: float = 172., latitude: float = 48.85, daily_global_radiation: float = 28e6):
    """The incident PAR (µmol m-2 s-1) of a clear day, a function of the hour (universal time): the hourly share of a
    day's global radiation (J m-2), 48 % of it PAR, 4.57 µmol per joule."""
    def PAR(hour):
        return 0.48 * 4.57 * max(lights.hourly_global_radiation(daily_global_radiation, hour % 24., day_of_year,
                                                                latitude), 0.)
    return PAR


def _value_at(value, hour: float) -> float:
    """A given input at *hour*: a number; a function of the hour; a pandas Series indexed by hour, or a sequence (one
    value per hour from start_hour), linearly interpolated."""
    if callable(value):
        return float(value(hour))
    if isinstance(value, pd.Series):
        return float(np.interp(hour, value.index.to_numpy(dtype=float), value.to_numpy(dtype=float)))
    if np.ndim(value) > 0:
        values = np.asarray(value, dtype=float)
        return float(np.interp(hour, np.arange(values.size), values))
    return float(value)


@dataclass
class Simulation:
    """A lit stand: its *summary* (a DataFrame, one row per hour) and its *scene* (metafspm's: its DataStructure holds
    every variable of every element)."""
    scene: object
    summary: pd.DataFrame
    ground_area: float
    absorbed: np.ndarray                # per element, the PAR absorbed over the run (µmol)

    @property
    def data_structure(self):
        return self.scene.populations[0].data_structure

    def elements(self) -> pd.DataFrame:
        """The elements at the last hour, one row each: its plant (0, 1, ...), kind (leaf or stem), leaf rank from the
        bottom, height (m, its middle), lit area (m2), Erel (the PAR it absorbs per unit area, relative to the PAR
        above the stand), PARa (the PAR it absorbs per unit area at the last hour, µmol m-2 s-1), absorbed (the PAR it
        absorbed over the run, µmol)."""
        ds = self.data_structure
        triangles = ds.triangles()
        middle = triangles.mean_per_owner(triangles.vertices[..., 2].mean(axis=1), ds.n_nodes())
        labels = np.asarray(ds.labels())
        return pd.DataFrame({
            "plant": np.asarray(ds.owner("Plant")), "kind": np.where(labels == "LeafElement", "leaf", "stem"),
            "leaf_rank": np.asarray(ds.get("leaf_rank")).astype(int), "height": middle,
            "area": np.asarray(ds.get("lit_area")), "Erel": np.asarray(ds.get("Erel")),
            "PARa": np.asarray(ds.get("PARa")), "absorbed": self.absorbed})

    def plants(self) -> pd.DataFrame:
        """Per plant: its leaf area (m2) and the PAR it absorbed over the run (mol)."""
        elements = self.elements()
        return pd.DataFrame({"leaf_area": elements[elements["kind"] == "leaf"].groupby("plant")["area"].sum(),
                             "absorbed": elements.groupby("plant")["absorbed"].sum() / 1e6})

    def draw(self, ax=None, color_by: str = "Erel"):
        """The stand in 3D, its elements coloured by *color_by* (Erel by default); returns the matplotlib axes."""
        import matplotlib.pyplot as plt
        from openalea.metafspm.display import draw, value_colors

        if ax is None:
            ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
        values = np.asarray(self.data_structure.get(color_by), dtype=float)
        limits = (0., max(float(values.max()), 1e-12))
        draw(ax, self.data_structure.triangles(), colors=value_colors(values, limits=limits))
        ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*limits), cmap="viridis"), ax=ax, shrink=0.6,
                           pad=0.12, label=color_by)
        return ax


def _split(values: dict):
    """(the plants' parameters, the light's settings, the light's plant parameters) of *values*."""
    plant_names = {field.name for field in fields(SimplePlant)}
    light = CaribuLight.declarations().set_index("name")["kind"]
    plants, settings, optics = {}, {}, {}
    for name, value in values.items():
        if name in plant_names:
            plants[name] = value
        elif light.get(name) == "setting":
            settings[name] = value
        elif light.get(name) == "parameter":
            optics[name] = value
        else:
            raise KeyError(f"simulate: '{name}' is neither a SimplePlant nor a CaribuLight parameter or setting "
                           "(parameters() lists them)")
    return plants, settings, optics


def simulate(parameters: dict = None, hours: float = 24., start_hour: float = 0., day_of_year: float = 172.,
             PARi=None, diffuse_fraction=None, rows: int = 3, plants_per_row: int = 8, row_spacing: float = 0.15,
             plant_spacing: float = 0.05) -> Simulation:
    """Light a stand of *rows* rows of *plants_per_row* SimplePlants (*row_spacing*, *plant_spacing* m apart), repeated
    around it (an infinite canopy), hour by hour for *hours* hours from *start_hour* (universal time) of day
    *day_of_year*, the canopy lit every hour. *parameters* ({name: value}) changes the plants' parameters (SimplePlant)
    and the light's parameters and settings (CaribuLight): parameters() lists them all.

    Given inputs (each a number, a function of the hour of the day, or a pandas Series indexed by it):
    *PARi* (µmol m-2 s-1; a clear day's by default, clear_day_PAR), the PAR above the stand;
    *diffuse_fraction* (between 0 and 1; with the sun only, Spitters' from PARi by default), its diffuse share."""
    plant_values, settings, optics = _split(dict(parameters or {}))
    xrange, yrange = plants_per_row * plant_spacing, rows * row_spacing
    plants = [Planting(x=plant_spacing * (column + 0.5), y=row_spacing * (row + 0.5), rotation=(37. * k) % 180.)
              for k, (row, column) in enumerate((row, column) for row in range(rows) for column in range(plants_per_row))]
    table = planting(plants, SimplePlantModel, plant_values, xrange=xrange, yrange=yrange)
    # the scene's clock starts an hour before start_hour: a step lights the canopy at its end, start_hour first
    clock_start = start_hour - 1.
    settings = {"start_day_of_year": day_of_year, "start_hour": clock_start, "lighting_interval": HOUR, **settings}
    incident = PARi if PARi is not None else clear_day_PAR(settings["start_day_of_year"],
                                                           settings.get("latitude", CaribuLight.latitude))
    forcings = {"PARi": lambda time: _value_at(incident, clock_start + time / HOUR)}
    if diffuse_fraction is not None:
        forcings["diffuse_fraction"] = lambda time: _value_at(diffuse_fraction, clock_start + time / HOUR)
    built = Scene(table, environment=[CaribuLightModel], environment_scenarios=[settings], time_step=HOUR,
                  forcings=forcings)
    ds = built.populations[0].data_structure
    for name, value in optics.items():
        ds.set(name, np.full(len(ds.entity_ids("Plant")), float(value)))
    light = built.environment[0].light
    ground = xrange * yrange
    rows_out, absorbed = [], np.zeros(ds.n_nodes())
    for _ in range(int(round(hours))):
        built.run()
        hour = clock_start + built.time / HOUR
        PAR = _value_at(incident, hour)
        area = np.asarray(ds.get("lit_area"))
        absorbed_per_ground = float(np.sum(np.asarray(ds.get("Erel")) * area)) / ground
        intercepted = float(np.sum(np.asarray(ds.get("Erel_incident")) * area)) / ground
        absorbed += np.asarray(ds.get("PARa")) * area * HOUR
        rows_out.append(dict(hour=hour, PARi=PAR, PAR_absorbed=PAR * absorbed_per_ground,
                             absorbed_fraction=absorbed_per_ground,
                             soil_fraction=1. - intercepted if light.mode == "direct" else np.nan))
    return Simulation(scene=built, summary=pd.DataFrame(rows_out), ground_area=ground, absorbed=absorbed)
