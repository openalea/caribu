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
    simulate(outputs="outputs")                 the run's files too: run.json, scenario.json, the
                                                summary (summaries/hand_declared/summary.csv), the stand's
                                                summaries per plant (summaries/Plant/), a frame every hour

The stand: rows of identical plants (SimplePlant), each a vertical stem and leaves as flat rectangles leaving it
on alternate sides, repeated around the stand (an infinite canopy). Lengths in m, PAR in µmol m-2 s-1.
"""
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from math import radians

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   parameter, state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.forcings import as_forcing, forcing_mean
from openalea.metafspm.solve.decorator import state
from openalea.metafspm.coupling.geometry import Triangles
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scenarios import parameter_table
from openalea.metafspm.scene.scene import Scene

from . import lights
from .component import CaribuLight, CaribuLightModel
from .optics import PlantOptics

HOUR = 3600.
GEOMETRY = dict(default=0., scale=scales.SubOrgan, kind="descriptor")
PLANT = dict(by="the user")

# The parameters a beginner changes first: the plants' leaves, the light model, the site and the optics
MAIN_PARAMETERS = ("n_leaves", "leaf_length", "leaf_width", "leaf_inclination", "plant_height", "leaf_reflectance",
                   "leaf_transmittance", "mode", "sun", "latitude", "soil_reflectance")


@dataclass
class SimplePlant(StructuralComponent):
    """A simple plant for lighting: a vertical stem (a thin opaque strip) from the plant's position (the planting's x,
    y, z) and leaves (flat rectangles), evenly spaced up the stem, the top one at the plant's height above its base,
    leaving it on alternate sides, inclined above the horizontal. Each leaf and the stem is one element of the plant
    graph."""
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
    def initiate_plant(cls, mtg, plant, parameters):
        default = {field.name: field.default for field in fields(cls)}
        value = {name: parameters.get(name, default[name]) for name in (
            "n_leaves", "leaf_length", "leaf_width", "leaf_inclination", "plant_height", "stem_width")}
        s, labels = mtg.scales, mtg.labels
        axis = mtg.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
        unit = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
        phytomer = mtg.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/', label=labels.Phytomer.Shoot))
        stem = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.StemInternode))
        blade = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.Blade))
        # the plant's position, the planting's (written before an initiator builds; z minus its sowing depth)
        x, y, z, rotation = (float(mtg.values_at(name, [plant])[1][0]) for name in ("x", "y", "z", "rotation"))
        elements = [mtg.add_component(stem, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                        label=labels.SubOrgan.StemElement))]
        geometry = [(x, y, z, rotation, 90., value["plant_height"], value["stem_width"], 0)]
        n_leaves = int(value["n_leaves"])
        for rank in range(1, n_leaves + 1):
            leaf = PropsConfig(scale=s.SubOrgan, edge_type='/' if rank == 1 else '<', label=labels.SubOrgan.LeafElement)
            elements.append(mtg.add_component(blade, **leaf) if rank == 1 else mtg.add_child(elements[-1], **leaf))
            geometry.append((x, y, z + value["plant_height"] * rank / n_leaves, rotation + 180. * (rank % 2),
                             value["leaf_inclination"], value["leaf_length"], value["leaf_width"], rank))
        for k, name in enumerate(("start_x", "start_y", "start_z", "azimuth", "inclination", "element_length",
                                  "element_width", "leaf_rank")):
            mtg.set_values(name, elements, [float(values[k]) for values in geometry])

    def triangles(self, where=None):
        """Each element a rectangle of two triangles: from its start along its direction, its width horizontal."""
        data_structure = self.data_structure
        nodes = np.arange(data_structure.n_nodes()) if where is None else np.flatnonzero(where)
        start = np.stack([np.asarray(data_structure.get(name), dtype=float)[nodes] for name in ("start_x", "start_y", "start_z")],
                         1)
        azimuth = np.radians(np.asarray(data_structure.get("azimuth"), dtype=float)[nodes])
        inclination = np.radians(np.asarray(data_structure.get("inclination"), dtype=float)[nodes])
        length = np.asarray(data_structure.get("element_length"), dtype=float)[nodes][:, None]
        half_width = 0.5 * np.asarray(data_structure.get("element_width"), dtype=float)[nodes][:, None]
        along = np.stack([np.cos(inclination) * np.cos(azimuth), np.cos(inclination) * np.sin(azimuth),
                          np.sin(inclination)], 1)
        across = np.stack([-np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)], 1)
        a, b = start - half_width * across, start + half_width * across
        c, d = b + length * along, a + length * along
        vertices = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3, 3)
        return Triangles(vertices, np.repeat(nodes, 2))


@dataclass
class AbsorbedLight(FunctionalComponent):
    """The PAR each element absorbed so far: its PARa (per unit area) over its lit area, each step."""
    PARa: float = input_variable(unit="µmol m-2 s-1", description="The PAR the element absorbs per unit area",
                                 by="CaribuLight", default=0., scale=scales.SubOrgan, kind="intensive")
    lit_area: float = input_variable(unit="m2", description="The element's lit area", by="CaribuLight", default=0.,
                                     scale=scales.SubOrgan, kind="extensive")
    absorbed_PAR: float = state_variable(unit="µmol", description="The PAR the element absorbed since the start",
                                         default=0., scale=scales.SubOrgan, kind="extensive")

    @state
    def _absorbed(self, absorbed_PAR, PARa, lit_area) -> dict:
        return {"absorbed_PAR": absorbed_PAR + np.asarray(PARa) * np.asarray(lit_area) * self.step_duration}


class SimplePlantModel(CompositeModel):
    """The plant model: SimplePlant's structure, its optics (PlantOptics) and the light it absorbed (AbsorbedLight);
    the light is the scene's environment model (CaribuLightModel), linked by name."""
    initiators = (SimplePlant,)

    parts = {"plant": SimplePlant, "optics": PlantOptics, "absorbed": AbsorbedLight}
    schedule = ("optics", "absorbed")
    def spin_up(self, scene):
        """The optics before the first lighting (its state steps, its clock left at the start)."""
        self.optics.run_stage("state")


def parameters(values: dict = None, main_only: bool = False) -> pd.DataFrame:
    """The parameters and settings, one row each: its name, its value (the default, or the one in *values*), its
    unit, its description, the model it belongs to (SimplePlant, the plants' shape; PlantOptics, their optics;
    CaribuLight, the light), whether it is one of the main ones (those first), and its bounds (min_value,
    max_value)."""
    table = parameter_table(SimplePlant, PlantOptics, CaribuLight, values=values, main=MAIN_PARAMETERS)
    table = table.rename(columns={"component": "model"})
    return table[table["main"]].reset_index(drop=True) if main_only else table


def calendar_date(day_of_year: float, hour: float = 0., year: int = 2026) -> datetime:
    """The date (UTC) of *hour* of day *day_of_year* (1: 1 January) of *year* (2026, 365 days), a Scene's start=; an
    hour out of [0, 24) moves to the day before or after."""
    return datetime(year, 1, 1) + timedelta(days=float(day_of_year) - 1., hours=float(hour))


def hours_since(scene, midnight: datetime) -> float:
    """The hours from *midnight* (a date) to *scene*'s time: a summary's hour of the day, counted on from the first
    day (25 is the next day's 1 h)."""
    return ((scene.start - midnight).total_seconds() + scene.time) / HOUR


def clear_day_PAR(day_of_year: float = 172., latitude: float = 48.85, daily_global_radiation: float = 28e6):
    """The incident PAR (µmol m-2 s-1) of a clear day, a function of the hour (universal time): the hourly share of a
    day's global radiation (J m-2), 48 % of it PAR, 4.57 µmol per joule (lights.PAR_PER_GLOBAL)."""
    def PAR(hour):
        return lights.PAR_PER_GLOBAL * max(lights.hourly_global_radiation(daily_global_radiation, hour % 24., day_of_year,
                                                                latitude), 0.)
    return PAR


class _Summary:
    """The summary row of the hour just lit; none before the first lighting."""

    def __init__(self, incident, ground: float, midnight: datetime):
        self.incident, self.ground, self.midnight = incident, ground, midnight

    def __call__(self, scene):
        if scene.step_count == 0:
            return None
        light = scene.environment[0].light
        union = light.data_structure                                    # the elements' light: the light model's
        hour = hours_since(scene, self.midnight)                        # since the first day's midnight
        PAR = forcing_mean(self.incident, scene.time - scene.time_step, scene.time)     # the hour's, as lit
        area = np.asarray(union.get("lit_area"))
        absorbed_per_ground = float(np.sum(np.asarray(union.get("Erel")) * area)) / self.ground
        intercepted = float(np.sum(np.asarray(union.get("Erel_incident")) * area)) / self.ground
        return dict(hour=hour, PARi=PAR, PAR_absorbed=PAR * absorbed_per_ground, absorbed_fraction=absorbed_per_ground,
                    soil_fraction=1. - intercepted if light.mode == "direct" else np.nan)


@dataclass
class Simulation:
    """A lit stand: its *summary* (a DataFrame, one row per hour) and its *scene* (metafspm's: the plants'
    DataStructure holds the plants' variables, the light model's union the light's)."""
    scene: object
    summary: pd.DataFrame
    ground_area: float

    @property
    def data_structure(self):
        return self.scene.populations[0].data_structure

    @property
    def light_data_structure(self):
        """The light model's DataStructure: the plants' elements' union (one population: in the plants' order)."""
        return self.scene.environment[0].light.data_structure

    def values(self, name) -> np.ndarray:
        """Each element's *name*: the plants' variable, else the light's."""
        data_structure = self.data_structure if self.data_structure.has(name) else self.light_data_structure
        return np.asarray(data_structure.get(name))

    @property
    def absorbed(self) -> np.ndarray:
        """Per element, the PAR absorbed over the run (µmol): AbsorbedLight's absorbed_PAR."""
        return self.values("absorbed_PAR")

    def elements(self) -> pd.DataFrame:
        """The elements at the last hour, one row each: its plant (0, 1, ...), kind (leaf or stem), leaf rank from the
        bottom, height (m, its middle), lit area (m2), Erel (the PAR it absorbs per unit area, relative to the PAR
        above the stand), PARa (the PAR it absorbs per unit area at the last hour, µmol m-2 s-1), absorbed (the PAR it
        absorbed over the run, µmol)."""
        data_structure = self.data_structure
        triangles = data_structure.triangles()
        middle = triangles.mean_per_owner(triangles.vertices[..., 2].mean(axis=1), data_structure.n_nodes())
        labels = np.asarray(data_structure.labels())
        return pd.DataFrame({
            "plant": np.asarray(data_structure.owner("Plant")), "kind": np.where(labels == "LeafElement", "leaf", "stem"),
            "leaf_rank": np.asarray(data_structure.get("leaf_rank")).astype(int), "height": middle,
            "area": self.values("lit_area"), "Erel": self.values("Erel"), "PARa": self.values("PARa"),
            "absorbed": self.absorbed})

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
        values = np.asarray(self.values(color_by), dtype=float)
        limits = (0., max(float(values.max()), 1e-12))
        draw(ax, self.data_structure.triangles(), colors=value_colors(values, limits=limits))
        ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*limits), cmap="viridis"), ax=ax, shrink=0.6,
                           pad=0.12, label=color_by)
        return ax


def _split(values: dict):
    """(the plants' parameters: SimplePlant's and PlantOptics', the light's settings) of *values*."""
    plant_names = {field.name for field in fields(SimplePlant)} | set(PlantOptics.declared_parameters())
    light = CaribuLight.declarations().set_index("name")["role"]
    plants, settings = {}, {}
    for name, value in values.items():
        if name in plant_names:
            plants[name] = value
        elif light.get(name) == "setting" or name == "cadence":
            settings[name] = value
        else:
            raise KeyError(f"simulate: '{name}' is neither a plant parameter (SimplePlant, PlantOptics) nor a "
                           "CaribuLight setting (parameters() lists them)")
    return plants, settings


def simulate(parameters: dict = None, hours: float = 24., start_hour: float = 0., day_of_year: float = 172.,
             PARi=None, diffuse_fraction=None, rows: int = 3, plants_per_row: int = 8, row_spacing: float = 0.15,
             plant_spacing: float = 0.05, outputs: str = None, compute=None) -> Simulation:
    """Light a stand of *rows* rows of *plants_per_row* SimplePlants (*row_spacing*, *plant_spacing* m apart), repeated
    around it (an infinite canopy), hour by hour for *hours* hours from *start_hour* (universal time) of day
    *day_of_year*, the canopy lit every hour. *parameters* ({name: value}) changes the plants' parameters (SimplePlant,
    and their optics: PlantOptics) and the light's settings (CaribuLight; "cadence", {"light": ...}, its lighting's):
    parameters() lists them all.

    Given inputs (each a number, a function of the hour of the day, or a pandas Series indexed by it):
    *PARi* (µmol m-2 s-1; a clear day's by default, clear_day_PAR), the PAR above the stand;
    *diffuse_fraction* (between 0 and 1; with the sun only, Spitters' from PARi by default), its diffuse share.

    The clock: the scene's calendar (Scene(start=), the date of its time 0: *day_of_year* of 2026, a year of 365 days)
    is that of its time 0, and a step reads its
    forcings over its hour (PARi's mean) and its sun at the hour's middle (CaribuLight's sources by panel: one panel,
    so one sun, for an hourly step): the scene starts an hour before *start_hour*, so that its first step is the hour ending at *start_hour*, the summary's
    first hour (its PARi the hour's mean).

    *outputs*: a folder for the run's files (metafspm's SceneRecorder): run.json, scenario.json,
    summaries/hand_declared/summary.csv (the summary), summaries/Plant/SimplePlantModel/summaries.csv (the light each
    plant absorbs, every hour) and a frame of the stand every hour (frames.fspm.zip, for ParaView or another platform:
    the frames folder packed at the end, the archive checked, then the folder removed). *compute*: the Scene's run configuration
    (metafspm's Compute: its threads, the light's projection threads too, its sparse solver; None: Compute())."""
    plant_values, settings = _split(dict(parameters or {}))
    x_range, y_range = plants_per_row * plant_spacing, rows * row_spacing
    plants = [Planting(x=plant_spacing * (column + 0.5), y=row_spacing * (row + 0.5), rotation=(37. * k) % 180.)
              for k, (row, column) in enumerate((row, column) for row in range(rows) for column in range(plants_per_row))]
    table = planting(plants, SimplePlantModel, plant_values, x_range=x_range, y_range=y_range)
    start = calendar_date(day_of_year, start_hour - 1.)    # the first step ends at start_hour (the clock, above)
    settings = {"cadence": {"light": "1 h"}, **settings}
    incident = PARi if PARi is not None else clear_day_PAR(day_of_year, settings.get("latitude", CaribuLight.latitude))
    forcings = {"PARi": as_forcing(incident, time_unit=HOUR, origin="calendar")}          # by the hour of the day
    if diffuse_fraction is not None:
        forcings["diffuse_fraction"] = as_forcing(diffuse_fraction, time_unit=HOUR, origin="calendar")
    ground = x_range * y_range
    summary = _Summary(forcings["PARi"], ground, calendar_date(day_of_year))
    recorder = SceneRecorder(outputs, summaries=outputs is not None, tables={"summary": summary},
                             frame_period="step" if outputs is not None else None, pack_at_finish=outputs is not None)
    built = Scene(table, environment=[CaribuLightModel], environment_scenarios=[settings], time_step=HOUR,
                  forcings=forcings, recorder=recorder, start=start, compute=compute)
    built.simulate(int(round(hours)))
    return Simulation(scene=built, summary=recorder.table("summary"), ground_area=ground)
