"""
Caribu as a metafspm component: the light each element of a canopy absorbs, from its triangles.

    CaribuLight(data_structure=ds, mode="nested", latitude=45.)
                                            a FunctionalComponent on a population's DataStructure, or on a
                                            UnionDataStructure of several populations
    CaribuLightModel(plant_data_structures, xrange, yrange, time_step)
                                            the environment model of a metafspm Scene running it

Its steps follow the physics:
    _optics           each element's reflectance and transmittance: an optical model's (the inputs), else its kind's
                      (its MTG label among leaf_labels, stem_labels: Wheat-BRIDGES' leaves ρ 0.10, τ 0.05, stems ρ 0.10)
    _relative_light   the canopy lit by a sky of unit energy (ds.triangles(), above the soil): Erel, each element's
                      absorbed light per unit area relative to the PAR above the canopy, its incident light, its area
    _absorbed_PAR     PARa = Erel · PARi, PAR_incident = Erel_incident · PARi, PARi the scene's forcing

The canopy is lit every *lighting_interval* (and when elements appear); in between CaribuLightModel runs only
_absorbed_PAR, PARa following PARi with the last Erel (Wheat-BRIDGES' Caribu every 4 hours). How the canopy is lit,
the same for every plant, is the component's settings: the light model (direct light, nested radiosity in the
infinite canopy, pure SAIL, radiosity in the scene alone), the sky and the sun, the site, the soil, the numerics
(precision, corrections, GPU, the linear solver). The optics are plant parameters: they may differ between plants.
The elements' parts below the soil are cut off: the soil is opaque (Caribu's infinite canopy would lift the scene to
put its lowest point at 0, lighting them). The geometry is kept between lightings (a Canopy: the sky's projections
reused) while the triangles, their optics and the settings are the same.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import (FunctionalComponent, forcing_variable, input_variable, parameter,
                                                   setting, state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.solve.decorator import rate, state

from . import api, lights
from . import corrections as corrections_module
from .canopy import Canopy

ELEMENT = dict(scale=scales.SubOrgan)


@dataclass
class CaribuLight(FunctionalComponent):
    """The PAR each element absorbs, by Caribu on the canopy's triangles: its optics (_optics), its light relative to
    the incident PAR when the canopy is lit (_relative_light), its PAR at every step (_absorbed_PAR)."""

    PAR_reflectance: float = input_variable(
        unit="adim", description="The element's reflectance in the PAR, from an optical model (NaN: its kind's)",
        by="an optical model", default=np.nan, **ELEMENT)
    PAR_transmittance: float = input_variable(
        unit="adim", description="The element's transmittance in the PAR, from an optical model (NaN: opaque)",
        by="an optical model", default=np.nan, **ELEMENT)

    PARi: float = forcing_variable(unit="µmol m-2 s-1", unit_comment="or any unit: PARa, PAR_incident take it",
                                   description="The PAR above the canopy, on a horizontal plane")
    diffuse_fraction: float = forcing_variable(
        unit="adim", description="The diffuse share of the incident radiation, with the sun (else Spitters' from "
        "PARi)", required=False)

    reflectance: float = state_variable(
        unit="adim", description="The element's reflectance in the light model", kind="intensive",
        default=np.nan, **ELEMENT)
    transmittance: float = state_variable(
        unit="adim", description="The element's transmittance in the light model (NaN: opaque, one face)",
        kind="intensive", default=np.nan, **ELEMENT)
    Erel: float = state_variable(
        unit="adim", description="The PAR the element absorbs per unit area, relative to the PAR above the canopy",
        kind="intensive", default=0., **ELEMENT)
    Erel_incident: float = state_variable(
        unit="adim", description="The PAR reaching the element per unit area (its two faces), relative to the PAR "
                                 "above the canopy", kind="intensive", default=0., **ELEMENT)
    lit_area: float = state_variable(
        unit="m2", unit_comment="the scene's unit squared", description="The element's area in the light model (its "
        "triangles' above the soil, within the pattern)", kind="extensive", default=0., **ELEMENT)
    PARa: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR the element absorbs per unit area",
        kind="intensive", default=0., **ELEMENT)
    PAR_incident: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR reaching the element per unit area",
        kind="intensive", default=0., **ELEMENT)

    leaf_reflectance: float = parameter(unit="adim", description="A leaf element's reflectance in the PAR",
                                        by="Wheat-BRIDGES", default=0.10)
    leaf_transmittance: float = parameter(unit="adim", description="A leaf element's transmittance in the PAR",
                                          by="Wheat-BRIDGES", default=0.05)
    stem_reflectance: float = parameter(unit="adim", description="A stem element's reflectance in the PAR (opaque)",
                                        by="Wheat-BRIDGES", default=0.10)
    leaf_labels: tuple = parameter(unit="", description="The MTG labels of the leaf elements (two faces, translucent)",
                                   by="Wheat-BRIDGES", default=("LeafElement", "LeafElement1"))
    stem_labels: tuple = parameter(unit="", description="The MTG labels of the stem elements (opaque)",
                                   by="Wheat-BRIDGES", default=("StemElement",))

    # the light model: "direct" (first order, Wheat-BRIDGES'), "nested" (the scattering in the infinite canopy:
    # radiosity within a sphere of sphere_diameter around each triangle, a turbid medium of layers beyond; a null
    # diameter for pure SAIL) or "radiosity" (every order of scattering in the scene alone, not infinite)
    mode: str = setting(description="The light model: 'direct', 'nested' or 'radiosity'", default="direct",
                        choices=("direct", "nested", "radiosity"))
    infinite: bool = setting(description="Direct light: the pattern repeated (nested is always infinite)",
                             default=True, choices=(True, False))
    sphere_diameter: float = setting(unit="m", unit_comment="the scene's unit",
                                     description="Nested radiosity's sphere (0: pure SAIL)", default=0.5)
    layers: int = setting(unit="adim", description="Nested radiosity's far field layers", default=10)
    sky: str = setting(description="The sky: 'sectors' (azimuth_sectors x zenith_sectors) or a turtle",
                       default="sectors", choices=("sectors", "turtle46", "turtle16", "turtle1"))
    sky_type: str = setting(description="'soc' standard overcast or 'uoc' uniform", default="soc",
                            choices=("soc", "uoc"))
    azimuth_sectors: int = setting(unit="adim", description="The sector sky's azimuths (Wheat-BRIDGES: 4)", default=4)
    zenith_sectors: int = setting(unit="adim", description="The sector sky's zeniths (Wheat-BRIDGES: 5)", default=5)
    sun: bool = setting(description="The sun with the diffuse sky, its share 1 - the diffuse fraction",
                        default=False, choices=(True, False))
    global_per_PARi: float = setting(
        unit="J µmol-1", unit_comment="W m-2 of global radiation per µmol m-2 s-1 of PARi",
        description="For the diffuse fraction without its forcing: 1 / (0.48 x 4.57), PARi in µmol m-2 s-1 (McCree)",
        default=1 / (0.48 * 4.57))
    latitude: float = setting(unit="°", description="The scene's latitude", default=48.85)
    orientation: float = setting(unit="°", description="The angle, clockwise, from the scene's X+ to North (the "
                                 "sky's and the sun's azimuths with the correction 'sky')", default=0.)
    start_day_of_year: float = setting(unit="day", description="The day of the year at the scene's time 0",
                                       default=1.)
    start_hour: float = setting(unit="h", description="The universal time hour at the scene's time 0", default=0.)
    soil_level: float = setting(unit="m", unit_comment="the scene's unit", description="The soil's height: the "
                                "elements' parts below it (sown shoots, roots) left out", default=0.)
    soil_reflectance: float = setting(unit="adim", description="The soil's reflectance in the PAR (scattering)",
                                      default=0.15)
    lighting_interval: float = setting(unit="s", description="Time between two lightings (Wheat-BRIDGES: 4 h)",
                                       default=4 * 3600.)
    xrange: float = setting(unit="m", description="The infinite canopy's pattern (0, 0, xrange, "
                                  "yrange), the scene's (None: the triangles' extent)", default=None)
    yrange: float = setting(unit="m", description="The pattern's y range (see xrange)", default=None)
    screen_size: int = setting(unit="pixel", description="The projections' screen side", default=1536)
    precision: str = setting(description="'full' the computed values, 'legacy' the legacy's printed digits",
                             default="full", choices=("full", "legacy"))
    corrections: object = setting(description="The legacy's behaviours corrected: 'recommended' (all), a tuple of "
                                  "names (openalea.caribu.corrections), or () with precision 'legacy'",
                                  default="recommended")
    device: str = setting(description="The near field's device: 'cpu' or 'cuda'", default="cpu",
                          choices=("cpu", "cuda"))
    linear_solver: str = setting(description="The radiosity system's solver: 'mgcr' (canestrad's) or 'gmres' "
                                 "(scipy's)", default="mgcr", choices=("mgcr", "gmres"))

    def __post_init__(self):
        super().__post_init__()
        corrections_module.resolve(self.corrections, self.precision)

    @rate(include_inactive=True)
    def _optics(self, PAR_reflectance, PAR_transmittance, leaf_reflectance, leaf_transmittance,
                stem_reflectance) -> dict:
        """Each element's optics: an optical model's, else its kind's (a leaf ρ, τ; a stem ρ, opaque); NaN for an
        element of another kind."""
        leaf, stem = self.element_kinds()
        given = ~np.isnan(PAR_reflectance)
        reflectance = np.where(given, PAR_reflectance,
                               np.where(leaf, leaf_reflectance, np.where(stem, stem_reflectance, np.nan)))
        transmittance = np.where(given, PAR_transmittance, np.where(leaf, leaf_transmittance, np.nan))
        return {"reflectance": reflectance, "transmittance": transmittance}

    @rate(include_inactive=True)
    def _relative_light(self, reflectance, transmittance, start_day_of_year, start_hour, soil_level) -> dict:
        """The canopy lit by a sky of unit energy: each element's absorbed and incident light per unit area (the
        area-weighted means of its triangles' Eabs and Ei), relative to the incident PAR, and its area."""
        time = self.forcing_time()
        day = start_day_of_year + (start_hour * 3600. + time) // 86400.
        hour = (start_hour + time / 3600.) % 24.
        sources = self.sources(self.forcing("PARi"), day, hour)
        triangles = self.data_structure.triangles("emerged").clip_below(soil_level)
        absorbed, incident, area = self.lit(triangles, reflectance, transmittance, sources)
        return {"Erel": absorbed, "Erel_incident": incident, "lit_area": area}

    @state(include_inactive=True)
    def _absorbed_PAR(self, Erel, Erel_incident) -> dict:
        """PARa = Erel · PARi, at every step: between two lightings, with the last Erel (Wheat-BRIDGES')."""
        PARi = self.forcing("PARi")
        return {"PARa": Erel * PARi, "PAR_incident": Erel_incident * PARi}

    # ── the sky ──────────────────────────────────────────────────────────────

    def sources(self, PARi, day, hour) -> list:
        """The light sources [(E, (vx, vy, vz))] of unit horizontal energy: the diffuse sky, and the sun with its
        share when it is on and above the horizon (the forcing diffuse_fraction, else Spitters' of the global
        radiation PARi · global_per_PARi)."""
        diffuse = 1.
        sun = []
        if self.sun:
            sun = [(energy, vector) for energy, vector in
                   lights.sun_light(1., day, hour, self.latitude, self.corrections, self.orientation)
                   if vector[2] < 0]
            if sun and PARi > 0:
                try:
                    diffuse = self.forcing("diffuse_fraction")
                except KeyError:
                    diffuse = lights.diffuse_fraction(PARi * self.global_per_PARi, day, hour, self.latitude)
            if not sun or PARi <= 0:
                sun, diffuse = [], 1.
        if self.sky == "sectors":
            sky = lights.Sky(self.azimuth_sectors, self.zenith_sectors, self.corrections,
                             self.orientation).diffuse(diffuse, self.sky_type).lights()
        else:
            sky = lights.turtle_lights(self.sky[len("turtle"):], self.sky_type, diffuse, self.corrections,
                                       self.orientation)
        return list(sky) + [(energy * (1 - diffuse), vector) for energy, vector in sun]

    # ── the optics ───────────────────────────────────────────────────────────

    def element_labels(self) -> np.ndarray:
        """Each element's MTG label name ("LeafElement", ...; a union's parts')."""
        return self.data_structure.labels()

    def element_kinds(self):
        """(leaf, stem) masks of the elements, by their labels (leaf_labels, stem_labels)."""
        labels = self.element_labels()
        return np.isin(labels, list(self.leaf_labels)), np.isin(labels, list(self.stem_labels))

    @staticmethod
    def materials(reflectance, transmittance) -> list:
        """Each element's material, Caribu's: (ρ, τ) a leaf, (ρ,) opaque; None without optics."""
        return [None if np.isnan(rho) else ((rho,) if np.isnan(tau) else (rho, tau))
                for rho, tau in zip(reflectance.tolist(), transmittance.tolist())]

    # ── the lighting ─────────────────────────────────────────────────────────

    def pattern(self, vertices):
        """The infinite canopy's pattern: the scene's ranges, else the triangles' extent."""
        if self.xrange is not None and self.yrange is not None:
            return 0., 0., float(self.xrange), float(self.yrange)
        flat = vertices.reshape(-1, 3)
        return (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))

    def lit(self, triangles, reflectance, transmittance, sources):
        """(absorbed, incident, area) per element of the canopy of *triangles*: the area-weighted means of its
        triangles' Eabs and Ei (0 for an element without lit area), the triangles' area. With precision "legacy",
        the triangles' values rounded as canestrad prints them (6 decimals), as CaribuScene aggregates them: in
        metres, small triangles' areas lose most of their digits (Wheat-BRIDGES' Erel)."""
        n_elements = reflectance.size
        if len(triangles) == 0:
            return np.zeros(n_elements), np.zeros(n_elements), np.zeros(n_elements)
        materials = self.materials(np.asarray(reflectance, dtype=float), np.asarray(transmittance, dtype=float))
        missing = sorted({str(self.element_labels()[owner]) for owner in np.unique(triangles.owner)
                          if materials[owner] is None})
        if missing:
            raise ValueError(f"CaribuLight: elements labelled {missing} have triangles but no optics (inputs "
                             "PAR_reflectance, PAR_transmittance, or their labels in leaf_labels, stem_labels)")
        species, labels = api.species_and_labels([materials[owner] for owner in triangles.owner.tolist()])
        optics = api.optics(species, self.soil_reflectance)
        canopy = self.canopy(triangles.vertices, labels)
        if self.mode == "direct":
            result = canopy.direct(sources, optics)
        elif self.mode == "nested":
            result = canopy.nested(sources, optics, self.layers, float(triangles.vertices[..., 2].max()),
                                   self.sphere_diameter, self.precision, linear_solver=self.linear_solver)
        else:
            result = canopy.radiosity(sources, optics, linear_solver=self.linear_solver)
        values = {name: np.nan_to_num(np.asarray(result[name], dtype=float)) for name in ("area", "Eabs", "Ei")}
        if self.precision == "legacy":
            values = {name: np.array(api.printed(value)) for name, value in values.items()}
        weights = np.where(values["area"] > 0, values["area"], 0.)
        return (triangles.mean_per_owner(values["Eabs"], n_elements, weights),
                triangles.mean_per_owner(values["Ei"], n_elements, weights), triangles.per_owner(weights, n_elements))

    def canopy(self, vertices, labels) -> Canopy:
        """The Canopy of these triangles, kept (not checkpointed) while they, their labels and the settings it
        depends on are the same."""
        infinite = self.mode == "nested" or (self.infinite and self.mode == "direct")
        pattern = self.pattern(vertices) if infinite else None
        made_with = (pattern, self.screen_size, self.device, corrections_module.resolve(self.corrections))
        kept, kept_with = self.kept("canopy", (None, None))
        if (kept is None or kept_with != made_with or kept.vertices.shape != vertices.shape
                or not np.array_equal(kept.vertices, vertices) or not np.array_equal(kept.labels, labels)):
            kept = Canopy(vertices, labels, pattern=pattern, size=self.screen_size, device=self.device,
                          corrections=self.corrections)
            self.keep("canopy", (kept, made_with), saved=False)
        return kept


def above_soil(triangles, level=0.):
    """*triangles*' parts above the soil at height *level* (metafspm's Triangles.clip_below)."""
    return triangles.clip_below(level)


class CaribuLightModel(EnvironmentModel):
    """The environment model of a metafspm Scene: CaribuLight on the population's DataStructure, or on the union of
    the populations' (their elements' Erel, PARa, ... then reach them through the scene translator's links). The
    canopy is lit when due (every lighting_interval, or when elements appeared); at the other steps PARa only (the
    row "state"). A subclass may light with a subclass of CaribuLight (light_component). The scenario's entries (the
    light's settings and optics) are set by the Scene once it is built."""
    light_component = CaribuLight
    compartment = "Atmosphere"

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.data_structure = self.plant_data_structure()
        self.light = self.light_component(data_structure=self.data_structure, time_step=time_step,
                                          xrange=xrange, yrange=yrange)
        self.components = [self.light]
        self.lit_elements = None

    def due(self, time) -> bool:
        """Every lighting_interval, and when elements appeared."""
        if self.lit_elements != self.data_structure.n_nodes():
            return True
        return self.last_run is None or time - self.last_run >= self.light.lighting_interval - 1e-9

    def full_run(self):
        self.light()
        self.lit_elements = self.data_structure.n_nodes()

    def between(self):
        self.light.run_stage("state")
