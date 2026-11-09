"""
Caribu as a metafspm component: the light each element of a canopy absorbs, from its triangles.

    CaribuLight(data_structure=ds, mode="nested", latitude=45., pattern=(0., 0., 0.4, 0.45))
                                            a FunctionalComponent on a population's DataStructure, or on a
                                            UnionDataStructure of several populations; its pattern (m) the
                                            infinite canopy's (None: the triangles' extent)
    CaribuLightModel(plant_data_structures, xrange, yrange, time_step)
                                            the environment model of a metafspm Scene running it

    PlantOptics(data_structure=ds)          a plant model's: its elements' optics, by kind, per plant (optics)

Its steps follow the physics:
    _optics           each element's reflectance and transmittance in the light model: the plants' (PAR_reflectance,
                      PAR_transmittance, from their optical model, e.g. PlantOptics); an element without (NaN, e.g. a
                      root) is left out
    _relative_light   the canopy lit by a sky of unit energy (ds.triangles(), above the soil): Erel, each element's
                      absorbed light per unit area relative to the PAR above the canopy, its incident light, its area
    _absorbed_PAR     PARa = Erel · PARi, PAR_incident = Erel_incident · PARi, PARi the scene's forcing

The canopy is lit every 4 h (Wheat-BRIDGES'), its lighting covering the interval (metafspm's cadence at its start:
the sun's course over the interval, Erel its PAR-weighted mean), and when elements appear or go (geometry_input);
_absorbed_PAR runs every step, PARa following PARi with the last Erel: CaribuLightModel's schedule, the lighting
(its rate stage) at its cadence and the absorbed PAR (its state stage) every step; its scenario's "cadence"
({"light": "1 h"}) sets the lighting's. The sun is the Scene's calendar's (Scene(start="2026-06-21 11:00"), the date of its time 0). How the canopy is lit, the same for
every plant, is the component's settings: the light model (direct light, nested radiosity in the infinite canopy,
pure SAIL, radiosity in the scene alone), the sky and the sun, the site, the soil, the numerics (precision,
corrections, GPU, the radiosity method). The optics are the plants': an optical component of their plant model
(PlantOptics), whose parameters may differ between plants.
The elements' parts below the soil are cut off: the soil is opaque (Caribu's infinite canopy would lift the scene to
put its lowest point at 0, lighting them). The geometry is kept between lightings (a Canopy: the sky's projections
reused) while the triangles, their optics and the settings are the same.
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import (FunctionalComponent, forcing_variable, geometry_input,
                                                   input_variable, setting, state_variable)
from openalea.metafspm.coupling.schedule import Step
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.solve.decorator import rate, state

from . import api, lights
from . import corrections as corrections_module
from .canopy import Canopy

ELEMENT = dict(scale=scales.SubOrgan)
# The date of time 0 of a component lit without a Scene start (Scene(start=)): 1 January, midnight (UTC)
WITHOUT_START = "2026-01-01 00:00"
SCREEN_SIZE = 1536             # pixels: the projections' screen side without a pixel_size (the reference's)


@dataclass
class CaribuLight(FunctionalComponent):
    """The PAR each element absorbs, by Caribu on the canopy's triangles: its optics (_optics), its light relative to
    the incident PAR when the canopy is lit (_relative_light), its PAR at every step (_absorbed_PAR)."""

    PAR_reflectance: float = input_variable(
        unit="adim", description="The element's reflectance in the PAR, from its plant model's optics (NaN: not lit)",
        by="PlantOptics", default=np.nan, **ELEMENT)
    PAR_transmittance: float = input_variable(
        unit="adim", description="The element's transmittance in the PAR, from its plant model's optics (NaN: "
        "opaque)", by="PlantOptics", default=np.nan, **ELEMENT)
    geometry: object = geometry_input(description="The plants' elements' triangles (ds.triangles())", relight="topology")

    PARi: float = forcing_variable(unit="µmol m-2 s-1", unit_comment="or any unit: PARa, PAR_incident take it",
                                   description="The PAR above the canopy, on a horizontal plane")
    diffuse_fraction: float = forcing_variable(
        unit="adim", description="The diffuse share of the incident radiation, with the sun (else Spitters' from "
        "PARi)", required=False)

    reflectance: float = state_variable(
        unit="adim", description="The element's reflectance in the light model", kind="intensive", weight="lit_area",
        default=np.nan, **ELEMENT)
    transmittance: float = state_variable(
        unit="adim", description="The element's transmittance in the light model (NaN: opaque, one face)",
        kind="intensive", weight="lit_area", default=np.nan, **ELEMENT)
    Erel: float = state_variable(
        unit="adim", description="The PAR the element absorbs per unit area, relative to the PAR above the canopy",
        kind="intensive", weight="lit_area", default=0., **ELEMENT)
    Erel_incident: float = state_variable(
        unit="adim", description="The PAR reaching the element per unit area (its two faces), relative to the PAR "
                                 "above the canopy", kind="intensive", weight="lit_area", default=0., **ELEMENT)
    lit_area: float = state_variable(
        unit="m2", unit_comment="the scene's unit squared", description="The element's area in the light model (its "
        "triangles' above the soil, within the pattern)", kind="extensive", default=0., **ELEMENT)
    PARa: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR the element absorbs per unit area",
        kind="intensive", weight="lit_area", default=0., **ELEMENT)
    PAR_incident: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR reaching the element per unit area",
        kind="intensive", weight="lit_area", default=0., **ELEMENT)

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
        description="For the diffuse fraction without its forcing: 1 / (0.48 x 4.57), PARi in µmol m-2 s-1 (McCree; "
        "lights.PAR_PER_GLOBAL)", default=1 / lights.PAR_PER_GLOBAL)
    latitude: float = setting(unit="°", description="The scene's latitude", default=48.85)
    orientation: float = setting(unit="°", description="The angle, clockwise, from the scene's X+ to North (the "
                                 "sky's and the sun's azimuths with the correction 'sky')", default=0.)
    soil_reflectance: float = setting(unit="adim", description="The soil's reflectance in the PAR (scattering)",
                                      default=0.15)
    pixel_size: float = setting(unit="m", unit_comment="the scene's unit",
                                description="The projections' pixel side over the scene's horizontal extent (the "
                                "screen's side: that extent / pixel_size pixels), so that a larger pattern keeps its "
                                "resolution; None: a screen of SCREEN_SIZE pixels (1536) whatever the extent",
                                default=None)
    threads: object = setting(description="The threads projecting the light sources (each holding a screen of "
                              "depths): 'auto' half the processors, or a number; in a Scene, its Compute's threads "
                              "unless the scenario gives it", default="auto")
    precision: str = setting(description="'full' the computed values, 'legacy' the legacy's printed digits",
                             default="full", choices=("full", "legacy"))
    corrections: object = setting(description="The legacy's behaviours corrected: 'recommended' (all), a tuple of "
                                  "names (openalea.caribu.corrections), or () with precision 'legacy'",
                                  default="recommended")
    device: str = setting(description="The near field's device: 'cpu' or 'cuda'", default="cpu",
                          choices=("cpu", "cuda"))
    radiosity_method: str = setting(description="The radiosity system's iterative method: 'mgcr' (meschach's "
                                    "MGCR, canestrad's) or 'gmres' (scipy's GMRES)", default="mgcr",
                                    choices=("mgcr", "gmres"))

    # The infinite canopy's pattern (x1, y1, x2, y2, m), a construction argument, not a setting: in a Scene its
    # environment model's (EnvironmentModel.pattern, the scene's); None: the triangles' extent
    pattern: Optional[tuple] = None

    def __post_init__(self):
        super().__post_init__()
        corrections_module.resolve(self.corrections, self.precision)

    def check_settings(self) -> None:
        """The settings' choices, the threads ('auto' or a positive number) and the pixel size (positive or None),
        at construction and after a scenario."""
        super().check_settings()
        threads = self.threads
        if threads != "auto" and (isinstance(threads, bool) or not isinstance(threads, (int, np.integer))
                                  or threads < 1):
            raise ValueError(f"CaribuLight: threads is 'auto' or a positive number, got {self.threads!r}")
        if self.pixel_size is not None and not float(self.pixel_size) > 0:
            raise ValueError(f"CaribuLight: pixel_size is a positive length (m) or None, got {self.pixel_size!r}")

    @rate(include_inactive=True)
    def _optics(self, PAR_reflectance, PAR_transmittance) -> dict:
        """Each element's optics: its plant model's (NaN: not lit, e.g. a root; a NaN transmittance: opaque)."""
        return {"reflectance": np.asarray(PAR_reflectance, dtype=float),
                "transmittance": np.asarray(PAR_transmittance, dtype=float)}

    @rate(include_inactive=True)
    def _relative_light(self, reflectance, transmittance) -> dict:
        """The canopy lit by a sky of unit energy: each element's absorbed and incident light per unit area (the
        area-weighted means of its triangles' Eabs and Ei), relative to the incident PAR, and its area; its parts
        above the soil only (the scene's soil surface, metafspm's ds.geometry: "air"). The sun over the step
        (step_sources): its course by panels of at most an hour, each weighted by its PAR (the Scene's calendar; day
        1, 0 h at time 0 without one)."""
        sources = self.step_sources()
        triangles = self.data_structure.geometry.clipped("air", where="planted")
        absorbed, incident, area = self.lit(triangles, reflectance, transmittance, sources)
        return {"Erel": absorbed, "Erel_incident": incident, "lit_area": area}

    @state(include_inactive=True)
    def _absorbed_PAR(self, Erel, Erel_incident) -> dict:
        """PARa = Erel · PARi, at every step: between two lightings, with the last Erel (Wheat-BRIDGES')."""
        PARi = self.forcing("PARi")
        return {"PARa": Erel * PARi, "PAR_incident": Erel_incident * PARi}

    # ── the sky ──────────────────────────────────────────────────────────────

    def step_sources(self, over="substep") -> list:
        """
        The light sources of the step *over* (forcing_interval's: the step being lit) [(E, (vx, vy, vz))], of unit
        horizontal energy over it, consistent with PARi read as the step's mean: the step in panels of at most an
        hour (forcing_panels), each panel's sky and sun at its middle (sources), weighted by its share of the step's
        PAR (its duration times its mean PARi, over the step's); the diffuse skies summed into one, a sun per panel.
        An hourly step: the sun at the step's middle (not at its end, which is when PARi's mean is read up to); a
        daily step: the sun's course, 24 suns. Without sun, the diffuse sky alone, as at any instant.
        """
        start, end = self.forcing_interval(over)
        if not self.sun or not end > start:
            day, hour = self.calendar(end, default=WITHOUT_START)
            return self.sources(self.forcing("PARi", over=(start, end)), day, hour)
        mean = self.forcing("PARi", over=(start, end))
        diffuse, suns = 0., []
        for first, last in self.forcing_panels((start, end)):
            PARi = self.forcing("PARi", over=(first, last))
            share = (last - first) / (end - start) * (PARi / mean if mean > 0 else 1.)
            day, hour = self.calendar(0.5 * (first + last), default=WITHOUT_START)
            panel_diffuse, sun = self._diffuse_and_sun(PARi, day, hour, over=(first, last))
            diffuse += share * panel_diffuse
            suns += [(share * energy, vector) for energy, vector in sun]
        return self._sky(diffuse) + suns

    def _sky(self, diffuse) -> list:
        """The diffuse sky's sources, of total horizontal energy *diffuse*."""
        if self.sky == "sectors":
            return list(lights.Sky(self.azimuth_sectors, self.zenith_sectors, self.corrections,
                                   self.orientation).diffuse(diffuse, self.sky_type).lights())
        return list(lights.turtle_lights(self.sky[len("turtle"):], self.sky_type, diffuse, self.corrections,
                                         self.orientation))

    def sources(self, PARi, day, hour) -> list:
        """The light sources [(E, (vx, vy, vz))] of unit horizontal energy at one instant, *day* and *hour*: the
        diffuse sky, and the sun with its share when it is on and above the horizon (the forcing diffuse_fraction,
        else Spitters' of the global radiation PARi · global_per_PARi)."""
        diffuse, sun = self._diffuse_and_sun(PARi, day, hour)
        return self._sky(diffuse) + sun

    def _diffuse_and_sun(self, PARi, day, hour, over=None) -> tuple:
        """(the diffuse share, the sun's sources [(its share, its direction)]) at *day*, *hour*, the forcing
        diffuse_fraction read over *over* (the step being lit by default)."""
        if not self.sun:
            return 1., []
        sun = [(energy, vector) for energy, vector in
               lights.sun_light(1., day, hour, self.latitude, self.corrections, self.orientation) if vector[2] < 0]
        if not sun or PARi <= 0:
            return 1., []
        try:
            diffuse = self.forcing("diffuse_fraction", **({} if over is None else {"over": over}))
        except KeyError:
            diffuse = lights.diffuse_fraction(PARi * self.global_per_PARi, day, hour, self.latitude)
        return diffuse, [(energy * (1 - diffuse), vector) for energy, vector in sun]

    # ── the optics ───────────────────────────────────────────────────────────

    def element_labels(self) -> np.ndarray:
        """Each element's MTG label name ("LeafElement", ...; a union's parts')."""
        return self.data_structure.labels()

    @staticmethod
    def materials(reflectance, transmittance) -> list:
        """Each element's material, Caribu's: (ρ, τ) a leaf, (ρ,) opaque; None without optics."""
        return [None if np.isnan(rho) else ((rho,) if np.isnan(tau) else (rho, tau))
                for rho, tau in zip(reflectance.tolist(), transmittance.tolist())]

    # ── the lighting ─────────────────────────────────────────────────────────

    def repeated_pattern(self, vertices):
        """The infinite canopy's pattern: the component's (the scene's), else the triangles' extent."""
        if self.pattern is not None:
            return tuple(float(value) for value in self.pattern)
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
            raise ValueError(f"CaribuLight: elements labelled {missing} have triangles but no optics (their plant "
                             "model's PAR_reflectance, PAR_transmittance: an optical component, e.g. PlantOptics, "
                             "with their labels in its leaf_labels, stem_labels)")
        species, labels = api.species_and_labels([materials[owner] for owner in triangles.owner.tolist()])
        optics = api.optics(species, self.soil_reflectance)
        canopy = self.canopy(triangles.vertices, labels)
        if self.mode == "direct":
            result = canopy.direct(sources, optics)
        elif self.mode == "nested":
            result = canopy.nested(sources, optics, self.layers, float(triangles.vertices[..., 2].max()),
                                   self.sphere_diameter, self.precision, radiosity_method=self.radiosity_method)
        else:
            result = canopy.radiosity(sources, optics, radiosity_method=self.radiosity_method)
        values = {name: np.nan_to_num(np.asarray(result[name], dtype=float)) for name in ("area", "Eabs", "Ei")}
        if self.precision == "legacy":
            values = {name: np.array(api.printed(value)) for name, value in values.items()}
        weights = np.where(values["area"] > 0, values["area"], 0.)
        return (triangles.mean_per_owner(values["Eabs"], n_elements, weights),
                triangles.mean_per_owner(values["Ei"], n_elements, weights), triangles.per_owner(weights, n_elements))

    def screen_size(self, vertices, pattern) -> int:
        """The projections' screen side (pixels): SCREEN_SIZE without a pixel_size, else the scene's horizontal
        extent (the pattern's, or the triangles') over pixel_size, at least 2."""
        if self.pixel_size is None:
            return SCREEN_SIZE
        if pattern is not None:
            extent = max(pattern[2] - pattern[0], pattern[3] - pattern[1])
        else:
            flat = vertices.reshape(-1, 3)
            extent = float(max(np.ptp(flat[:, 0]), np.ptp(flat[:, 1]))) if flat.size else 0.
        return max(2, int(np.ceil(extent / float(self.pixel_size) - 1e-9)))       # within round-off

    def projection_threads(self) -> int:
        """The threads projecting the sources: threads, "auto" half the processors (at least 1)."""
        if self.threads == "auto":
            return max(1, (os.cpu_count() or 1) // 2)
        return int(self.threads)

    def canopy(self, vertices, labels) -> Canopy:
        """The Canopy of these triangles, kept (not checkpointed) while they, their labels and the settings it
        depends on are the same."""
        infinite = self.mode == "nested" or (self.infinite and self.mode == "direct")
        pattern = self.repeated_pattern(vertices) if infinite else None
        size, threads = self.screen_size(vertices, pattern), self.projection_threads()
        made_with = (pattern, size, threads, self.device, corrections_module.resolve(self.corrections))
        kept, kept_with = self.kept("canopy", (None, None))
        if (kept is None or kept_with != made_with or kept.vertices.shape != vertices.shape
                or not np.array_equal(kept.vertices, vertices) or not np.array_equal(kept.labels, labels)):
            kept = Canopy(vertices, labels, pattern=pattern, size=size, threads=threads, device=self.device,
                          corrections=self.corrections)
            self.keep("canopy", (kept, made_with), saved=False)
        return kept


class CaribuLightModel(EnvironmentModel):
    """The environment model of a metafspm Scene: CaribuLight on the populations' union (their elements' Erel, PARa,
    ... reach the plant models through the scene translator's links: identity ones by default for the inputs they
    declare by name, and their optics reach it). Its schedule: the lighting (CaribuLight's rate stage: its optics and
    Erel) every 4 h, covering the interval (metafspm's Cadence at its start), and when elements appeared or went;
    PARa every step (its state stage, "absorbed_light"). A subclass may light with a subclass of CaribuLight
    (light_component). The scenario's entries (the light's settings, "cadence": {"light": "1 h"}) are set by the
    Scene once it is built."""
    light_component = CaribuLight
    compartment = "Atmosphere"
    schedule = (Step("light", stages="rate", cadence=Cadence("4 h", at="start")),
                Step("light", stages="state", name="absorbed_light"))

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.data_structure = self.plant_data_structure()
        self.light = self.light_component(data_structure=self.data_structure, time_step=time_step,
                                          pattern=self.pattern)
        self.components = [self.light]
