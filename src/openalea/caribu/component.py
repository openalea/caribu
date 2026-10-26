"""
Caribu as a metafspm component: the light each element of a canopy absorbs, from its triangles.

    CaribuLight(data_structure=ds)          a FunctionalComponent on a population's DataStructure, or on a
                                            UnionDataStructure of several populations
    CaribuLightModel(populations, scene_xrange, scene_yrange, time_step)
                                            the environment model of a metafspm Scene running it

Each run reads the elements' triangles (``ds.triangles()``, from the structural component drawing them), their
optics, the incident PAR (the forcing ``PARi``) and lights the canopy, infinite by default (the pattern repeated over
the scene's ranges, as Wheat-BRIDGES): under a sky of unit energy (the diffuse sky, plus the sun when *sun* is on),
each element's absorbed light per unit area relative to the PAR above the canopy, Erel, and PARa = Erel · PARi. The
canopy is lit every *lighting_interval*; in between, PARa follows PARi with the last Erel (Wheat-BRIDGES' Caribu every
4 hours). The light models are Caribu's (`mode`): "direct" (first order, Wheat-BRIDGES'), "nested" (the scattering in
the infinite canopy: radiosity within a sphere of *sphere_diameter* around each triangle, a turbid medium of
*layers* beyond; a null diameter for pure SAIL), "radiosity" (every order in the scene alone, not infinite).

Optics: an element's PAR_reflectance and PAR_transmittance when a coupled model writes them (a NaN transmittance: an
opaque element, a stem), else its MTG label's in *optical_properties* (Wheat-BRIDGES' (0.10, 0.05) leaves, (0.10,)
stems). The geometry is kept between runs (a Canopy: the sky's projections reused) while the triangles and their
optics are the same.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.cross import UnionDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate

from . import api, lights
from . import corrections as corrections_module
from .canopy import Canopy
from .labels import encode_labels

ELEMENT = dict(scale=scales.SubOrgan)


@dataclass
class CaribuLight(FunctionalComponent):
    """The PAR each element absorbs, by Caribu on the canopy's triangles."""

    PAR_reflectance: float = input_variable(
        unit="adim", description="The element's reflectance in the PAR (NaN: its label's in optical_properties)",
        by="an optical model", initialize=np.nan, **ELEMENT)
    PAR_transmittance: float = input_variable(
        unit="adim", description="The element's transmittance in the PAR (NaN: opaque, a stem)",
        by="an optical model", initialize=np.nan, **ELEMENT)

    Erel: float = state_variable(
        unit="adim", description="The PAR the element absorbs per unit area, relative to the PAR above the canopy",
        state_variable_type="intensive", initialize=0., **ELEMENT)
    PARa: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR the element absorbs per unit area",
        state_variable_type="intensive", initialize=0., **ELEMENT)
    Erel_incident: float = state_variable(
        unit="adim", description="The PAR reaching the element per unit area (its two faces), relative to the PAR "
                                 "above the canopy", state_variable_type="intensive", initialize=0., **ELEMENT)
    PAR_incident: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR reaching the element per unit area",
        state_variable_type="intensive", initialize=0., **ELEMENT)
    lit_area: float = state_variable(
        unit="m2", unit_comment="the scene's unit squared", description="The element's area in the light model (its "
        "triangles' within the pattern)", state_variable_type="extensive", initialize=0., **ELEMENT)

    optical_properties: tuple = parameter(
        unit="adim", description="(label, material) pairs for the elements without optics of their own: (ρ, τ) a "
        "leaf, (ρ,) opaque", by="Wheat-BRIDGES", default=(("LeafElement", (0.10, 0.05)), ("StemElement", (0.10,))))
    mode: str = parameter(unit="", description="'direct', 'nested' (scattering in the infinite canopy) or "
                                               "'radiosity' (scattering in the scene alone)", by="Caribu",
                          default="direct")
    infinite: bool = parameter(unit="", description="The pattern repeated (direct light); nested is always infinite",
                               by="Wheat-BRIDGES", default=True)
    sky: str = parameter(unit="", description="'sectors' (azimuth_sectors x zenith_sectors) or a turtle sky "
                                              "('turtle46', 'turtle16', 'turtle6', 'turtle1')", by="Wheat-BRIDGES",
                         default="sectors")
    sky_type: str = parameter(unit="", description="'soc' standard overcast or 'uoc' uniform", by="Wheat-BRIDGES",
                              default="soc")
    azimuth_sectors: int = parameter(unit="", description="The sector sky's azimuths", by="Wheat-BRIDGES", default=4)
    zenith_sectors: int = parameter(unit="", description="The sector sky's zeniths", by="Wheat-BRIDGES", default=5)
    sun: bool = parameter(unit="", description="The sun with the diffuse sky (its share 1 - the diffuse fraction)",
                          by="Caribu", default=False)
    latitude: float = parameter(unit="°", description="The scene's latitude", by="Wheat-BRIDGES", default=48.85)
    start_day_of_year: float = parameter(unit="day", description="The day of the year at the scene's time 0",
                                         by="the scenario", default=1.)
    start_hour: float = parameter(unit="h", description="The universal time hour at the scene's time 0",
                                  by="the scenario", default=0.)
    global_per_PARi: float = parameter(
        unit="W m-2 per PARi unit", description="The global radiation per unit of PARi, for the diffuse fraction "
        "without its forcing: 1 / (0.48 x 4.57), PARi in µmol m-2 s-1 (McCree)", by="Caribu", default=1 / (0.48 * 4.57))
    lighting_interval: float = parameter(unit="s", description="Time between two lightings of the canopy",
                                         by="Wheat-BRIDGES", default=4 * 3600.)
    sphere_diameter: float = parameter(unit="m", unit_comment="the scene's unit", description="Nested radiosity's "
                                       "sphere (0: pure SAIL)", by="Caribu", default=0.5)
    layers: int = parameter(unit="", description="Nested radiosity's far field layers", by="Caribu", default=10)
    soil_reflectance: float = parameter(unit="adim", description="The soil's reflectance in the PAR (scattering)",
                                        by="Caribu", default=0.15)
    screen_size: int = parameter(unit="pixel", description="The projections' screen side", by="Caribu", default=1536)
    precision: str = parameter(unit="", description="'full' the computed values, 'legacy' the legacy's printed "
                                                    "digits", by="Caribu", default="full")
    corrections: str = parameter(unit="", description="The legacy's behaviours corrected: 'recommended' (all), a "
                                 "tuple of names (openalea.caribu.corrections), or () for none (with precision "
                                 "'legacy')", by="Caribu", default="recommended")
    device: str = parameter(unit="", description="'cpu' or 'cuda' (radiosity's near field)", by="Caribu",
                            default="cpu")

    scene_xrange: float = None          # the pattern (0, 0, scene_xrange, scene_yrange), the scene's
    scene_yrange: float = None

    @rate(include_inactive=True)
    def _light(self, PAR_reflectance, PAR_transmittance, azimuth_sectors, zenith_sectors, latitude,
               start_day_of_year, start_hour, global_per_PARi, lighting_interval, sphere_diameter, layers,
               soil_reflectance, screen_size) -> dict:
        """Erel when the canopy is due to be lit, PARa = Erel · PARi at every step."""
        PARi = self.forcing("PARi")
        time = self.forcing_time()
        last = self.__dict__.get("_lit_at")
        relative = self.__dict__.get("_relative")
        n_elements = PAR_reflectance.size
        first = lambda values: float(np.asarray(values).flat[0])
        if relative is None or relative[0].size != n_elements or time - last >= first(lighting_interval) - 1e-9:
            day = first(start_day_of_year) + (first(start_hour) * 3600. + time) // 86400.
            hour = (first(start_hour) + time / 3600.) % 24.
            sources = self.sources(PARi, day, hour, first(latitude), int(first(azimuth_sectors)),
                                   int(first(zenith_sectors)), first(global_per_PARi))
            relative = self.lit(self.data_structure.triangles(), PAR_reflectance, PAR_transmittance, sources,
                                first(sphere_diameter),
                                int(first(layers)), first(soil_reflectance), int(first(screen_size)))
            self.__dict__["_lit_at"], self.__dict__["_relative"] = time, relative
        absorbed, incident, area = relative
        return {"Erel": absorbed, "PARa": absorbed * PARi, "Erel_incident": incident, "PAR_incident": incident * PARi,
                "lit_area": area}

    # ── the sky ──────────────────────────────────────────────────────────────

    def sources(self, PARi, day, hour, latitude, azimuth_sectors, zenith_sectors, global_per_PARi) -> list:
        """The light sources [(E, (vx, vy, vz))] of unit horizontal energy: the diffuse sky, and the sun with its
        share when *sun* is on and above the horizon (the forcing diffuse_fraction, else Spitters' of the global
        radiation PARi · global_per_PARi)."""
        diffuse = 1.
        sun = []
        if self.sun:
            sun = [(energy, vector) for energy, vector in lights.sun_light(1., day, hour, latitude) if vector[2] < 0]
            if sun and PARi > 0:
                try:
                    diffuse = self.forcing("diffuse_fraction")
                except KeyError:
                    diffuse = lights.diffuse_fraction(PARi * global_per_PARi, day, hour, latitude)
            if not sun or PARi <= 0:
                sun, diffuse = [], 1.
        if self.sky == "sectors":
            sky = lights.Sky(azimuth_sectors, zenith_sectors).diffuse(diffuse, self.sky_type).lights()
        elif self.sky.startswith("turtle"):
            sky = lights.turtle_lights(self.sky[len("turtle"):], self.sky_type, diffuse)
        else:
            raise ValueError(f"CaribuLight.sky: 'sectors' or 'turtle46', 'turtle16', ..., got '{self.sky}'")
        return list(sky) + [(energy * (1 - diffuse), vector) for energy, vector in sun]

    # ── the optics ───────────────────────────────────────────────────────────

    def element_labels(self) -> np.ndarray:
        """Each element's MTG label name (its part's MTG for a union: "LeafElement", ...), read once per topology."""
        def build():
            ds = self.data_structure
            parts = ds.parts if isinstance(ds, UnionDataStructure) else [ds]
            labels = []
            for part in parts:
                g = part.structure
                names = g.labels.translator             # the MPG's label codes → names
                labels += [names.get(int(code), code) if isinstance(code, (int, float, np.number)) else code
                           for code in (g.label(v) for v in part.entity_ids("node").tolist())]
            return np.array(labels, dtype=object)
        return self.cached("element_labels", build)

    def materials(self, reflectance, transmittance) -> list:
        """Each element's material, Caribu's: (ρ, τ) a leaf, (ρ,) opaque; None without optics."""
        table = dict(self.optical_properties)
        materials = []
        for label, rho, tau in zip(self.element_labels().tolist(), reflectance.tolist(), transmittance.tolist()):
            if np.isnan(rho):
                material = table.get(label)
                materials.append(None if material is None else tuple(map(float, material)))
            else:
                materials.append((rho,) if np.isnan(tau) else (rho, tau))
        return materials

    # ── the lighting ─────────────────────────────────────────────────────────

    def pattern(self, vertices):
        """The infinite canopy's pattern: the scene's ranges, else the triangles' extent."""
        if self.scene_xrange is not None and self.scene_yrange is not None:
            return 0., 0., float(self.scene_xrange), float(self.scene_yrange)
        flat = vertices.reshape(-1, 3)
        return (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))

    def lit(self, triangles, reflectance, transmittance, sources, sphere_diameter, layers, soil_reflectance,
            screen_size):
        """(absorbed, incident, area) per element of the canopy of *triangles*: the area-weighted means of its
        triangles' Eabs and Ei (0 for an element without lit area), the triangles' area. With *precision* "legacy",
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
                             "PAR_reflectance, PAR_transmittance or the parameter optical_properties)")
        triangle_materials = [materials[owner] for owner in triangles.owner.tolist()]
        species, labels = api.species_and_labels(triangle_materials)
        optics = api.optics(species, soil_reflectance)
        if self.mode not in ("direct", "nested", "radiosity"):
            raise ValueError(f"CaribuLight.mode: 'direct', 'nested' or 'radiosity', got '{self.mode}'")
        canopy = self.canopy(triangles.vertices, labels, screen_size)
        if self.mode == "direct":
            result = canopy.direct(sources, optics)
        elif self.mode == "nested":
            height = float(triangles.vertices[..., 2].max())
            result = canopy.nested(sources, optics, layers, height, sphere_diameter, self.precision)
        else:
            result = canopy.radiosity(sources, optics)
        values = {name: np.nan_to_num(np.asarray(result[name], dtype=float)) for name in ("area", "Eabs", "Ei")}
        if self.precision == "legacy":
            values = {name: np.array(api.printed(value)) for name, value in values.items()}
        weights = np.where(values["area"] > 0, values["area"], 0.)
        total = triangles.per_owner(weights, n_elements)
        means = []
        for name in ("Eabs", "Ei"):
            summed = triangles.per_owner(values[name], n_elements, weights=weights)
            means.append(np.divide(summed, total, out=np.zeros(n_elements), where=total > 0))
        return means[0], means[1], total

    def canopy(self, vertices, labels, screen_size) -> Canopy:
        """The Canopy of these triangles, kept while they and their labels are the same."""
        kept = self.__dict__.get("_canopy")
        infinite = self.mode == "nested" or (self.infinite and self.mode == "direct")
        pattern = self.pattern(vertices) if infinite else None
        if (kept is None or kept.size != screen_size or kept.pattern != pattern
                or kept.corrections != corrections_module.resolve(self.corrections, self.precision)
                or kept.vertices.shape != vertices.shape or not np.array_equal(kept.vertices, vertices)
                or not np.array_equal(kept.labels, labels)):
            kept = Canopy(vertices, labels, pattern=pattern, size=screen_size, device=self.device,
                          corrections=self.corrections)
            self.__dict__["_canopy"] = kept
        return kept


class CaribuLightModel:
    """The environment model of a metafspm Scene: CaribuLight on the population's DataStructure, or on the union of
    the populations' (their elements' Erel, PARa, ... then reach them through the scene translator's links). It runs
    every scene step: PARa follows PARi, the canopy lit every *lighting_interval*."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **parameters):
        populations = list(populations)
        self.data_structure = populations[0] if len(populations) == 1 else UnionDataStructure(populations)
        self.light = CaribuLight(data_structure=self.data_structure, time_step=time_step, **parameters)
        self.light.scene_xrange, self.light.scene_yrange = scene_xrange, scene_yrange
        self.components = [self.light]

    def run(self):
        self.light()
