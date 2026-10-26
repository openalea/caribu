"""
Caribu as a metafspm component: the light each element of a canopy absorbs, from its triangles.

    CaribuLight(data_structure=ds, lighting=Lighting(...))
                                            a FunctionalComponent on a population's DataStructure, or on a
                                            UnionDataStructure of several populations
    CaribuLightModel(populations, scene_xrange, scene_yrange, time_step)
                                            the environment model of a metafspm Scene running it

Its steps follow the physics:
    _optics           each element's reflectance and transmittance: an optical model's (the inputs), else its kind's
                      (its MTG label among leaf_labels, stem_labels: Wheat-BRIDGES' leaves ρ 0.10, τ 0.05, stems ρ 0.10)
    _relative_light   the canopy lit by a sky of unit energy (ds.triangles(), above the soil): Erel, each element's
                      absorbed light per unit area relative to the PAR above the canopy, its incident light, its area
    _absorbed_PAR     PARa = Erel · PARi, PAR_incident = Erel_incident · PARi, PARi the scene's forcing

The canopy is lit every *lighting_interval* (and when elements appear); in between CaribuLightModel runs only
_absorbed_PAR, PARa following PARi with the last Erel (Wheat-BRIDGES' Caribu every 4 hours). How the canopy is lit,
the same for every plant, is a `Lighting`: the light model (direct light, nested radiosity in the infinite canopy,
pure SAIL, radiosity in the scene alone), the sky and the sun, the site, the soil, the numerics (precision,
corrections, GPU, the linear solver). The optics are plant parameters: they may differ between plants. The elements'
parts below the soil are cut off: the soil is opaque (Caribu's infinite canopy would lift the scene to put its lowest
point at 0, lighting them). The geometry is kept between lightings (a Canopy: the sky's projections reused) while
the triangles, their optics and the lighting are the same.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.cross import UnionDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate, state

from . import api, lights
from . import corrections as corrections_module
from .canopy import Canopy

ELEMENT = dict(scale=scales.SubOrgan)


@dataclass(frozen=True)
class Lighting:
    """How CaribuLight lights the canopy: settings of the scene, not of the plants (one value for all)."""
    # the light model: "direct" (first order, Wheat-BRIDGES'), "nested" (the scattering in the infinite canopy:
    # radiosity within a sphere of *sphere_diameter* around each triangle, a turbid medium of *layers* beyond; a null
    # diameter for pure SAIL) or "radiosity" (every order of scattering in the scene alone, not infinite)
    mode: str = "direct"
    infinite: bool = True                   # direct light: the pattern repeated (nested is always infinite)
    sphere_diameter: float = 0.5            # m (the scene's unit)
    layers: int = 10
    # the sky: "sectors" (azimuth_sectors × zenith_sectors) or a turtle ("turtle46", "turtle16", ...), "soc" standard
    # overcast or "uoc" uniform (Wheat-BRIDGES: 4 × 5 sectors, soc)
    sky: str = "sectors"
    sky_type: str = "soc"
    azimuth_sectors: int = 4
    zenith_sectors: int = 5
    # the sun with the diffuse sky, its share 1 − the diffuse fraction (the forcing diffuse_fraction, else Spitters' of
    # the global radiation PARi · global_per_PARi: 1 / (0.48 × 4.57) for PARi in µmol m-2 s-1, McCree)
    sun: bool = False
    global_per_PARi: float = 1 / (0.48 * 4.57)
    latitude: float = 48.85                 # °
    orientation: float = 0.                 # °, clockwise, from the scene's X+ to North
    start_day_of_year: float = 1.           # the day and the universal time hour at the scene's time 0
    start_hour: float = 0.
    soil_level: float = 0.                  # m: the elements' parts below it (sown shoots, roots) left out
    soil_reflectance: float = 0.15          # the soil's in the PAR (scattering)
    lighting_interval: float = 4 * 3600.    # s between two lightings (Wheat-BRIDGES: 4 hours)
    # numerics: the projections' screen (pixels), "full" the computed values or "legacy" the legacy's printed digits,
    # the legacy's behaviours corrected ("recommended", names of openalea.caribu.corrections, or () with "legacy"),
    # the near field's device ("cpu", "cuda"), the radiosity system's solver ("mgcr", "gmres")
    screen_size: int = 1536
    precision: str = "full"
    corrections: object = "recommended"
    device: str = "cpu"
    linear_solver: str = "mgcr"

    def __post_init__(self):
        if self.mode not in ("direct", "nested", "radiosity"):
            raise ValueError(f"Lighting.mode: 'direct', 'nested' or 'radiosity', got '{self.mode}'")
        if not (self.sky == "sectors" or self.sky.startswith("turtle")):
            raise ValueError(f"Lighting.sky: 'sectors' or 'turtle46', 'turtle16', ..., got '{self.sky}'")
        corrections_module.resolve(self.corrections, self.precision)


@dataclass
class CaribuLight(FunctionalComponent):
    """The PAR each element absorbs, by Caribu on the canopy's triangles: its optics (_optics), its light relative to
    the incident PAR when the canopy is lit (_relative_light), its PAR at every step (_absorbed_PAR)."""

    PAR_reflectance: float = input_variable(
        unit="adim", description="The element's reflectance in the PAR, from an optical model (NaN: its kind's)",
        by="an optical model", initialize=np.nan, **ELEMENT)
    PAR_transmittance: float = input_variable(
        unit="adim", description="The element's transmittance in the PAR, from an optical model (NaN: opaque)",
        by="an optical model", initialize=np.nan, **ELEMENT)

    reflectance: float = state_variable(
        unit="adim", description="The element's reflectance in the light model", state_variable_type="intensive",
        initialize=np.nan, **ELEMENT)
    transmittance: float = state_variable(
        unit="adim", description="The element's transmittance in the light model (NaN: opaque, one face)",
        state_variable_type="intensive", initialize=np.nan, **ELEMENT)
    Erel: float = state_variable(
        unit="adim", description="The PAR the element absorbs per unit area, relative to the PAR above the canopy",
        state_variable_type="intensive", initialize=0., **ELEMENT)
    Erel_incident: float = state_variable(
        unit="adim", description="The PAR reaching the element per unit area (its two faces), relative to the PAR "
                                 "above the canopy", state_variable_type="intensive", initialize=0., **ELEMENT)
    lit_area: float = state_variable(
        unit="m2", unit_comment="the scene's unit squared", description="The element's area in the light model (its "
        "triangles' above the soil, within the pattern)", state_variable_type="extensive", initialize=0., **ELEMENT)
    PARa: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR the element absorbs per unit area",
        state_variable_type="intensive", initialize=0., **ELEMENT)
    PAR_incident: float = state_variable(
        unit="µmol m-2 s-1", unit_comment="PARi's unit", description="The PAR reaching the element per unit area",
        state_variable_type="intensive", initialize=0., **ELEMENT)

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
    lighting: Lighting = parameter(unit="", description="How the canopy is lit: the light model, the sky and the "
                                   "sun, the site, the numerics (Lighting)", by="Caribu", default=Lighting())

    scene_xrange: float = None          # the pattern (0, 0, scene_xrange, scene_yrange): the scene's
    scene_yrange: float = None

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
    def _relative_light(self, reflectance, transmittance) -> dict:
        """The canopy lit by a sky of unit energy: each element's absorbed and incident light per unit area (the
        area-weighted means of its triangles' Eabs and Ei), relative to the incident PAR, and its area."""
        lighting = self.lighting
        time = self.forcing_time()
        day = lighting.start_day_of_year + (lighting.start_hour * 3600. + time) // 86400.
        hour = (lighting.start_hour + time / 3600.) % 24.
        sources = self.sources(self.forcing("PARi"), day, hour)
        triangles = above_soil(self.data_structure.triangles(), lighting.soil_level)
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
        lighting = self.lighting
        diffuse = 1.
        sun = []
        if lighting.sun:
            sun = [(energy, vector) for energy, vector in
                   lights.sun_light(1., day, hour, lighting.latitude, lighting.corrections, lighting.orientation)
                   if vector[2] < 0]
            if sun and PARi > 0:
                try:
                    diffuse = self.forcing("diffuse_fraction")
                except KeyError:
                    diffuse = lights.diffuse_fraction(PARi * lighting.global_per_PARi, day, hour, lighting.latitude)
            if not sun or PARi <= 0:
                sun, diffuse = [], 1.
        if lighting.sky == "sectors":
            sky = lights.Sky(lighting.azimuth_sectors, lighting.zenith_sectors, lighting.corrections,
                             lighting.orientation).diffuse(diffuse, lighting.sky_type).lights()
        else:
            sky = lights.turtle_lights(lighting.sky[len("turtle"):], lighting.sky_type, diffuse, lighting.corrections,
                                       lighting.orientation)
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
        if self.scene_xrange is not None and self.scene_yrange is not None:
            return 0., 0., float(self.scene_xrange), float(self.scene_yrange)
        flat = vertices.reshape(-1, 3)
        return (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))

    def lit(self, triangles, reflectance, transmittance, sources):
        """(absorbed, incident, area) per element of the canopy of *triangles*: the area-weighted means of its
        triangles' Eabs and Ei (0 for an element without lit area), the triangles' area. With precision "legacy",
        the triangles' values rounded as canestrad prints them (6 decimals), as CaribuScene aggregates them: in
        metres, small triangles' areas lose most of their digits (Wheat-BRIDGES' Erel)."""
        lighting = self.lighting
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
        optics = api.optics(species, lighting.soil_reflectance)
        canopy = self.canopy(triangles.vertices, labels)
        if lighting.mode == "direct":
            result = canopy.direct(sources, optics)
        elif lighting.mode == "nested":
            result = canopy.nested(sources, optics, lighting.layers, float(triangles.vertices[..., 2].max()),
                                   lighting.sphere_diameter, lighting.precision, linear_solver=lighting.linear_solver)
        else:
            result = canopy.radiosity(sources, optics, linear_solver=lighting.linear_solver)
        values = {name: np.nan_to_num(np.asarray(result[name], dtype=float)) for name in ("area", "Eabs", "Ei")}
        if lighting.precision == "legacy":
            values = {name: np.array(api.printed(value)) for name, value in values.items()}
        weights = np.where(values["area"] > 0, values["area"], 0.)
        total = triangles.per_owner(weights, n_elements)
        means = []
        for name in ("Eabs", "Ei"):
            summed = triangles.per_owner(values[name], n_elements, weights=weights)
            means.append(np.divide(summed, total, out=np.zeros(n_elements), where=total > 0))
        return means[0], means[1], total

    def canopy(self, vertices, labels) -> Canopy:
        """The Canopy of these triangles, kept while they, their labels and the lighting are the same."""
        lighting = self.lighting
        kept = self.__dict__.get("_canopy")
        infinite = lighting.mode == "nested" or (lighting.infinite and lighting.mode == "direct")
        pattern = self.pattern(vertices) if infinite else None
        if (kept is None or kept.pattern != pattern or self.__dict__.get("_canopy_lighting") != lighting
                or kept.vertices.shape != vertices.shape or not np.array_equal(kept.vertices, vertices)
                or not np.array_equal(kept.labels, labels)):
            kept = Canopy(vertices, labels, pattern=pattern, size=lighting.screen_size, device=lighting.device,
                          corrections=lighting.corrections)
            self.__dict__["_canopy"], self.__dict__["_canopy_lighting"] = kept, lighting
        return kept


def above_soil(triangles, level=0.):
    """The parts of *triangles* (metafspm's Triangles) above the soil at height *level*: a triangle across it cut at
    its plane (one triangle or two, the same owner), one below it left out. The soil is opaque: a sown shoot's buried
    part receives no light (Caribu's infinite canopy would otherwise lift the scene to put its lowest point at 0)."""
    vertices = np.asarray(triangles.vertices, dtype=float)
    below = vertices[:, :, 2] < level
    count = below.sum(axis=1)
    kept_vertices, kept_owners = [vertices[count == 0]], [triangles.owner[count == 0]]
    for t in np.flatnonzero((count == 1) | (count == 2)):
        points, buried = vertices[t], below[t]
        polygon = []
        for i in range(3):  # Sutherland–Hodgman against z >= level
            current, following = points[i], points[(i + 1) % 3]
            if not buried[i]:
                polygon.append(current)
            if buried[i] != buried[(i + 1) % 3]:
                fraction = (level - current[2]) / (following[2] - current[2])
                crossing = current + fraction * (following - current)
                crossing[2] = level
                polygon.append(crossing)
        for k in range(1, len(polygon) - 1):
            kept_vertices.append(np.array([[polygon[0], polygon[k], polygon[k + 1]]]))
            kept_owners.append(triangles.owner[t:t + 1])
    return type(triangles)(np.concatenate(kept_vertices), np.concatenate(kept_owners))


class CaribuLightModel:
    """The environment model of a metafspm Scene: CaribuLight on the population's DataStructure, or on the union of
    the populations' (their elements' Erel, PARa, ... then reach them through the scene translator's links). It runs
    every scene step: the canopy lit when due (every lighting_interval, or when elements appeared), else PARa only
    (the steps of the row "state")."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **parameters):
        populations = list(populations)
        self.data_structure = populations[0] if len(populations) == 1 else UnionDataStructure(populations)
        self.light = CaribuLight(data_structure=self.data_structure, time_step=time_step, **parameters)
        self.light.scene_xrange, self.light.scene_yrange = scene_xrange, scene_yrange
        self.components = [self.light]
        self.lit_at, self.lit_elements = None, None

    def due(self) -> bool:
        """Whether the canopy is to be lit at this step."""
        if self.lit_at is None or self.lit_elements != self.data_structure.n_nodes():
            return True
        return self.light.forcing_time() - self.lit_at >= self.light.lighting.lighting_interval - 1e-9

    def run(self):
        if self.due():
            time = self.light.forcing_time()            # the call advances the component's clock
            self.light()
            self.lit_at, self.lit_elements = time, self.data_structure.n_nodes()
        else:
            self.light.run_stage("state")
