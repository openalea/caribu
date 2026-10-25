"""
The legacy's entry points over the port: `raycasting` and `x_raycasting` (caribu.py's) on lists of triangles and
materials, and `CaribuScene` (its scenes by primitive, bands, soil, units, aggregation). Their outputs are the
legacy's: per triangle `index`, `label`, `area`, `Eabs`, `Ei`, `Ei_sup`, `Ei_inf` (and `sensors`).

A material is a 1-tuple (ρ, opaque), a 2-tuple (ρ, τ, a symmetric leaf) or a 4-tuple (ρ_sup, τ_sup, ρ_inf, τ_inf).
Light sources are (E, (vx, vy, vz)): E the irradiance on a horizontal plane, (vx, vy, vz) the beam's direction.

precision="legacy" (the default, until the corrections step) gives the legacy's numbers exactly: canestrad prints
its results with 6 decimals and caribu reads them back, so the values are rounded so; `Ei` is then the legacy's
Eabs / absorptance. precision="full" keeps the computed values (float32 budgets).
"""
import os

import numpy as np

from . import direct, formats, periodic
from .labels import encode_labels

GREEN_LEAF_PAR = (0.06, 0.07)
GREEN_STEM_PAR = (0.13,)
SOIL_REFLECTANCE_PAR = 0.2
DEFAULT_LIGHT = (1, (0, 0, -1))


# --- Materials ---

def absorptance(material) -> float:
    """1 − ρ − τ of a material (the faces' mean for an asymmetric leaf)."""
    return 1 - sum(material) if len(material) <= 2 else 1 - sum(material) / 2.


def species_and_labels(materials):
    """({species: material}, labels): the legacy's optical species, numbered from 1 over the distinct materials (in
    the order of Python's set, as caribu.py), and each triangle's label (plant 1, a leaf when translucent)."""
    species = {number + 1: material for number, material in enumerate(list(set(materials)))}
    number_of = {material: number for number, material in species.items()}
    numbers = np.array([number_of[material] for material in materials], dtype=np.int64)
    leaf = np.array([len(material) > 1 for material in materials], dtype=np.int64)
    return species, encode_labels(numbers, leaf=leaf, plant=1)


def optics(species, soil_reflectance=-1) -> tuple:
    """The port's optics (soil reflectance, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}) of
    {species: material}, as canestrad reads them from their `.opt` text."""
    _, soil, table = formats.parse_opt(formats.opt_string(species, soil_reflectance))
    return soil, table


def get_incident(absorbed, materials) -> list:
    """The legacy's incident light Eabs / absorptance (Eabs where the absorptance is 0); for an asymmetric leaf, a
    mean estimate."""
    return [float(value) / alpha if alpha != 0 else value
            for value, alpha in zip(absorbed, (absorptance(material) for material in materials))]


# --- Precision ---

def _printed(values, layout="%f") -> list:
    """Values as canestrad prints them and caribu reads them back."""
    return [float(layout % value) for value in values]


def _outputs(labels, result, precision) -> dict:
    """The legacy's output columns of a direct.run result."""
    accepted = result["accepted"]
    label_strings = []
    for label in labels:
        text = "%.0f" % label
        label_strings.append((12 - len(text)) * "0" + text if len(text) < 11 else text)
    out = {"index": [float(number) for number in range(len(labels))], "label": label_strings}
    for name in ("area", "Eabs", "Ei_sup", "Ei_inf"):
        values = np.asarray(result[name], dtype=np.float64)
        if precision == "legacy":
            values = np.where(accepted, values, np.nan if name != "area" else 0.)
            out[name] = [value if not accepted[number] else printed for number, (value, printed) in
                         enumerate(zip(values.tolist(), _printed(values)))]
        else:
            out[name] = values.tolist()
    return out


def _sensor_outputs(sensors, precision) -> dict:
    """The legacy's sensor columns (sensor_id, Ei0 the first order, Ei the total, area)."""
    irradiance, area = sensors["Ei"], sensors["area"]
    identifiers = [float(number + 1) for number in range(len(area))]
    if precision == "legacy":
        first = _printed(irradiance, "%.10f")
        return {"sensor_id": identifiers, "Ei0": first, "Ei": list(first), "area": _printed(area, "%.6f")}
    return {"sensor_id": identifiers, "Ei0": list(map(float, irradiance)), "Ei": list(map(float, irradiance)),
            "area": list(map(float, area))}


# --- Direct light ---

def raycasting(triangles, materials, lights=(DEFAULT_LIGHT,), domain=None, screen_size=1536, sensors=None,
               debug=False, precision="legacy", workers=None) -> dict:
    """Direct light on triangles of *materials* (caribu.raycasting): per triangle index, label, area, Eabs, Ei,
    Ei_sup, Ei_inf, and "sensors" when *sensors* are given. *domain* (xmin, ymin, xmax, ymax): an infinite canopy
    repeating it. *debug*: the legacy's (it kept its files), without effect here."""
    if len(triangles) != len(materials):
        raise ValueError("The number of triangles and materials should match")
    species, labels = species_and_labels(materials)
    sensor_points = None if sensors is None else _as_array(sensors)
    result = direct.run(_as_array(triangles), labels, list(lights), optics(species), screen_size,
                        sensor_points, workers, pattern=domain)
    columns = _outputs(labels, result, precision)
    columns["Ei"] = get_incident(columns["Eabs"], materials)
    out = {name: columns[name] for name in ("index", "label", "area", "Eabs", "Ei", "Ei_sup", "Ei_inf")}
    if sensors is not None:
        out["sensors"] = _sensor_outputs(result["sensors"], precision)
    return out


def x_raycasting(triangles, x_materials, lights=(DEFAULT_LIGHT,), domain=None, screen_size=1536, sensors=None,
                 debug=False, precision="legacy", workers=None) -> dict:
    """raycasting for several bands, {band: materials}: the projections are the same for all, so the last band is
    computed and the others' Eabs follow from its Ei (their absorptance times it), as caribu.x_raycasting."""
    x_materials = dict(x_materials)
    band, materials = x_materials.popitem()
    out = raycasting(triangles, materials, lights, domain, screen_size, sensors, debug, precision, workers)
    x_out = {band: out}
    for other in x_materials:
        x_out[other] = {name: values for name, values in out.items() if name != "Eabs"}
        x_out[other]["Eabs"] = [absorptance(material) * incident
                                for material, incident in zip(x_materials[other], out["Ei"])]
    return x_out


def _as_array(triangles) -> np.ndarray:
    """(n, 3, 3) of triangles given as sequences of points."""
    return np.array([[[float(value) for value in point] for point in triangle] for triangle in triangles],
                    dtype=np.float64).reshape(-1, 3, 3)


# --- Aggregation ---

def _aggregate(values, groups, keys, how):
    """{group: aggregate} in sorted group order: "list" the values, "sum" their sum, "mean" their area-weighted
    mean (triangles of area 0 left out; 0 for a group of no area), as CaribuScene's _agregate and _wsum (Python's
    sum, compensated since Python 3.12)."""
    lists = {key: [] for key in keys}
    for value, group in zip(values if how != "mean" else zip(*values), groups):
        lists[group].append(value)
    if how == "list":
        return lists
    if how == "sum":
        return {key: sum(group) for key, group in lists.items()}
    means = {}
    for key, group in lists.items():
        total = sum(area for _, area in group)
        means[key] = 0 if total == 0 else sum(energy * area if area > 0 else 0 for energy, area in group) / total
    return means


def domain_mesh(domain, z=0.):
    """Two triangles covering the domain (xmin, ymin, xmax, ymax) at height z (the soil)."""
    xmin, ymin, xmax, ymax = domain
    a, b, c, d = (xmin, ymin, z), (xmax, ymin, z), (xmin, ymax, z), (xmax, ymax, z)
    return [(a, b, c), (b, d, c)]


# --- The scene ---

class CaribuScene:
    """Light on a scene of primitives, each a list of triangles: the legacy's CaribuScene over the port.

        scene = CaribuScene(scene={pid: [triangles]}, light=[(E, direction)], pattern=(xmin, ymin, xmax, ymax),
                            opt={band: {pid: material}}, soil_reflectance={band: rho}, scene_unit="cm")
        raw, aggregated = scene.run(direct=True, infinite=True)

    raw[band][name] {pid: [per triangle values]}, aggregated[band][name] {pid: value} (the area summed, energies
    area-weighted), energies per m² and areas in m² whatever the scene unit.
    """

    default_material = (0.06, 0.07)
    default_soil_reflectance = 0.15
    default_light = (1, (0, 0, -1))
    default_band = "default_band"
    units = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1, "dam": 10, "hm": 100, "km": 1000}

    def __init__(self, scene=None, light=None, pattern=None, opt=None, soil_reflectance=None, soil_mesh=None,
                 z_soil=None, scene_unit="m", precision="legacy"):
        """*scene* {pid: [triangles]}, a `.can` path (its labels as primitives), a PlantGL scene (its shapes) or an
        MTG (its "geometry" property); *light* [(E, (vx, vy, vz))] or a
        `.light` path; *pattern* (xmin, ymin, xmax, ymax), ((xmin, ymin), (xmax, ymax)) or a `.8` path; *opt*
        {band: {pid: material}}, {band: material} for all, or a list of `.opt` paths (with a `.can` scene, the band
        named by the file); *soil_mesh* adds a soil of two triangles over the pattern at *z_soil* (by default the
        scene's lowest point); *scene_unit* the unit of the coordinates and the pattern."""
        if scene_unit not in self.units:
            raise ValueError("unrecognised scene unit: " + scene_unit)
        self.conv_unit = self.units[scene_unit]
        self.precision = precision
        self.scene = None
        if scene is not None:
            if isinstance(scene, dict):
                self.scene = {key: [tuple(tuple(map(float, point)) for point in triangle) for triangle in triangles]
                              for key, triangles in scene.items()}
            elif isinstance(scene, (str, os.PathLike)):
                self.scene = formats.read_can(scene)
            elif all(hasattr(scene, name) for name in ("_scale", "_complex", "_components", "_parent")):
                self.scene = formats.mtg_scene(scene)
            elif type(scene).__name__ == "Scene" and hasattr(scene, "todict"):
                self.scene = formats.plantgl_scene(scene)
            else:
                raise ValueError("Unrecognised scene format")
        self.light = [self.default_light]
        if light is not None:
            self.setLight(light)
        self.pattern = None
        if pattern is not None:
            if isinstance(pattern, tuple):
                if len(pattern) == 2:
                    pattern = sum(pattern, ())
                if len(pattern) != 4:
                    raise ValueError("Unrecognised pattern format")
                self.pattern = pattern
            elif isinstance(pattern, (str, os.PathLike)):
                self.pattern = formats.read_pattern(pattern)
            else:
                raise ValueError("Unrecognised pattern format")
        self._set_materials(opt, soil_reflectance, scene)
        self.soil = None
        self.soil_label = "soil"
        if self.scene and all(isinstance(key, (int, float, np.integer)) for key in self.scene):
            self.soil_label = -1
        if soil_mesh is not None and soil_mesh != -1:
            if self.pattern is None:
                raise ValueError("Adding a soil needs the scene domain to be defined")
            if z_soil is None:
                z_soil = 0 if self.scene is None else self.bbox()[0][2]
            self.soil = domain_mesh(self.pattern, z_soil)
        self.soil_raw, self.soil_aggregated = None, None

    def _set_materials(self, opt, soil_reflectance, scene):
        self.material = None
        if opt is None:
            self.soil_reflectance = ({self.default_band: self.default_soil_reflectance} if soil_reflectance is None
                                     else soil_reflectance)
            if self.scene is not None:
                self.material = {band: {key: self.default_material for key in self.scene}
                                 for band in self.soil_reflectance}
        elif isinstance(opt, list):
            if not isinstance(scene, (str, os.PathLike)):
                raise ValueError("un-compatible inputs types: opt file and scene not a can file")
            self.material, self.soil_reflectance = {}, {}
            for path in opt:
                band = os.path.basename(path).split(".")[0]
                _, soil, table = formats.read_opt(path)
                self.material[band] = _materials_of_labels(list(self.scene), table, soil)
                self.soil_reflectance[band] = soil
        elif isinstance(opt, dict):
            first = opt[list(opt)[0]]
            if isinstance(first, tuple):
                self.material = {band: {key: opt[band] for key in self.scene} for band in opt} if self.scene else {}
            elif isinstance(first, dict):
                self.material = opt
            else:
                raise ValueError("Unrecognised opt format")
            if soil_reflectance is None:
                self.soil_reflectance = {band: self.default_soil_reflectance for band in self.material}
            elif isinstance(soil_reflectance, dict):
                if len(soil_reflectance) != len(opt):
                    raise ValueError("the number of bands for optical properties and soil reflectance should match")
                self.soil_reflectance = soil_reflectance
            else:
                raise ValueError("Unrecognised soil_reflectance format")
        else:
            raise ValueError("Unrecognised opt format")

    def setLight(self, light):
        if isinstance(light, list):
            if not (isinstance(light[0], tuple) and isinstance(light[0][1], tuple)):
                raise ValueError("Unrecognised light format")
            self.light = light
        elif isinstance(light, (str, os.PathLike)):
            self.light = formats.read_light(light)
        else:
            raise ValueError("Unrecognised light format")

    def _triangles(self):
        """(triangles, groups) of all primitives, in the scene's order."""
        triangles = [triangle for triangles in self.scene.values() for triangle in triangles]
        groups = [key for key, triangles in self.scene.items() for _ in triangles]
        return triangles, groups

    def triangle_areas(self, convert=True) -> np.ndarray:
        """Each triangle's area (in m² when *convert*, else in the scene's unit)."""
        triangles = np.asarray(self._triangles()[0], dtype=np.float64)
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                               axis=1) / 2.0
        return areas * self.conv_unit ** 2 if convert else areas

    def bbox(self):
        """(xmin, ymin, zmin), (xmax, ymax, zmax) of the scene."""
        points = np.asarray(self._triangles()[0], dtype=np.float64).reshape(-1, 3)
        return tuple(points.min(axis=0).tolist()), tuple(points.max(axis=0).tolist())

    def auto_screen(self, screen_resolution) -> int:
        """The screen size giving pixels of *screen_resolution* (m) over the scene's diagonal."""
        pixel = screen_resolution * self.conv_unit
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bbox()
        diagonal = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        return max(2, int(diagonal / pixel))

    def getIncidentEnergy(self):
        """(Qi the sources' total horizontal irradiance, Qem the sum of their normal irradiances, Einc the energy
        entering the pattern, None without one)."""
        energies, directions = zip(*self.light)
        total = sum(energies)
        normal = sum(energy / abs(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)) for energy, (x, y, z) in self.light)
        entering = None
        if self.pattern is not None:
            xmin, ymin, xmax, ymax = self.pattern
            entering = total * abs((xmax - xmin) * (ymax - ymin)) * self.conv_unit ** 2
        return total, normal, entering

    def getSoilEnergy(self):
        """(Qi the soil's irradiance, Einc the energy it receives over the pattern) after a run with a soil."""
        if self.soil is None:
            raise ValueError("A soil should be added to allow SoilEnergy Estimation")
        if not self.soil_aggregated:
            raise ValueError("Caribu should have been called to allow SoilEnergy Estimation")
        incident = list(self.soil_aggregated.values())[0]["Ei"]
        entering = None
        if self.pattern is not None:
            xmin, ymin, xmax, ymax = self.pattern
            entering = incident * abs((xmax - xmin) * (ymax - ymin)) * self.conv_unit ** 2
        return incident, entering

    def run(self, direct=True, infinite=False, d_sphere=0.5, layers=10, height=None, screen_size=1536,
            screen_resolution=None, sensors=None, split_face=False, simplify=False, workers=None):
        """Light on the scene: (raw, aggregated), each {band: {name: {pid: values}}}, names Eabs, Ei, area (and
        Ei_inf, Ei_sup with *split_face*; "sensors" {name: {sensor id: values}} with *sensors* {id: [triangles]}).
        *direct*: first order only; *infinite*: the pattern repeated; *simplify*: no band level for a single band.
        Scattering (direct=False: radiosity, nested radiosity with *d_sphere*, *layers*, *height*) is not yet
        ported."""
        if not direct:
            raise NotImplementedError("scattering (radiosity, nested radiosity) is not yet ported")
        if infinite and self.pattern is None:
            raise ValueError("infinite canopy illumination needs a pattern to be defined")
        raw, aggregated = {}, {}
        self.soil_raw, self.soil_aggregated = {}, {}
        results = ["Eabs", "Ei", "area"] + (["Ei_inf", "Ei_sup"] if split_face else [])
        lights = self.light
        if self.conv_unit != 1:
            lights = [(energy * self.conv_unit ** 2, vector) for energy, vector in self.light]
        if self.scene is None:
            return raw, aggregated
        triangles, groups = self._triangles()
        bands = list(self.material)
        materials = {band: [self.material[band][group] for group in groups] for band in bands}
        if self.soil is not None:
            triangles = triangles + list(self.soil)
            groups = groups + [self.soil_label] * len(self.soil)
            for band in bands:
                materials[band] = materials[band] + [(self.soil_reflectance[band],)] * len(self.soil)
        if screen_resolution is not None:
            screen_size = self.auto_screen(screen_resolution)
        sensor_groups = None
        if sensors is not None:
            sensor_groups = [key for key, values in sensors.items() for _ in values]
            sensors = [triangle for values in sensors.values() for triangle in values]
        domain = self.pattern if infinite else None
        if len(bands) == 1:
            out = {bands[0]: raycasting(triangles, materials[bands[0]], lights, domain, screen_size, sensors,
                                        precision=self.precision, workers=workers)}
        else:
            out = x_raycasting(triangles, materials, lights, domain, screen_size, sensors, precision=self.precision,
                               workers=workers)
        keys = sorted(set(groups))
        for band in bands:
            output = _convert(out[band], self.conv_unit)
            raw[band], aggregated[band] = {}, {}
            if "sensors" in output:
                measured = output.pop("sensors")
                sensor_keys = sorted(set(sensor_groups))
                raw[band]["sensors"], aggregated[band]["sensors"] = {}, {}
                for name in ("Ei", "Ei0", "area"):
                    raw[band]["sensors"][name] = _aggregate(measured[name], sensor_groups, sensor_keys, "list")
                    aggregated[band]["sensors"][name] = _aggregate(
                        measured[name] if name == "area" else (measured[name], measured["area"]), sensor_groups,
                        sensor_keys, "sum" if name == "area" else "mean")
            for name in results:
                raw[band][name] = _aggregate(output[name], groups, keys, "list")
                aggregated[band][name] = _aggregate(output[name] if name == "area" else (output[name], output["area"]),
                                                    groups, keys, "sum" if name == "area" else "mean")
            if self.soil is not None:
                self.soil_raw[band] = {name: raw[band][name].pop(self.soil_label) for name in results}
                self.soil_aggregated[band] = {name: aggregated[band][name].pop(self.soil_label) for name in results}
        if simplify and len(bands) == 1:
            raw, aggregated = raw[bands[0]], aggregated[bands[0]]
        return raw, aggregated

    def runPeriodise(self):
        """The scene's triangles moved into the pattern, as periodise moves them (6 significant digits); the
        primitives keep their ids."""
        triangles, groups = self._triangles()
        moved = periodic.periodise(formats_points(triangles), self.pattern).astype(np.float64)
        scene = {}
        for triangle, group in zip(moved.tolist(), groups):
            scene.setdefault(group, []).append(tuple(map(tuple, triangle)))
        self.scene = scene
        return self


def formats_points(triangles):
    """Triangles as canestrad and periodise read them (6 decimals, float32)."""
    from .geometry import as_legacy_vertices
    return as_legacy_vertices(np.asarray(triangles, dtype=np.float64))


def _materials_of_labels(labels, table, soil_reflectance) -> dict:
    """{label: material} of `.can` labels from an `.opt` table (file_adaptor.build_materials)."""
    materials = {}
    for label in labels:
        number = int(label)
        species, leaf = number // 10 ** 11, (number // 1000) % 1000
        if leaf == 0 and species != 0:
            materials[label] = (table[species][0],)
        elif leaf == 0:
            materials[label] = (soil_reflectance,)
        else:
            rho_sup, tau_sup, rho_inf, tau_inf = table[species][1:]
            materials[label] = (rho_sup, tau_sup) if (rho_sup, tau_sup) == (rho_inf, tau_inf) else (
                rho_sup, tau_sup, rho_inf, tau_inf)
    return materials


def _convert(output, conv_unit) -> dict:
    """Outputs to m² and per m² from the scene's unit (CaribuScene's _convert)."""
    if conv_unit == 1:
        return output
    meter_square = conv_unit ** 2
    output = dict(output)
    output["area"] = [area * meter_square for area in output["area"]]
    for name in ("Eabs", "Ei", "Ei_sup", "Ei_inf"):
        output[name] = [energy / meter_square for energy in output[name]]
    if "sensors" in output:
        sensors = dict(output["sensors"])
        sensors["area"] = [area * meter_square for area in sensors["area"]]
        for name in ("Ei", "Ei0"):
            sensors[name] = [energy / meter_square for energy in sensors[name]]
        output["sensors"] = sensors
    return output
