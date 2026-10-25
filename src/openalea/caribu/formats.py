"""
Caribu's file formats, read and written: the scene (`.can`: one triangle per line, `p 1 <label> 3 x y z x y z x y z`),
the lights (`.light`: `E vx vy vz` per line), the optical properties (`.opt`), the pattern of an infinite canopy
(`.8`: `xmin ymin` / `xmax ymax`) and the sensors. The writers give the legacy's text exactly (caribu.py's
builders), so that the port and the legacy can be fed the same scenes; the port itself needs no file.
"""
import numpy as np


# --- Readers ---

def read_can_arrays(path):
    """(vertices (n, 3, 3), labels (n,) int64) of a `.can` file, in file order (comments skipped; the last nine numbers
    of a line are its triangle's coordinates)."""
    vertices, labels = [], []
    with open(path) as lines:
        for line in lines:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            labels.append(int(float(fields[2])))
            vertices.append([float(value) for value in fields[-9:]])
    return np.array(vertices, dtype=np.float64).reshape(-1, 3, 3), np.array(labels, dtype=np.int64)


def read_can(path) -> dict:
    """{label (str): [triangles]} of a `.can` file, as the legacy's file_adaptor.read_can."""
    scene = {}
    with open(path) as lines:
        for line in lines:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            coordinates = [float(value) for value in fields[-9:]]
            scene.setdefault(fields[2], []).append([tuple(coordinates[:3]), tuple(coordinates[3:6]),
                                                    tuple(coordinates[6:])])
    return scene


def read_light(path) -> list:
    """[(E, (vx, vy, vz))] of a `.light` file: E the irradiance on a horizontal plane, (vx, vy, vz) the propagation
    direction."""
    lights = []
    with open(path) as lines:
        for line in lines:
            fields = line.split()
            if fields:
                energy, x, y, z = (float(value) for value in fields[:4])
                lights.append((energy, (x, y, z)))
    return lights


def read_pattern(path) -> tuple:
    """(xmin, ymin, xmax, ymax) of a `.8` file."""
    values = []
    with open(path) as lines:
        for line in lines:
            values.extend(float(value) for value in line.split())
    return tuple(values[:4])


def read_opt(path) -> tuple:
    """(n, soil reflectance, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}) of a `.opt` file, species
    numbered from 1 in the order of its `e` lines."""
    with open(path) as text:
        return parse_opt(text.read())


def parse_opt(text) -> tuple:
    """read_opt of a `.opt` text."""
    n, soil, species = None, None, {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "n":
            n = int(fields[1])
        elif fields[0] == "s":
            soil = float(fields[2])
        elif fields[0] == "e":
            species[len(species) + 1] = tuple(float(fields[index]) for index in (2, 4, 5, 7, 8))
    return n, soil, species


# --- PlantGL ---

def plantgl_triangles(geometry, tesselator=None) -> list:
    """The triangles of a PlantGL geometry or shape, tesselated (PlantGL imported only here)."""
    import openalea.plantgl.all as pgl
    tesselator = pgl.Tesselator() if tesselator is None else tesselator
    geometry.apply(tesselator)
    mesh = tesselator.triangulation
    if not mesh:
        return []
    points = list(map(tuple, mesh.pointList))
    return [(points[index[0]], points[index[1]], points[index[2]]) for index in mesh.indexList]


def plantgl_scene(scene) -> dict:
    """{shape id: [triangles]} of a PlantGL scene (shapes without triangles left out)."""
    import openalea.plantgl.all as pgl
    tesselator = pgl.Tesselator()
    triangles = {}
    for key, shapes in scene.todict().items():
        found = [triangle for shape in shapes for triangle in plantgl_triangles(shape, tesselator)]
        if found:
            triangles[key] = found
    return triangles


def mtg_scene(g, property_name="geometry") -> dict:
    """{vertex: [triangles]} of an MTG's PlantGL geometries."""
    import openalea.plantgl.all as pgl
    tesselator = pgl.Tesselator()
    return {vertex: plantgl_triangles(geometry, tesselator) for vertex, geometry in g.property(property_name).items()}


# --- Writers (the legacy's text) ---

def can_string(vertices, labels) -> str:
    """The `.can` text of triangles and their labels (6 decimals, as caribu.py's triangles_string)."""
    return "".join("p 1 %s 3" % label + "".join(" %.6f %.6f %.6f" % tuple(point) for point in triangle) + "\n"
                   for triangle, label in zip(np.asarray(vertices, dtype=np.float64).tolist(), labels))


def light_string(lights) -> str:
    """The `.light` text of [(E, (vx, vy, vz))]."""
    return "".join(" ".join(map(str, [energy] + list(direction))) + "\n" for energy, direction in lights)


def opt_string(species, soil_reflectance=-1) -> str:
    """The `.opt` text of {species number: material}: a material is (rho,) opaque, (rho, tau) translucent symmetric,
    (rho_sup, tau_sup, rho_inf, tau_inf) translucent asymmetric; black bodies are refused."""
    text = "n %s\n" % len(species) + "s d %s\n" % soil_reflectance
    for key in sorted(species):
        material = species[key]
        if sum(material) <= 0:
            raise ValueError("Caribu does not accept black body materials (absorptance 1)")
        if len(material) == 1:
            text += "e d %s   d -1 -1  d -1 -1\n" % material
        elif len(material) == 2:
            text += "e d -1   d %s %s" % material + " d %s %s\n" % material
        else:
            text += "e d -1   d %s %s  d %s %s\n" % material
    return text


def pattern_string(pattern) -> str:
    """The `.8` text of (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = pattern
    low, high = (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))
    return "\n".join([" ".join(map(str, low)), " ".join(map(str, high)), " "])


def sensor_string(vertices) -> str:
    """The sensors' text: `#n`, then one triangle per line, numbered from 1."""
    vertices = np.asarray(vertices, dtype=np.float64).tolist()
    return "#%s\n" % len(vertices) + "".join(
        "p 1 %s 3" % (index + 1) + "".join(" %.6f %.6f %.6f" % tuple(point) for point in triangle) + "\n"
        for index, triangle in enumerate(vertices))
