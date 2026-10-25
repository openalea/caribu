"""
The first-order radiosities: each face's light from the sources only, before any scattering (canestrad's main light
loop, B.6 of the description). A source of irradiance E (on a horizontal plane) and its beam give each face a sunlit
area A (projection.projected_areas); the face then re-emits

    B0 = Σ_sources E · ρ · A / S          for the lit face (ρ its reflectance, S the triangle's area)
    B0 = Σ_sources E · τ · |A| / S        for a leaf's other face (τ the lit face's transmittance)

Faces are numbered as canestrad's diffusers: one per opaque triangle, two per leaf (upper, then lower), in scene order
of the accepted triangles. The legacy's numerics are kept (legacy_surfaces: Polygone::surface on float32 points).
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from . import budgets, geometry, periodic, projection
from .labels import is_translucent, species_of

F32 = np.float32


@njit(cache=True, nogil=True, error_model="numpy")
def _legacy_norm(x, y, z):
    return np.float64(np.sqrt(F32(x * x + y * y + z * z)))


@njit(cache=True, nogil=True, error_model="numpy")
def legacy_surfaces(points):
    """Each triangle's area as Polygone::surface computes it on float32 points: |P0P1| |P0P2| sin(angle) / 2, the
    angle the arc cosine of the unit edges' float32 dot product."""
    surfaces = np.zeros(points.shape[0])
    for t in range(points.shape[0]):
        first = points[t, 1] - points[t, 0]
        second = points[t, 2] - points[t, 0]
        first_norm = _legacy_norm(first[0], first[1], first[2])
        second_norm = _legacy_norm(second[0], second[1], second[2])
        dot = F32(0.)
        for q in range(3):
            dot = F32(dot + F32(F32(np.float64(first[q]) / first_norm) * F32(np.float64(second[q]) / second_norm)))
        cosine = np.float64(dot)
        if abs(cosine) > 1.:
            angle = 0. if cosine > 0 else math.pi
        else:
            angle = math.acos(cosine)
        surfaces[t] = first_norm * second_norm * math.sin(angle) / 2.0
    return surfaces


def face_numbering(translucent):
    """(first face of each triangle, number of faces): one face per opaque triangle, two per leaf."""
    faces_per_triangle = np.where(translucent, 2, 1)
    first_face = np.concatenate([[0], np.cumsum(faces_per_triangle)[:-1]]).astype(np.int64)
    return first_face, int(faces_per_triangle.sum())


def face_optics(labels, optics):
    """(n_faces,) reflectance and transmittance of each face, as canestrad's diffusers: an opaque triangle the stem
    reflectance of its species (the soil's for species 0), a leaf its species' upper then lower (ρ, τ).
    *optics*: (soil reflectance, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)})."""
    soil, species_optics = optics
    translucent = is_translucent(labels)
    species = species_of(labels)
    table = np.full((max([0, *species_optics]) + 1, 5), np.nan)
    table[0, 0] = soil
    for number, material in species_optics.items():
        table[number] = material
    first_face, n_faces = face_numbering(translucent)
    reflectance, transmittance = np.zeros(n_faces), np.zeros(n_faces)
    opaque = ~translucent
    reflectance[first_face[opaque]] = table[species[opaque], 0]
    leaves = first_face[translucent]
    reflectance[leaves], transmittance[leaves] = table[species[translucent], 1], table[species[translucent], 2]
    reflectance[leaves + 1], transmittance[leaves + 1] = table[species[translucent], 3], table[species[translucent], 4]
    return reflectance, transmittance


@njit(cache=True, nogil=True, error_model="numpy")
def accumulate(radiosity, areas, energy, first_face, translucent, reflectance, transmittance, surfaces):
    """Adds a source's contribution to *radiosity* (n_faces,): E ρ A / S on the lit face, E τ(lit face) |A| / S on a
    leaf's other face (A < 0 there), in canestrad's order of operations."""
    for t in range(first_face.shape[0]):
        faces = 2 if translucent[t] else 1
        for offset in range(faces):
            face = first_face[t] + offset
            area = areas[face]
            if area == 0.0:
                continue
            if area > 0:
                rho = reflectance[face] * area / surfaces[t]
            else:
                rho = -transmittance[first_face[t] + 1 - offset] * area / surfaces[t]
            radiosity[face] += energy * rho


def first_order(points, labels, lights, optics, size=1536, bounds=None, sensors=None, workers=None, pattern=None,
                flags=None, cache=None):
    """The accepted triangles' first-order radiosities (n_faces,) in canestrad's face order, their areas, and the
    sensors' (m,) irradiances (their first-order radiosities: a sensor reflects all, ρ = 1). *points*: float32
    (n, 3, 3) accepted triangles; *lights*: [(E, (vx, vy, vz))]; *bounds*: the scene's (vmin, vmax) float32, by default
    those of *points*; *sensors*: float32 (m, 3, 3) virtual sensors, seen only from above their upper face, which
    neither shade nor scatter; *workers*: the threads projecting the sources (by default, one per processor). With a
    *pattern* (xmin, ymin, xmax, ymax) the canopy is infinite: the screen spans the pattern, the triangles across its
    borders (*flags*, periodic.border_flags') drawn with their shifted copies, the image tiled. *cache*: a dict of
    each direction's lit areas, filled and reused (the same scene, size and sensors: the caller's to keep so)."""
    translucent = is_translucent(labels)
    first_face, n_faces = face_numbering(translucent)
    reflectance, transmittance = face_optics(labels, optics)
    surfaces = legacy_surfaces(points)
    if bounds is None:
        flat = points.reshape(-1, 3) if len(points) else np.zeros((1, 3), dtype=F32)
        bounds = flat.min(axis=0).astype(F32), flat.max(axis=0).astype(F32)
        if pattern is not None:
            low, high = periodic.pattern_bounds(pattern)
            bounds[0][:2], bounds[1][:2] = low, high
    vmin, vmax = bounds
    normal = geometry.legacy_normals(points)
    if pattern is None:
        copies = np.zeros((len(points), 3), dtype=bool)
        copies[:, 0] = True
        shifts = np.zeros((3, 3), dtype=F32)
    else:
        copies, shifts = periodic.border_copies(flags), periodic.border_shifts(vmin, vmax)
    if sensors is None:
        sensors = np.zeros((0, 3, 3), dtype=F32)
    sensor_normal = geometry.legacy_normals(sensors)
    sensor_surfaces = legacy_surfaces(sensors)
    sensor_copies = np.zeros((len(sensors), 3), dtype=bool)
    sensor_copies[:, 0] = True

    def lit_areas(direction):
        """A source's per-face sunlit areas and the sensors' (projplan)."""
        origin, v, w, u, du, dv, costeta = projection.legacy_screen(vmin, vmax, np.array(direction, dtype=F32))
        owner, depth = projection.rasterise(points, copies, shifts, origin, v, w, u, du, dv, size)
        if pattern is not None and np.float64(u[2]) > -1 + 1e-6:
            steps = periodic.tiling_steps(periodic.roof_pixels(vmin, vmax, origin, v, w, u, du, dv, size))
            periodic.tile(owner, depth, periodic.translations(steps, size), periodic.distance_step(u[2], dv, size))
        with np.errstate(divide="ignore", invalid="ignore"):  # a horizontal beam: canestrad's IEEE division
            pixel_area = np.float64(du * dv / float(size * size)) / np.float64(costeta)
        areas = projection.projected_areas(owner, n_faces, first_face, translucent, _facing(u, normal), pixel_area)
        seen = projection.sensor_areas(depth, sensors, _facing(u, sensor_normal), sensor_copies, shifts, origin, v,
                                       w, u, du, dv, size, pixel_area)
        return areas, seen

    directions = [tuple(float(value) for value in direction) for _, direction in lights]
    cache = {} if cache is None else cache
    missing = [direction for direction in dict.fromkeys(directions) if direction not in cache]
    radiosity, sensor_radiosity = np.zeros(n_faces), np.zeros(len(sensors))
    # the sources' projections in parallel (numba releases the GIL), their contributions summed in the sources' order
    with ThreadPoolExecutor(max_workers=workers) as executor:
        cache.update(zip(missing, executor.map(lit_areas, missing)))
        for (energy, _), (areas, seen) in zip(lights, (cache[direction] for direction in directions)):
            accumulate(radiosity, areas, float(energy), first_face, translucent, reflectance, transmittance, surfaces)
            lit = seen != 0.0
            sensor_radiosity[lit] += float(energy) * (seen[lit] / sensor_surfaces[lit])
    return radiosity, surfaces, sensor_radiosity, sensor_surfaces


def _facing(direction, normal):
    """Whether the beam meets each triangle's upper face (direction · normal < 0, the float32 dot product)."""
    product = F32(direction[0]) * normal[:, 0] + F32(direction[1]) * normal[:, 1]
    return (product + F32(direction[2]) * normal[:, 2]) < 0


def scene(vertices, labels, pattern=None) -> tuple:
    """The triangles as canestrad reads them: (points float32 (n, 3, 3), labels (n,), accepted (n,), border flags
    of the accepted ones or None); with a *pattern*, periodised first, those whose centre lies outside rejected."""
    points = geometry.as_legacy_vertices(vertices)
    labels = np.broadcast_to(np.asarray(labels, dtype=np.int64), len(points))
    flags = None
    if pattern is not None:
        points = periodic.periodise(points, pattern)
    accepted = ~geometry.rejected(points)
    if pattern is not None:
        inside, flags = periodic.border_flags(points, pattern)
        accepted &= inside
        flags = flags[accepted]
    return points, labels, accepted, flags


def results(labels, accepted, points, surfaces, radiosity, optics) -> dict:
    """genres on the accepted triangles' face *radiosity*: per input triangle "area", "Eabs", "Ei", "Ei_sup",
    "Ei_inf" (float32, NaN if rejected), "accepted", "points"."""
    kept = labels[accepted]
    reflectance, transmittance = face_optics(kept, optics)
    values = budgets.triangle_budgets(radiosity, is_translucent(kept), reflectance, transmittance)
    result = {"accepted": accepted, "area": np.zeros(len(labels)), "points": points}
    result["area"][accepted] = surfaces
    for name, value in zip(("Eabs", "Ei", "Ei_sup", "Ei_inf"), values):
        result[name] = np.full(len(labels), np.nan, dtype=F32)
        result[name][accepted] = value
    return result


def run(vertices, labels, lights, optics, size=1536, sensors=None, workers=None, pattern=None, cache=None,
        prepared=None):
    """Direct light on a scene, as canestrad -1: per input triangle (n,) its area (canestrad's, 0 if rejected), Eabs,
    Ei (sup + inf), Ei_sup, Ei_inf (float32; NaN if rejected, Ei_inf −1 for an opaque triangle) and whether it was
    accepted. *vertices* (n, 3, 3), *labels* (n,) or one for all, *lights* [(E, (vx, vy, vz))], *optics* (soil reflectance,
    {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}); *size* the z-buffer's side in pixels. With *sensors*
    (m, 3, 3), result["sensors"] holds their irradiance "Ei" and "area". *workers*: first_order's. With a *pattern*
    (x1, y1, x2, y2) the canopy is infinite (canestrad -8, after periodise): the triangles are first moved into the
    pattern (result["points"], the scene canestrad reads), those whose centre still lies outside it rejected.
    *cache*: first_order's; *prepared*: scene's result on these vertices, labels and pattern."""
    points, labels, accepted, flags = scene(vertices, labels, pattern) if prepared is None else prepared
    kept = labels[accepted]
    if not accepted.any():
        lights = []  # nothing to light: every row rejected
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    radiosity, surfaces, sensor_radiosity, sensor_surfaces = first_order(points[accepted], kept, lights, optics, size,
                                                                         sensors=sensor_points, workers=workers,
                                                                         pattern=pattern, flags=flags, cache=cache)
    result = results(labels, accepted, points, surfaces, radiosity, optics)
    if sensors is not None:
        result["sensors"] = {"Ei": sensor_radiosity, "area": sensor_surfaces}
    return result
