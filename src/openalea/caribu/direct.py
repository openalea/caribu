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

import numpy as np
from numba import njit

from . import budgets, geometry, projection
from .labels import is_translucent, species_of

F32 = np.float32


@njit(cache=True)
def _legacy_norm(x, y, z):
    return np.float64(np.sqrt(F32(x * x + y * y + z * z)))


@njit(cache=True)
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


@njit(cache=True)
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


def first_order(points, labels, lights, optics, size=1536, bounds=None, sensors=None):
    """The accepted triangles' first-order radiosities (n_faces,) in canestrad's face order, their areas, and the
    sensors' (m,) irradiances (their first-order radiosities: a sensor reflects all, ρ = 1). *points*: float32
    (n, 3, 3) accepted triangles; *lights*: [(E, (vx, vy, vz))]; *bounds*: the scene's (vmin, vmax) float32, by default
    those of *points*; *sensors*: float32 (m, 3, 3) virtual sensors, seen only from above their upper face, which
    neither shade nor scatter."""
    translucent = is_translucent(labels)
    first_face, n_faces = face_numbering(translucent)
    reflectance, transmittance = face_optics(labels, optics)
    surfaces = legacy_surfaces(points)
    if bounds is None:
        flat = points.reshape(-1, 3)
        bounds = flat.min(axis=0).astype(F32), flat.max(axis=0).astype(F32)
    vmin, vmax = bounds
    normal = geometry.legacy_normals(points)
    copies = np.zeros((len(points), 3), dtype=bool)
    copies[:, 0] = True
    shifts = np.zeros((3, 3), dtype=F32)
    radiosity = np.zeros(n_faces)
    if sensors is None:
        sensors = np.zeros((0, 3, 3), dtype=F32)
    sensor_normal = geometry.legacy_normals(sensors)
    sensor_surfaces = legacy_surfaces(sensors)
    sensor_copies = np.zeros((len(sensors), 3), dtype=bool)
    sensor_copies[:, 0] = True
    sensor_radiosity = np.zeros(len(sensors))
    for energy, direction in lights:
        origin, v, w, u, du, dv, costeta = projection.legacy_screen(vmin, vmax, np.array(direction, dtype=F32))
        lit_upper = _facing(u, normal)
        owner, depth = projection.rasterise(points, copies, shifts, origin, v, w, u, du, dv, size)
        pixel_area = du * dv / float(size * size) / costeta
        areas = projection.projected_areas(owner, n_faces, first_face, translucent, lit_upper, pixel_area)
        accumulate(radiosity, areas, float(energy), first_face, translucent, reflectance, transmittance, surfaces)
        if len(sensors):
            sensor_areas = projection.sensor_areas(depth, sensors, _facing(u, sensor_normal), sensor_copies, shifts,
                                                   origin, v, w, u, du, dv, size, pixel_area)
            lit = sensor_areas != 0.0
            sensor_radiosity[lit] += float(energy) * (sensor_areas[lit] / sensor_surfaces[lit])
    return radiosity, surfaces, sensor_radiosity, sensor_surfaces


def _facing(direction, normal):
    """Whether the beam meets each triangle's upper face (direction · normal < 0, the float32 dot product)."""
    product = F32(direction[0]) * normal[:, 0] + F32(direction[1]) * normal[:, 1]
    return (product + F32(direction[2]) * normal[:, 2]) < 0


def run(vertices, labels, lights, optics, size=1536, sensors=None):
    """Direct light on a scene, as canestrad -1: per input triangle (n,) its area (canestrad's, 0 if rejected), Eabs,
    Ei (sup + inf), Ei_sup, Ei_inf (float32; NaN if rejected, Ei_inf −1 for an opaque triangle) and whether it was
    accepted. *vertices* (n, 3, 3), *labels* (n,) or one for all, *lights* [(E, (vx, vy, vz))], *optics* (soil reflectance,
    {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}); *size* the z-buffer's side in pixels. With *sensors*
    (m, 3, 3), result["sensors"] holds their irradiance "Ei" and "area"."""
    points = geometry.as_legacy_vertices(vertices)
    labels = np.broadcast_to(np.asarray(labels, dtype=np.int64), len(points))
    accepted = ~geometry.rejected(points)
    kept = labels[accepted]
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    radiosity, surfaces, sensor_radiosity, sensor_surfaces = first_order(points[accepted], kept, lights, optics, size,
                                                                         sensors=sensor_points)
    reflectance, transmittance = face_optics(kept, optics)
    names = ("Eabs", "Ei", "Ei_sup", "Ei_inf")
    values = budgets.triangle_budgets(radiosity, is_translucent(kept), reflectance, transmittance)
    result = {"accepted": accepted, "area": np.zeros(len(labels))}
    result["area"][accepted] = surfaces
    for name, value in zip(names, values):
        result[name] = np.full(len(labels), np.nan, dtype=F32)
        result[name][accepted] = value
    if sensors is not None:
        result["sensors"] = {"Ei": sensor_radiosity, "area": sensor_surfaces}
    return result
