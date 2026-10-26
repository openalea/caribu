"""
Radiosity on a closed (non-periodic) scene: the direct light, then every order of scattering between the triangles
(canestrad -d −1, B.7 and C of the description). The sphere of the near field holds the whole scene (its radius the
scene's diagonal), so every pair of triangles exchanges light through its form factors.

    result = run(vertices, labels, lights, optics)      as direct.run, the radiosities of the full system
"""
import numpy as np

from . import direct, geometry, nearfield, solver
from .labels import is_translucent

F32 = np.float32


def run(vertices, labels, lights, optics, size=1536, sensors=None, workers=None, resolution=52, tolerance=1e-6,
        iterations=1000, counts=None, cache=None, prepared=None, device="cpu", corrections=frozenset()):
    """Radiosity on the triangles: direct.run's result, its budgets from the solved radiosities; result["steps"]
    the solver's steps, result["counts"] the near field's. *resolution*: the receivers' discs, NB × NB pixels;
    *tolerance*, *iterations*: MGCR's (relative to the first residual); *counts*: a previous run's on the same
    triangles and sensors (another band), not computed again; *cache*: first_order's; *prepared*: direct.scene's
    result; *device* "cuda": the near field on the GPU; *corrections*: corrections.resolve's names."""
    points, labels, accepted, _ = direct.scene(vertices, labels, None, corrections) if prepared is None else prepared
    kept_points, kept = points[accepted], labels[accepted]
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    first, surfaces, sensor_first, sensor_surfaces = direct.first_order(kept_points, kept, lights, optics, size,
                                                                        sensors=sensor_points, workers=workers,
                                                                        cache=cache, corrections=corrections)
    translucent = is_translucent(kept)
    first_face, n_faces = direct.face_numbering(translucent)
    reflectance, transmittance = direct.face_optics(kept, optics)
    flat = kept_points.reshape(-1, 3)
    extent = (flat.max(axis=0) - flat.min(axis=0)).astype(F32).astype(np.float64)
    radius = np.sqrt(np.sum(extent * extent))
    normals, centres = geometry.legacy_normals(kept_points), geometry.legacy_centres(kept_points)
    sensor_count = 0 if sensor_points is None else len(sensor_points)
    sensor_normals = sensor_centres = None
    if sensor_count:
        sensor_normals, sensor_centres = geometry.legacy_normals(sensor_points), geometry.legacy_centres(sensor_points)
    if counts is None:
        counts = nearfield.counts(kept_points, kept, translucent, first_face, n_faces, normals, centres, radius,
                                  sensor_points, sensor_normals, sensor_centres, resolution, workers, device=device,
                                  options=nearfield.options(corrections))
    indptr, faces, values = counts
    receiver_faces = np.concatenate([first_face, n_faces + np.arange(sensor_count)]).astype(np.int64)
    receiver_translucent = np.concatenate([translucent, np.zeros(sensor_count, dtype=bool)])
    all_reflectance = np.concatenate([reflectance, np.ones(sensor_count)])
    all_transmittance = np.concatenate([transmittance, np.zeros(sensor_count)])
    rhs = np.concatenate([first, sensor_first])
    radiosities, steps = solver.mgcr(rhs, indptr, faces, values, receiver_faces, receiver_translucent,
                                     all_reflectance, all_transmittance, nearfield.form_factor(resolution), tolerance,
                                     100, iterations)
    result = direct.results(labels, accepted, points, surfaces, radiosities[:n_faces], optics)
    result.update(steps=steps, first_order=first, radiosity=radiosities[:n_faces], counts=counts)
    if sensors is not None:
        result["sensors"] = {"Ei0": sensor_first, "Ei": radiosities[n_faces:], "area": sensor_surfaces}
    return result
