"""
Light in an infinite canopy with scattering (caribu's mixed radiosity: canestrad -8 -d D -e band.env after
periodise, s2v and mcsail). The far field is the canopy as a turbid medium (sail): its layers' fluxes. With a null
sphere (D = 0, pure SAIL), each triangle's scattered light is the first scattering of the far field's fluxes at its
height (canestrad's sail_pur), added to its direct light; with D > 0, the triangles within a sphere of diameter D
exchange light by radiosity (their periodic images included), the far field lighting what each does not see of
its neighbours (nested radiosity, Chelle, Andrieu & Bouatouch 1998).

    result = run(vertices, labels, lights, optics, pattern, layers=10, height=1.)    as direct.run's result, B0
                                                                                      plus the far field's light
"""
import numpy as np
from numba import njit

from . import direct, geometry, nearfield, sail, solver
from .labels import is_translucent


def far_field_profile(points, labels, lights, optics, pattern, layers, height, precision="legacy",
                      corrections=frozenset()) -> tuple:
    """The far field of the periodised *points*: (profile, structure), s2v's layers and mcsail's fluxes from the
    soil up, each read back with its file's printed digits when *precision* is the legacy's; *corrections*:
    sail.layers'."""
    structure = sail.layers(points, labels, layers, height, pattern, [optics], corrections=corrections)
    read = sail.printed_layers(structure) if precision == "legacy" else structure
    profile = sail.profiles(read["frequencies"][::-1], read["leaf_area_index"][::-1], read["reflectance"][0][::-1],
                            read["transmittance"][0][::-1], read["soil_reflectance"][0], read["thickness"], lights)
    if precision == "legacy":
        profile.update(sail.printed_profile(profile, read["thickness"]))
    else:
        profile["thickness"] = read["thickness"]
    return profile, structure


def run(vertices, labels, lights, optics, pattern, layers=10, height=1., diameter=0., size=1536, sensors=None,
        workers=None, precision="legacy", bias=True, resolution=52, tolerance=1e-6, iterations=1000, threshold=50,
        ratio=1., weights=None, cache=None, prepared=None, device="cpu", corrections=frozenset()):
    """Direct light and its scattering in the infinite canopy repeating *pattern* (x1, y1, x2, y2), the far field
    in *layers* up to *height*: direct.run's result, its budgets from the radiosities; result["profile"] the far
    field's fluxes, result["first_order"] B0, result["far_field"] B_far. *diameter*: the near field's sphere (0:
    pure SAIL, B = B0 + B_far; > 0: B − F B = B0 + B_far, F the near field's form factors, solved as radiosity.run;
    result["counts"], result["weights"] the far field's levels' weights). *precision* "legacy": the intermediate
    files' printed digits (s2v's, mcsail's), as canestrad reads them. *bias* False: canestrad's `-B` (emitters by
    their distance to the sphere, hits beyond it dropped); *threshold*, *ratio*: the grid's (`grille_par`).
    *weights*: (counts, weights) of a run on the same scene in another band, the far field from them (canestrad's
    `-w`, its scaling by the last source's energy kept while *precision* is the legacy's). *cache*: first_order's;
    *prepared*: direct.scene's result; *device* "cuda": the near field on the GPU; *corrections*:
    corrections.resolve's names."""
    if diameter != 0:
        return _nested(vertices, labels, lights, optics, pattern, layers, height, diameter, size, sensors, workers,
                       precision, bias, resolution, tolerance, iterations, threshold, ratio, weights, cache, prepared,
                       device, corrections)
    points, labels, accepted, flags = (direct.scene(vertices, labels, pattern, corrections) if prepared is None
                                       else prepared)
    kept_points, kept = points[accepted], labels[accepted]
    profile, structure = far_field_profile(points, labels, lights, optics, pattern, layers, height, precision,
                                           corrections)
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    first, surfaces, sensor_first, sensor_surfaces, incident = direct.first_order(
        kept_points, kept, lights if accepted.any() else [], optics, size, sensors=sensor_points, workers=workers,
        pattern=pattern, flags=flags, cache=cache, corrections=corrections)
    translucent = is_translucent(kept)
    first_face, n_faces = direct.face_numbering(translucent)
    reflectance, transmittance = direct.face_optics(kept, optics)
    far, far_incident = sail.far_field(geometry.legacy_centres(kept_points), geometry.legacy_normals(kept_points),
                                       translucent, first_face, n_faces, reflectance, transmittance, profile,
                                       corrections, incident=True)
    result = direct.results(labels, accepted, points, surfaces, first + far, optics,
                            incident + far_incident if "face_irradiance" in corrections else None)
    result.update(first_order=first, far_field=far, profile=profile, structure=structure, radiosity=first + far)
    if sensors is not None:
        count = len(sensor_points)
        sensor_far = sail.far_field(geometry.legacy_centres(sensor_points), geometry.legacy_normals(sensor_points),
                                    np.zeros(count, dtype=bool), np.arange(count), count, np.ones(count),
                                    np.zeros(count), profile, corrections)
        result["sensors"] = {"Ei0": sensor_first, "Ei": sensor_first + sensor_far, "area": sensor_surfaces,
                             "first_order": sensor_first}
    return result


def _nested(vertices, labels, lights, optics, pattern, layers, height, diameter, size, sensors, workers, precision,
            bias, resolution, tolerance, iterations, threshold, ratio, weights, cache, prepared, device, corrections):
    points, labels, accepted, flags = (direct.scene(vertices, labels, pattern, corrections) if prepared is None
                                       else prepared)
    kept_points, kept = points[accepted], labels[accepted]
    profile, structure = far_field_profile(points, labels, lights, optics, pattern, layers, height, precision,
                                           corrections)
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    first, surfaces, sensor_first, sensor_surfaces, incident = direct.first_order(
        kept_points, kept, lights if accepted.any() else [], optics, size, sensors=sensor_points, workers=workers,
        pattern=pattern, flags=flags, cache=cache, corrections=corrections)
    translucent = is_translucent(kept)
    first_face, n_faces = direct.face_numbering(translucent)
    reflectance, transmittance = direct.face_optics(kept, optics)
    sensor_count = 0 if sensor_points is None else len(sensor_points)
    receiver_translucent = np.concatenate([translucent, np.zeros(sensor_count, dtype=bool)])
    lower = np.where(translucent, first_face + 1, first_face)
    receiver_optics = np.zeros((len(kept) + sensor_count, 4))
    receiver_optics[:len(kept), 0], receiver_optics[:len(kept), 1] = reflectance[first_face], transmittance[first_face]
    receiver_optics[:len(kept), 2] = np.where(translucent, reflectance[lower], 0.)
    receiver_optics[:len(kept), 3] = np.where(translucent, transmittance[lower], 0.)
    receiver_optics[len(kept):, 0] = 1.  # a sensor reflects all
    radius = float(str(diameter)) / 2.  # canestrad -d: the diameter's text, halved
    if weights is None:
        normals, centres = geometry.legacy_normals(kept_points), geometry.legacy_centres(kept_points)
        sensor_normals = sensor_centres = None
        if sensor_count:
            sensor_normals = geometry.legacy_normals(sensor_points)
            sensor_centres = geometry.legacy_centres(sensor_points)
        cells = nearfield.grid(kept_points, sensor_points, radius, pattern, threshold, ratio)
        counts, far, level_weights = nearfield.counts(
            kept_points, kept, translucent, first_face, n_faces, normals, centres, radius, sensor_points,
            sensor_normals, sensor_centres, resolution, workers, grid=cells, profile=profile, optics=receiver_optics,
            bias=bias, device=device, options=nearfield.options(corrections))
    else:
        counts, level_weights = weights
        far = reused_far_field(level_weights, receiver_optics, receiver_translucent, profile, lights, precision)
    receiver_faces = np.concatenate([first_face, n_faces + np.arange(sensor_count)]).astype(np.int64)
    far_faces = np.zeros(n_faces + sensor_count)
    far_faces[receiver_faces] = far[:, 0]
    far_faces[receiver_faces[receiver_translucent] + 1] = far[receiver_translucent, 1]
    rhs = np.concatenate([first, sensor_first]) + far_faces
    indptr, faces, values = counts
    radiosities, steps = solver.mgcr(rhs, indptr, faces, values, receiver_faces, receiver_translucent,
                                     np.concatenate([reflectance, np.ones(sensor_count)]),
                                     np.concatenate([transmittance, np.zeros(sensor_count)]),
                                     nearfield.form_factor(resolution), tolerance, 100, iterations)
    if "face_irradiance" in corrections:  # the near field's and the far field's light on each face
        incident = incident + solver.incoming(radiosities, indptr, faces, values, receiver_faces,
                                              receiver_translucent, nearfield.form_factor(resolution), n_faces)
        incident[first_face] += far_incoming(level_weights, profile, 0)[:len(kept)]
        lower = translucent.nonzero()[0]
        incident[first_face[lower] + 1] += far_incoming(level_weights, profile, 2)[lower]
    result = direct.results(labels, accepted, points, surfaces, radiosities[:n_faces], optics,
                            incident if "face_irradiance" in corrections else None)
    result.update(first_order=first, far_field=far_faces[:n_faces], profile=profile, structure=structure,
                  counts=counts, weights=level_weights, steps=steps, radiosity=radiosities[:n_faces])
    if sensors is not None:  # canestrad adds the far field to B0 in place: its sensors' "first order" includes it
        result["sensors"] = {"Ei0": rhs[n_faces:], "Ei": radiosities[n_faces:], "area": sensor_surfaces,
                             "first_order": sensor_first}
    return result


def far_incoming(weights, profile, disc):
    """Each receiver's irradiance from the far field through its upper (*disc* 0) or lower (2) disc: its levels'
    *weights* (receivers, 4, levels) times the profile's E− and E+."""
    weights = np.asarray(weights, dtype=np.float64)
    return (weights[:, disc] @ np.asarray(profile["downward"], dtype=np.float64)
            + weights[:, disc + 1] @ np.asarray(profile["upward"], dtype=np.float64))


def reused_far_field(weights, receiver_optics, receiver_translucent, profile, lights, precision="legacy"):
    """hd_calc_Bfar: each receiver's far field from its levels' *weights* (another band's or call's run) and this
    band's profile and optics: (receivers, 2). While *precision* is the legacy's, the weights are canestrad's
    float32 and its scaling by the last source's energy (Eclt) is kept with its index slip: a translucent
    receiver's faces are scaled, an opaque one's only when the next receiver is opaque too (the slip scales the
    previous face; the last receiver is never scaled)."""
    legacy = precision == "legacy"
    energy = float(lights[-1][0]) if (legacy and len(lights)) else 1.
    weights = np.asarray(weights).astype(np.float32).astype(np.float64) if legacy else np.asarray(weights)
    return _reused(weights, np.asarray(receiver_optics, dtype=np.float64), np.asarray(receiver_translucent),
                   np.asarray(profile["downward"], dtype=np.float64), np.asarray(profile["upward"], dtype=np.float64),
                   energy)


@njit(cache=True, error_model="numpy")
def _reused(weights, receiver_optics, receiver_translucent, downward, upward, energy):
    count = weights.shape[0]
    flat = np.zeros(2 * count + 1)  # Cenv by face, after a slot for the slip's write before the first
    index = 1
    for r in range(count):
        rho_upper, tau_upper = receiver_optics[r, 0], receiver_optics[r, 1]
        rho_lower, tau_lower = receiver_optics[r, 2], receiver_optics[r, 3]
        for level in range(downward.shape[0]):
            flux = weights[r, 0, level] * downward[level] + weights[r, 1, level] * upward[level]
            flat[index] += rho_upper * flux
            if receiver_translucent[r]:
                flat[index + 1] += tau_upper * flux
        if receiver_translucent[r]:
            for level in range(downward.shape[0]):
                flux = weights[r, 2, level] * downward[level] + weights[r, 3, level] * upward[level]
                flat[index] += tau_lower * flux
                flat[index + 1] += rho_lower * flux
            index += 1
            flat[index] *= energy
        flat[index - 1] *= energy
        index += 1
    far = np.zeros((count, 2))
    index = 1
    for r in range(count):
        far[r, 0] = flat[index]
        if receiver_translucent[r]:
            far[r, 1] = flat[index + 1]
            index += 1
        index += 1
    return far
