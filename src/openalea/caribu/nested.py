"""
Light in an infinite canopy with scattering (caribu's mixed radiosity: canestrad -8 -d D -e band.env after
periodise, s2v and mcsail). The far field is the canopy as a turbid medium (sail): its layers' fluxes. With a null
sphere (D = 0, pure SAIL), each triangle's scattered light is the first scattering of the far field's fluxes at its
height (canestrad's sail_pur), added to its direct light; with D > 0, the triangles within the sphere exchange
light by radiosity, the far field lighting what they do not see of each other (nested radiosity, not yet ported).

    result = run(vertices, labels, lights, optics, pattern, layers=10, height=1.)    as direct.run's result, B0
                                                                                      plus the far field's light
"""
import numpy as np

from . import direct, geometry, sail
from .labels import is_translucent


def far_field_profile(points, labels, lights, optics, pattern, layers, height, precision="legacy") -> tuple:
    """The far field of the periodised *points*: (profile, structure), s2v's layers and mcsail's fluxes from the
    soil up, each read back with its file's printed digits when *precision* is the legacy's."""
    structure = sail.layers(points, labels, layers, height, pattern, [optics])
    read = sail.printed_layers(structure) if precision == "legacy" else structure
    profile = sail.profiles(read["frequencies"][::-1], read["leaf_area_index"][::-1], read["reflectance"][0][::-1],
                            read["transmittance"][0][::-1], read["soil_reflectance"][0], read["thickness"], lights)
    if precision == "legacy":
        profile.update(sail.printed_profile(profile, read["thickness"]))
    else:
        profile["thickness"] = read["thickness"]
    return profile, structure


def run(vertices, labels, lights, optics, pattern, layers=10, height=1., diameter=0., size=1536, sensors=None,
        workers=None, precision="legacy"):
    """Direct light and its scattering in the infinite canopy repeating *pattern* (x1, y1, x2, y2), the far field
    in *layers* up to *height*: direct.run's result, its budgets from B0 + B_far; result["profile"] the far field's
    fluxes, result["first_order"] B0, result["far_field"] B_far. *diameter*: the near field's sphere (0: pure
    SAIL). *precision* "legacy": the intermediate files' printed digits (s2v's, mcsail's), as canestrad reads them."""
    if diameter != 0:
        raise NotImplementedError("nested radiosity (a sphere diameter > 0) is not yet ported")
    points, labels, accepted, flags = direct.scene(vertices, labels, pattern)
    kept_points, kept = points[accepted], labels[accepted]
    profile, structure = far_field_profile(points, labels, lights, optics, pattern, layers, height, precision)
    sensor_points = None if sensors is None else geometry.as_legacy_vertices(sensors)
    first, surfaces, sensor_first, sensor_surfaces = direct.first_order(
        kept_points, kept, lights if accepted.any() else [], optics, size, sensors=sensor_points, workers=workers,
        pattern=pattern, flags=flags)
    translucent = is_translucent(kept)
    first_face, n_faces = direct.face_numbering(translucent)
    reflectance, transmittance = direct.face_optics(kept, optics)
    far = sail.far_field(geometry.legacy_centres(kept_points), geometry.legacy_normals(kept_points), translucent,
                         first_face, n_faces, reflectance, transmittance, profile)
    result = direct.results(labels, accepted, points, surfaces, first + far, optics)
    result.update(first_order=first, far_field=far, profile=profile, structure=structure)
    if sensors is not None:
        count = len(sensor_points)
        sensor_far = sail.far_field(geometry.legacy_centres(sensor_points), geometry.legacy_normals(sensor_points),
                                    np.zeros(count, dtype=bool), np.arange(count), count, np.ones(count),
                                    np.zeros(count), profile)
        result["sensors"] = {"Ei0": sensor_first, "Ei": sensor_first + sensor_far, "area": sensor_surfaces}
    return result
