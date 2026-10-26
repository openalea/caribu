"""
Caribu in Python: light on a static canopy of triangles (direct light, radiosity, nested radiosity with a turbid
medium's far field), ported from libcaribu's C++ (the original frozen in openalea.caribu.legacy), the legacy's
numerics reproduced first, its behaviours corrected on demand.

    labels        the triangles' labels (optical species, leaf or opaque)
    geometry      areas, normals, the triangles canestrad rejects
    formats       Caribu's files, read and written as the legacy does; PlantGL scenes and MTGs
    lights        light sources: the turtle and sector skies, the sun, the diffuse fraction
    projection    the z-buffer: each triangle's sunlit area along a beam
    periodic      the infinite canopy: periodise, the border copies, the image's tiling
    direct        first-order radiosities, direct.run: the direct light on a scene
    nearfield     the near field's form factors: the receivers' hemispheres, the grid of voxels
    solver        the radiosity system, meschach's MGCR
    radiosity     radiosity.run: every order of scattering in the whole scene
    sail          the turbid medium: s2v's layers, mcsail's SAIL profiles, pure SAIL
    nested        nested.run: scattering in an infinite canopy (pure SAIL, nested radiosity)
    budgets       radiosities to the triangles' incident and absorbed light
    canopy        Canopy: a static scene lit repeatedly, its geometry's results kept
    gpu           the near field on CUDA
    api           the legacy's entry points (raycasting, radiosity, mixed_radiosity, CaribuScene, star)
    corrections   the corrections of the legacy's behaviours, by name ("recommended": all)
    component     CaribuLight, the metafspm component, and CaribuLightModel, its environment model
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openalea.caribu")
except PackageNotFoundError:
    # package is not installed
    pass
