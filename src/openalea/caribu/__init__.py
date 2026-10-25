"""
Caribu in Python: light on a static canopy of triangles (direct light, radiosity, nested radiosity), ported from
libcaribu's C++ (the original frozen in openalea.caribu.legacy), the legacy's numerics reproduced first.

    labels        the triangles' labels (optical species, leaf or opaque)
    geometry      areas, normals, the triangles canestrad rejects
    formats       Caribu's files, read and written as the legacy does
    projection    the z-buffer: each triangle's sunlit area along a beam
    direct        first-order radiosities, direct.run: the direct light on a scene
    budgets       radiosities to the triangles' incident and absorbed light
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openalea.caribu")
except PackageNotFoundError:
    # package is not installed
    pass
