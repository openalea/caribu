from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openalea.caribu")
except PackageNotFoundError:
    # package is not installed
    pass
