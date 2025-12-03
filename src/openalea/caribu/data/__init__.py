from importlib.resources import files


def get_path():
    """Return a pathlib.Path representing the data directory."""
    return files(__package__)


def get(filename: str):
    return files(__package__) / filename