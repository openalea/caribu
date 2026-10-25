"""
libcaribu's executables run on a scene given as arrays, in a temporary directory: the scene written as the legacy
writes it (formats), the outputs read back as text. The reference the port is compared with.

    etri = direct_etri(vertices, labels, lights, materials)       canestrad -1's Etri.vec0 body (no header)
    outputs = direct(..., sensors=sensors)                         and its solem.dat (the sensors)
    outputs = radiosity(vertices, labels, lights, materials)       canestrad -d -1 (every order of scattering)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import formats


def executable(name) -> str:
    """The path of libcaribu's executable *name* (on PATH, or next to this Python), None if not installed."""
    return shutil.which(name, path=os.pathsep.join([os.environ.get("PATH", ""), str(Path(sys.executable).parent)]))


def available() -> bool:
    """Whether libcaribu's executables are installed."""
    return executable("canestrad") is not None


def run(steps, files, outputs=("Etri.vec0",)) -> dict:
    """{output name: text} of libcaribu's executables run one after the other on *files* ({name: text}) in a
    temporary directory; *steps* [(executable name, arguments)]."""
    with tempfile.TemporaryDirectory() as directory:
        for name, text in files.items():
            Path(directory, name).write_text(text)
        for name, arguments in steps:
            subprocess.run([executable(name), *arguments], cwd=directory, capture_output=True, check=True)
        return {name: Path(directory, name).read_text() for name in outputs if Path(directory, name).exists()}


def body(text) -> str:
    """A canestrad output without its `#` header lines."""
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def direct(vertices, labels, lights, materials, soil_reflectance=-1, size=1536, sensors=None, pattern=None) -> dict:
    """canestrad -1 (direct light) on the triangles: {"Etri.vec0": its body, "solem.dat": the sensors' rows (with
    *sensors*)}. *materials* {species: (rho,) | (rho, tau) | (rho_sup, tau_sup, rho_inf, tau_inf)}, *lights*
    [(E, (vx, vy, vz))], *sensors* (m, 3, 3), *pattern* (x1, y1, x2, y2) an infinite canopy (periodise, then
    canestrad -8, as caribu runs them)."""
    files = {"scene.can": formats.can_string(vertices, labels), "scene.light": formats.light_string(lights),
             "scene.opt": formats.opt_string(materials, soil_reflectance)}
    arguments = ["-M", "scene.can", "-l", "scene.light", "-p", "scene.opt", "-A", "-1", "-L", str(size)]
    steps = []
    if pattern is not None:
        files["pattern.8"] = formats.pattern_string(pattern)
        steps.append(("periodise", ["-m", "scene.can", "-8", "pattern.8", "-o", "scene_8.can"]))
        arguments[1] = "scene_8.can"
        arguments += ["-8", "pattern.8"]
    if sensors is not None:
        files["sensors.can"] = formats.sensor_string(sensors)
        arguments += ["-C", "sensors.can"]
    outputs = run(steps + [("canestrad", arguments)], files, ("Etri.vec0", "solem.dat", "scene_8.can"))
    outputs["Etri.vec0"] = body(outputs["Etri.vec0"])
    return outputs


def radiosity(vertices, labels, lights, materials, soil_reflectance=-1, size=1536, sensors=None) -> dict:
    """canestrad -d -1 (radiosity in the whole scene, caribu's radiosity), as direct's outputs; its first order and
    solved radiosities ("E0.dat", "B.dat", one value per face) too."""
    files = {"scene.can": formats.can_string(vertices, labels), "scene.light": formats.light_string(lights),
             "scene.opt": formats.opt_string(materials, soil_reflectance)}
    arguments = ["-M", "scene.can", "-l", "scene.light", "-p", "scene.opt", "-A", "-d", "-1", "-f", "form_factors",
                 "-L", str(size)]
    if sensors is not None:
        files["sensors.can"] = formats.sensor_string(sensors)
        arguments += ["-C", "sensors.can"]
    outputs = run([("canestrad", arguments)], files, ("Etri.vec0", "solem.dat", "E0.dat", "B.dat"))
    outputs["Etri.vec0"] = body(outputs["Etri.vec0"])
    return outputs


def direct_etri(vertices, labels, lights, materials, soil_reflectance=-1, size=1536) -> str:
    """direct's Etri.vec0 body."""
    return direct(vertices, labels, lights, materials, soil_reflectance, size)["Etri.vec0"]
