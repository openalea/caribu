"""
libcaribu's executables run on a scene given as arrays, in a temporary directory: the scene written as the legacy
writes it (formats), the outputs read back as text. The reference the port is compared with.

    etri = direct_etri(vertices, labels, lights, materials)       canestrad -1's Etri.vec0 body (no header)
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


def canestrad(files, arguments, outputs=("Etri.vec0",)) -> dict:
    """{output name: text} of canestrad run on *files* ({name: text}) with *arguments*."""
    with tempfile.TemporaryDirectory() as directory:
        for name, text in files.items():
            Path(directory, name).write_text(text)
        subprocess.run([executable("canestrad"), *arguments], cwd=directory, capture_output=True, check=True)
        return {name: Path(directory, name).read_text() for name in outputs if Path(directory, name).exists()}


def body(text) -> str:
    """A canestrad output without its `#` header lines."""
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def direct_etri(vertices, labels, lights, materials, soil_reflectance=-1, size=1536) -> str:
    """canestrad -1 (direct light) on the triangles: its Etri.vec0 body. *materials* {species: (rho,) | (rho, tau) |
    (rho_sup, tau_sup, rho_inf, tau_inf)}, *lights* [(E, (vx, vy, vz))]."""
    files = {"scene.can": formats.can_string(vertices, labels), "scene.light": formats.light_string(lights),
             "scene.opt": formats.opt_string(materials, soil_reflectance)}
    arguments = ["-M", "scene.can", "-l", "scene.light", "-p", "scene.opt", "-A", "-1", "-L", str(size)]
    return body(canestrad(files, arguments)["Etri.vec0"])
