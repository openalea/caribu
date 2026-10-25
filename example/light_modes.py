"""
The light models on the 100-plant Adel scene under the 46-direction turtle sky (unit energy), in an infinite
canopy: direct light only; pure SAIL (the first scattering of a turbid medium's fluxes); nested radiosity with
spheres of 0.25 and 0.5 m (radiosity among neighbours, the turbid medium beyond). Printed: the canopy's absorption
per m² of ground, the part scattering adds, the time; and, when libcaribu is installed, the legacy's time.

    python example/light_modes.py
"""
import time

import numpy as np

from openalea.caribu import direct, formats, lights, nested
from openalea.caribu.comparison import legacy_run, port_run
from openalea.caribu.legacy.data_samples import data_path

MATERIALS = {1: (0.10, 0.05)}               # leaves: ρ, τ (the legacy's description of the optics)
SOIL = 0.15
vertices, labels = formats.read_can_arrays(data_path("f331s1_100plantes.can"))
flat = vertices.reshape(-1, 3)
pattern = (float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 0].max()), float(flat[:, 1].max()))
ground = (pattern[2] - pattern[0]) * (pattern[3] - pattern[1])
height = float(flat[:, 2].max())
sky = lights.turtle_lights("46", "soc", 1.)
optics = port_run.optics(MATERIALS, SOIL)
nested.run(vertices[:300], labels[:300], sky[:2], optics, pattern, 3, height, 0.5)  # compile once


def absorbed(result):
    """The canopy's absorbed light per m² of ground."""
    return float(np.nansum(result["Eabs"].astype(np.float64) * result["area"])) / ground


rows = []
for name, run, legacy in (
        ("direct", lambda: direct.run(vertices, labels, sky, optics, pattern=pattern),
         lambda: legacy_run.direct(vertices, labels, sky, MATERIALS, SOIL, pattern=pattern)),
        ("pure SAIL", lambda: nested.run(vertices, labels, sky, optics, pattern, 10, height, 0),
         lambda: legacy_run.layered(vertices, labels, sky, MATERIALS, pattern, 10, height, SOIL)),
        ("nested 0.25 m", lambda: nested.run(vertices, labels, sky, optics, pattern, 10, height, 0.25),
         lambda: legacy_run.layered(vertices, labels, sky, MATERIALS, pattern, 10, height, SOIL, diameter=0.25)),
        ("nested 0.5 m", lambda: nested.run(vertices, labels, sky, optics, pattern, 10, height, 0.5),
         lambda: legacy_run.layered(vertices, labels, sky, MATERIALS, pattern, 10, height, SOIL, diameter=0.5))):
    started = time.perf_counter()
    result = run()
    elapsed = time.perf_counter() - started
    legacy_time = None
    if legacy_run.available():
        started = time.perf_counter()
        legacy()
        legacy_time = time.perf_counter() - started
    rows.append((name, absorbed(result), elapsed, legacy_time))

first = rows[0][1]
print(f"{'model':<14} {'absorbed':>9} {'scattering':>11} {'port':>8} {'legacy':>8}")
for name, value, elapsed, legacy_time in rows:
    print(f"{name:<14} {value:9.4f} {value - first:+11.4f} {elapsed:7.2f}s "
          + (f"{legacy_time:7.2f}s" if legacy_time is not None else "       -"))
