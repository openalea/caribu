"""
A static canopy prepared for repeated lighting (a FunctionalComponent's state): what depends only on its geometry
is computed at its first use and kept, so that a new sky, new energies, other bands or optics cost little.

    canopy = Canopy(vertices, labels, pattern=None, sensors=None)
    canopy.direct(lights, optics)                                 as direct.run
    canopy.radiosity(lights, optics)                              as radiosity.run (full radiosity)
    canopy.nested(lights, optics, layers, height, diameter)       as nested.run (an infinite canopy)

Kept: each light direction's lit areas (exact: the same projections, summed in the same order); the full
radiosity's form factors (exact); in nested radiosity, the near field's form factors and the far field's levels'
weights, reused only when *precision* is "full" (the far field then summed over the levels instead of the pixels: a
different rounding), the legacy's precision recomputing them at each call as canestrad does.
"""
from . import corrections as corrections_module
from . import direct, nested, radiosity


class Canopy:
    """A static scene: *vertices* (n, 3, 3), *labels* (n,), *pattern* (x1, y1, x2, y2) for an infinite canopy,
    *sensors* (m, 3, 3); *size* the projections' screen (pixels), *workers* the threads; *device* "cuda": the near
    field on the GPU (gpu.available()); *corrections*: the legacy's behaviours corrected (corrections.resolve:
    "recommended", names, or none), the same for every call."""

    def __init__(self, vertices, labels, pattern=None, sensors=None, size=1536, workers=None, device="cpu",
                 corrections=None):
        self.vertices, self.labels, self.pattern, self.sensors = vertices, labels, pattern, sensors
        self.size, self.workers, self.device = size, workers, device
        self.corrections = corrections_module.resolve(corrections)
        # the triangles as canestrad reads them
        self.prepared = direct.scene(vertices, labels, pattern, self.corrections)
        self._lit = {}            # direction → lit areas
        self._form_factors = {}   # disc resolution → the full radiosity's counts
        self._near_field = {}     # (layers, height, diameter, bias, resolution, threshold, ratio) → (counts, weights)

    def direct(self, lights, optics) -> dict:
        """Direct light (canestrad -1, or -8 with a pattern): direct.run's result."""
        return direct.run(self.vertices, self.labels, lights, optics, self.size, self.sensors, self.workers,
                          self.pattern, cache=self._lit, prepared=self.prepared, corrections=self.corrections)

    def radiosity(self, lights, optics, resolution=52, tolerance=1e-6, iterations=1000) -> dict:
        """Radiosity in the whole scene (not an infinite canopy): radiosity.run's result."""
        if self.pattern is not None:
            raise ValueError("full radiosity is for a closed scene: use nested() in an infinite canopy")
        result = radiosity.run(self.vertices, self.labels, lights, optics, self.size, self.sensors, self.workers,
                               resolution, tolerance, iterations, counts=self._form_factors.get(resolution),
                               cache=self._lit, prepared=self.prepared, device=self.device,
                               corrections=self.corrections)
        self._form_factors[resolution] = result["counts"]
        return result

    def nested(self, lights, optics, layers=10, height=1., diameter=0., precision="legacy", bias=True,
               resolution=52, tolerance=1e-6, iterations=1000, threshold=50, ratio=1.) -> dict:
        """Scattering in the infinite canopy (pure SAIL with a null *diameter*): nested.run's result."""
        if self.pattern is None:
            raise ValueError("nested radiosity needs a pattern (an infinite canopy)")
        corrections_module.resolve(self.corrections, precision)          # none with the legacy's precision
        key = (layers, float(height), float(diameter), bias, resolution, threshold, ratio)  # the weights' levels too
        reused = self._near_field.get(key) if precision == "full" else None
        result = nested.run(self.vertices, self.labels, lights, optics, self.pattern, layers, height, diameter,
                            self.size, self.sensors, self.workers, precision, bias, resolution, tolerance,
                            iterations, threshold, ratio, weights=reused, cache=self._lit, prepared=self.prepared,
                            device=self.device, corrections=self.corrections)
        if diameter != 0 and reused is None:
            self._near_field[key] = (result["counts"], result["weights"])
        return result
