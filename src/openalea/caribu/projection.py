"""
Direct light: for each source, the scene seen along the beam on an orthographic z-buffer of T × T pixels; the pixels
each triangle owns give its sunlit area (canestrad's Canopy::projplan and colorie_triangle; doc/legacy.rst).

The legacy's numerics are reproduced, to compare pixel counts exactly: points and vectors in float32 (canestrad's
`reel`), its locals in double, the same operation order; the screen on the scene's bounds, lifted one length unit
above it; a pixel scale of (T − 1)/extent; a pixel owned by the triangle whose centre is nearest along the beam, the
first drawn winning ties; the sunlit (horizontal-equivalent) area of a pixel `Apix = du dv / T² / cos θ`.

    screen = legacy_screen(vmin, vmax, direction)
    owner = rasterise(points, screen, size)                   (T, T) triangle index, -1 where no triangle
    areas = projected_areas(owner, ...)                       per face, +Apix lit, -Apix for a leaf's other face
"""

import numpy as np

F32 = np.float32


def pattern_cell(origin, v, w, u, du, dv, size, low, high) -> np.ndarray:
    """(CELL,) the pattern as a cell of the infinite canopy's lattice, for the pixel counts of the correction
    "border_copies": a pixel counts for a triangle only when its visible point lies in [low − h, high − h) in x and
    y, h half a pixel's pitch (a half-open cell: each point of the periodic canopy counted once, whatever the beam;
    under a vertical beam, the screen's last row and column, which repeat its first ones, left out)."""
    half = 0.5 * min(du, dv) / (size - 1)
    return np.concatenate([[1.], origin, v, w, u, [(size - 1) / du, (size - 1) / dv],
                           [low[0] - half, low[1] - half, high[0] - half, high[1] - half]]).astype(np.float64)


