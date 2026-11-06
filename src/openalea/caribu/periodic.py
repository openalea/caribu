"""
The infinite (toric) canopy: a pattern (xmin, ymin, xmax, ymax) repeated in x and y, light entering it from every
side (doc/legacy.rst).

    points = periodise(points, pattern)          each triangle moved into the pattern by whole pattern lengths
                                                 (periodise), the scene lifted to z ≥ 0
    accepted, flags = border_flags(points, pattern)
                                                 canestrad's -8 reading: a triangle whose centre lies outside the
                                                 pattern rejected, the others flagged when across its x, y borders
    copies = border_copies(flags)                which shifted copies of a triangle projplan draws (+dx, +dy)
    tile(owner, depth, roof, ...)                the image of the pattern tiled over the screen (infinitise): a
                                                 pixel where a neighbouring pattern lies in front is in its shade

The legacy's numerics are kept: periodise's output written with 6 significant digits and read back as float32.
"""
import math

import numpy as np

from . import kernels
from .kernels import _moved, _periodise_centres

F32 = np.float32


def periodise(points, pattern, corrections=frozenset()) -> np.ndarray:
    """The triangles (float32 (n, 3, 3), as canestrad reads them) as periodise moves them: each translated by whole
    pattern lengths so that its centre lies in (xmin, xmax] × (ymin, ymax], the whole scene raised to z ≥ 0 when it
    goes below, written with `%.6g` and read back (float32). Correction "periodise": the moved coordinates kept
    (float32), the scene lifted to exactly min z = 0."""
    return periodised(points, pattern, corrections).astype(F32)


def periodised(points, pattern, corrections=frozenset()) -> np.ndarray:
    """periodise's output as written, `%.6g` (float64, as s2v reads it)."""
    points = np.asarray(points, dtype=F32)
    low, high = pattern_bounds(pattern)
    bottom = F32(points[:, :, 2].min())
    top = F32(points[:, :, 2].max())
    if "periodise" in corrections:
        lift = -np.float64(bottom) if bottom < 0 else 0.
    else:
        bottom = F32(np.float64(bottom) - np.float64(F32(top - bottom)) / 100.0)
        lowest = (0.01 * np.float64(top) + np.float64(bottom)) / 1.01
        lift = -lowest if lowest < 0 else 0.
    lengths = np.array([np.float64(F32(high[axis] - low[axis])) for axis in range(2)])
    moved = _moved(points, _periodise_centres(points), low.astype(np.float64), high.astype(np.float64), lengths,
                   lift)
    if "periodise" in corrections:
        return moved.astype(F32).astype(np.float64)
    values = moved.reshape(-1)
    text = ("%.6g " * len(values)) % tuple(values.tolist())  # one formatting call, then one parse
    return np.array(text.split(), dtype=np.float64).reshape(points.shape)


def pattern_bounds(pattern):
    """(low, high) float32 (2,) of a pattern (x1, y1, x2, y2), as the legacy writes it in its `.8` file and reads it."""
    x1, y1, x2, y2 = (F32(float(str(value))) for value in pattern)
    return np.array([min(x1, x2), min(y1, y2)], dtype=F32), np.array([max(x1, x2), max(y1, y2)], dtype=F32)


def border_flags(points, pattern):
    """(accepted (n,) bool, flags (n,) int): canestrad's reading of an infinite scene. A triangle is kept when its
    centre lies in (xmin, xmax] × (ymin, ymax]; its flag is 1 when a vertex lies beyond the pattern in x, plus 2 in y
    (Polygone::nb_in: vertices beyond either side, counted with signs, so that one across both sides of an axis may
    count 0)."""
    low, high = pattern_bounds(pattern)
    centres = kernels.legacy_centres(points)
    accepted = ((centres[:, 0] > low[0]) & (centres[:, 1] > low[1]) & (centres[:, 0] <= high[0])
                & (centres[:, 1] <= high[1]))
    flags = np.zeros(len(points), dtype=np.int64)
    for axis, weight in ((0, 1), (1, 2)):
        beyond = (np.sum(points[:, :, axis] > high[axis], axis=1) - np.sum(points[:, :, axis] < low[axis], axis=1))
        flags += np.where(beyond != 0, weight, 0)
    return accepted, flags


def border_copies(flags) -> np.ndarray:
    """(n, 3) bool: the copies of each triangle projplan draws, (original, +dx, +dy). Flag 0: the original; 1: and
    +dx; 2: and +dy. A triangle across both borders (flag 3) is not in projplan's switch: it gets the copies count of
    the triangle before it, without the skip of +dx (the legacy's behaviour, kept)."""
    copies = np.zeros((len(flags), 3), dtype=bool)
    last = 0
    for t, flag in enumerate(flags):
        if flag in (0, 1):
            last = int(flag)
        elif flag in (2, 4):
            last = 2
        for index in range(last + 1):
            if not (index == 1 and flag == 2):
                copies[t, index] = True
    return copies


def border_shifts(vmin, vmax) -> np.ndarray:
    """(3, 3) float32: projplan's shifts of the copies, none, +dx, +dy."""
    shifts = np.zeros((3, 3), dtype=F32)
    shifts[1, 0] = F32(vmax[0] - vmin[0])
    shifts[2, 1] = F32(vmax[1] - vmin[1])
    return shifts


def copy_shifts(points, flags, pattern, corrected=False):
    """(shifts (n, 9, 3) float32, counts (n,) int): each triangle's copies to draw, in order, the original first.
    The legacy's (border_copies, border_shifts): the original, +dx, +dy. *corrected* (correction "border_copies"):
    a copy on each side the triangle crosses, −dx beyond xmax and +dx below xmin (the same in y), and the corner's
    when it crosses both axes, so that its parts beyond the pattern are drawn where the pattern repeats them."""
    low, high = pattern_bounds(pattern)
    length = (F32(high[0] - low[0]), F32(high[1] - low[1]))
    shifts = np.zeros((len(points), 9, 3), dtype=F32)
    counts = np.ones(len(points), dtype=np.int64)
    if not corrected:
        legacy = border_shifts(np.r_[low, F32(0)], np.r_[high, F32(0)])
        for t, copies in enumerate(border_copies(flags)):
            shifts[t, :int(copies.sum())] = legacy[copies]
            counts[t] = int(copies.sum())
        return shifts, counts
    beyond = points[:, :, :2] > high[None, None, :]
    below = points[:, :, :2] < low[None, None, :]
    for t in range(len(points)):
        moves = [[F32(0)], [F32(0)]]
        for axis in range(2):
            if beyond[t, :, axis].any():
                moves[axis].append(F32(-length[axis]))
            if below[t, :, axis].any():
                moves[axis].append(length[axis])
        ordered = [(dx, dy) for dx in moves[0][1:] for dy in (F32(0),)] + [(F32(0), dy) for dy in moves[1][1:]]
        ordered += [(dx, dy) for dx in moves[0][1:] for dy in moves[1][1:]]
        for index, (dx, dy) in enumerate([(F32(0), F32(0))] + ordered):
            shifts[t, index, 0], shifts[t, index, 1] = dx, dy
        counts[t] = 1 + len(ordered)
    return shifts, counts


def _c_round(value):
    """C's round: half away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def tiling_steps(roof) -> np.ndarray:
    """infinitise's tiling steps (8,) int in pixels from the pattern's top corners *roof* ((4, 2) int, in the image):
    tr[0:2] the pattern's y side, tr[2:4] its x side, then their opposites (each the mean of two opposite edges,
    rounded as C's round)."""
    steps = np.zeros(8, dtype=np.int64)
    r = [[int(value) for value in row] for row in roof]

    def side(a, b, c, d, axis):  # (a − b) + (c − d) in C's int, then halved in double and rounded
        return _c_round(_int32(_int32(r[a][axis] - r[b][axis]) + _int32(r[c][axis] - r[d][axis])) / 2.)
    steps[0] = side(3, 0, 2, 1, 0)
    steps[1] = side(3, 0, 2, 1, 1)
    steps[2] = side(1, 0, 2, 3, 0)
    steps[3] = side(1, 0, 2, 3, 1)
    steps[4], steps[5], steps[6], steps[7] = -steps[2], -steps[3], -steps[0], -steps[1]
    return steps


def lattice_translations(pattern, v, w, u, du, dv, size):
    """The correction "tiling": ((k, 2) int pixel shifts, (k,) depth offsets) of the neighbouring patterns' images
    overlapping the screen, the untranslated left out. Each translate a·(Lx, 0, 0) + b·(0, Ly, 0) of the lattice is
    placed by its exact image vector, rounded once (infinitise rounds the steps, then repeats them), its depth offset
    u · translation exact (infinitise's tan θ · dv / T per row)."""
    low, high = pattern_bounds(pattern)
    lengths = np.float64(high[0] - low[0]), np.float64(high[1] - low[1])
    scales = (size - 1) / np.float64(du), (size - 1) / np.float64(dv)
    vectors, depths = [], []
    for axis in range(2):
        translation = np.zeros(3)
        translation[axis] = lengths[axis]
        vectors.append((np.dot(np.asarray(v, np.float64), translation) * scales[0],
                        np.dot(np.asarray(w, np.float64), translation) * scales[1]))
        depths.append(np.dot(np.asarray(u, np.float64), translation))
    shortest = min(math.hypot(*vectors[0]), math.hypot(*vectors[1]))
    # the image of the pattern's lattice cell has an area det(vectors): a translate (a, b) within the screen needs
    # |a|, |b| up to the screen's diagonal over the cell's smallest height
    determinant = abs(vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0])
    height = determinant / max(math.hypot(*vectors[0]), math.hypot(*vectors[1]), 1e-300)
    reach = int(math.ceil(2 * size / max(min(height, shortest), 1e-9))) + 1
    shifts, offsets = [], []
    for a in range(-reach, reach + 1):
        for b in range(-reach, reach + 1):
            if a == 0 and b == 0:
                continue
            x = a * vectors[0][0] + b * vectors[1][0]
            y = a * vectors[0][1] + b * vectors[1][1]
            if abs(x) < size and abs(y) < size:
                shifts.append((int(round(x)), int(round(y))))
                offsets.append(a * depths[0] + b * depths[1])
    return np.array(shifts, dtype=np.int64).reshape(-1, 2), np.array(offsets, dtype=np.float64)


def roof_pixels(vmin, vmax, origin, v, w, u, du, dv, size) -> np.ndarray:
    """(4, 2) int: the pattern's corners at the scene's top, (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), in
    image pixels as projplan computes them (float32 image coordinates times the size, divided by the screen's
    extents, truncated)."""
    from .kernels import _to_image
    corners = [(vmin[0], vmin[1]), (vmax[0], vmin[1]), (vmax[0], vmax[1]), (vmin[0], vmax[1])]
    roof = np.zeros((4, 2), dtype=np.int64)
    zero = np.zeros(3, dtype=F32)
    for index, (x, y) in enumerate(corners):
        image_x, image_y, _ = _to_image(F32(x), F32(y), F32(vmax[2]), zero, origin, v, w, u)
        with np.errstate(divide="ignore", invalid="ignore"):
            roof[index, 0] = _c_int(np.float64(F32(image_x * F32(size))) / np.float64(du))
            roof[index, 1] = _c_int(np.float64(F32(image_y * F32(size))) / np.float64(dv))
    return roof


def _c_int(value):
    """C's (int) of a double on x86-64: truncated, INT_MIN when NaN or beyond the int range (a horizontal beam)."""
    if not math.isfinite(value) or not -2 ** 31 <= value < 2 ** 31:
        return -2 ** 31
    return int(value)


def _int32(value):
    """An int's 32-bit wrap (C's int arithmetic on x86-64)."""
    return (int(value) + 2 ** 31) % 2 ** 32 - 2 ** 31


def distance_step(direction_z, dv, size) -> float:
    """infinitise's depth increase per pixel of image row shift: tan(θ) dv / T (θ the beam's zenith angle)."""
    cosine = -np.float64(direction_z)
    angle = (0. if cosine > 0 else math.pi) if abs(cosine) > 1. else math.acos(cosine)
    return math.tan(angle) * dv / float(size)
