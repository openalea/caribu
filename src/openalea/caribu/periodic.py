"""
The infinite (toric) canopy: a pattern (xmin, ymin, xmax, ymax) repeated in x and y, light entering it from every
side (B.5 and B.6 of the description).

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
from numba import njit

from . import geometry

F32 = np.float32


def periodise(points, pattern) -> np.ndarray:
    """The triangles (float32 (n, 3, 3), as canestrad reads them) as periodise moves them: each translated by whole
    pattern lengths so that its centre lies in (xmin, xmax] × (ymin, ymax], the whole scene raised to z ≥ 0 when it
    goes below, written with `%.6g` and read back (float32)."""
    return periodised(points, pattern).astype(F32)


def periodised(points, pattern) -> np.ndarray:
    """periodise's output as written, `%.6g` (float64, as s2v reads it)."""
    points = np.asarray(points, dtype=F32)
    low, high = pattern_bounds(pattern)
    bottom = F32(points[:, :, 2].min())
    top = F32(points[:, :, 2].max())
    bottom = F32(np.float64(bottom) - np.float64(F32(top - bottom)) / 100.0)
    lowest = (0.01 * np.float64(top) + np.float64(bottom)) / 1.01
    lift = -lowest if lowest < 0 else 0.
    lengths = np.array([np.float64(F32(high[axis] - low[axis])) for axis in range(2)])
    moved = _moved(points, _periodise_centres(points), low.astype(np.float64), high.astype(np.float64), lengths,
                   lift)
    values = moved.reshape(-1)
    text = ("%.6g " * len(values)) % tuple(values.tolist())  # one formatting call, then one parse
    return np.array(text.split(), dtype=np.float64).reshape(points.shape)


@njit(cache=True, nogil=True)
def _moved(points, centres, low, high, lengths, lift):
    """Each triangle translated by whole pattern lengths, its centre into (low, high] (periodise's walk, its
    comparisons in double), the scene raised by *lift* (float64)."""
    moved = np.empty(points.shape, dtype=np.float64)
    for t in range(points.shape[0]):
        shift = np.zeros(2)
        for axis in range(2):
            centre, m = np.float64(centres[t, axis]), 0
            while centre <= low[axis] + m * lengths[axis]:
                m -= 1
            while centre > high[axis] + m * lengths[axis]:
                m += 1
            shift[axis] = -m * lengths[axis]
        for vertex in range(3):
            moved[t, vertex, 0] = np.float64(points[t, vertex, 0]) + shift[0]
            moved[t, vertex, 1] = np.float64(points[t, vertex, 1]) + shift[1]
            moved[t, vertex, 2] = np.float64(points[t, vertex, 2]) + lift
    return moved


def pattern_bounds(pattern):
    """(low, high) float32 (2,) of a pattern (x1, y1, x2, y2), as the legacy writes it in its `.8` file and reads it."""
    x1, y1, x2, y2 = (F32(float(str(value))) for value in pattern)
    return np.array([min(x1, x2), min(y1, y2)], dtype=F32), np.array([max(x1, x2), max(y1, y2)], dtype=F32)


@njit(cache=True, nogil=True)
def _periodise_centres(points):
    """periodise's Polygone::centre: float32 sums, divided by 3 in double, stored as float32."""
    centre = np.zeros((points.shape[0], 3), dtype=np.float32)
    for t in range(points.shape[0]):
        for q in range(3):
            total = F32(F32(F32(F32(0.) + points[t, 0, q]) + points[t, 1, q]) + points[t, 2, q])
            centre[t, q] = F32(np.float64(total) / 3.0)
    return centre


def border_flags(points, pattern):
    """(accepted (n,) bool, flags (n,) int): canestrad's reading of an infinite scene. A triangle is kept when its
    centre lies in (xmin, xmax] × (ymin, ymax]; its flag is 1 when a vertex lies beyond the pattern in x, plus 2 in y
    (Polygone::nb_in: vertices beyond either side, counted with signs, so that one across both sides of an axis may
    count 0)."""
    low, high = pattern_bounds(pattern)
    centres = geometry.legacy_centres(points)
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


def _c_round(value):
    """C's round: half away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def tiling_steps(roof) -> np.ndarray:
    """infinitise's tiling steps (8,) int in pixels from the pattern's top corners *roof* ((4, 2) int, in the image):
    tr[0:2] the pattern's y side, tr[2:4] its x side, then their opposites (each the mean of two opposite edges,
    rounded as C's round)."""
    steps = np.zeros(8, dtype=np.int64)
    steps[0] = _c_round(((roof[3, 0] - roof[0, 0]) + (roof[2, 0] - roof[1, 0])) / 2.)
    steps[1] = _c_round(((roof[3, 1] - roof[0, 1]) + (roof[2, 1] - roof[1, 1])) / 2.)
    steps[2] = _c_round(((roof[1, 0] - roof[0, 0]) + (roof[2, 0] - roof[3, 0])) / 2.)
    steps[3] = _c_round(((roof[1, 1] - roof[0, 1]) + (roof[2, 1] - roof[3, 1])) / 2.)
    steps[4], steps[5], steps[6], steps[7] = -steps[2], -steps[3], -steps[0], -steps[1]
    return steps


@njit(cache=True, nogil=True)
def _outside(x, y, step_x, step_y, size):
    """infinitise's out(): the walk stops once the translated image has left the screen along each axis it moves on."""
    return (step_x == 0 or abs(x) >= size) and (step_y == 0 or abs(y) >= size)


@njit(cache=True, nogil=True)
def translations(steps, size):
    """(k, 2) int: the image translations infinitise's walk (pave, lateral) visits that overlap the screen, the
    untranslated image left out. A pave walk goes along tr[0:2] (and from its first step back along tr[6:8]), at each
    stop a lateral walk each way along tr[2:4]."""
    found = []
    for start_x, start_y, direction in ((0, 0, 0), (steps[6], steps[7], 6)):
        x, y = start_x, start_y
        while not _outside(x, y, steps[direction], steps[direction + 1], size):
            if not (x == 0 and y == 0) and abs(x) < size and abs(y) < size:
                found.append((x, y))
            for side in (2, 4):
                lateral_x, lateral_y = x + steps[side], y + steps[side + 1]
                while not _outside(lateral_x, lateral_y, steps[side], steps[side + 1], size):
                    if abs(lateral_x) < size and abs(lateral_y) < size:
                        found.append((lateral_x, lateral_y))
                    lateral_x += steps[side]
                    lateral_y += steps[side + 1]
            x += steps[direction]
            y += steps[direction + 1]
    result = np.zeros((len(found), 2), dtype=np.int64)
    for index in range(len(found)):
        result[index, 0], result[index, 1] = found[index]
    return result


@njit(cache=True, nogil=True)
def tile(owner, depth, shifts, distance_step):
    """infinitise without duplication: the image translated by each of *shifts* ((k, 2) pixels), its depth increased
    by *distance_step* per pixel of the second shift; where a translated pixel lies in front, the pixel is in a
    neighbouring pattern's shade (owner −1) and takes its depth. In place."""
    size = depth.shape[0]
    original = depth.copy()
    # a pixel ends in the shade when any translate lies in front of the original image, whatever their order: the
    # rows are walked in memory order, each translate's row in turn
    for k in range(size):
        for index in range(shifts.shape[0]):
            i, j = shifts[index, 0], shifts[index, 1]
            if k - i < 0 or k - i >= size:
                continue
            offset = distance_step * j
            source = original[k - i]
            target, owners = depth[k], owner[k]
            for l in range(max(j, 0), size + min(j, 0)):
                value = np.float64(source[l - j]) + offset
                if value < target[l]:
                    target[l] = F32(value)
                    owners[l] = -1


def roof_pixels(vmin, vmax, origin, v, w, u, du, dv, size) -> np.ndarray:
    """(4, 2) int: the pattern's corners at the scene's top, (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), in
    image pixels as projplan computes them (float32 image coordinates times the size, divided by the screen's
    extents, truncated)."""
    from .projection import _to_image
    corners = [(vmin[0], vmin[1]), (vmax[0], vmin[1]), (vmax[0], vmax[1]), (vmin[0], vmax[1])]
    roof = np.zeros((4, 2), dtype=np.int64)
    zero = np.zeros(3, dtype=F32)
    for index, (x, y) in enumerate(corners):
        image_x, image_y, _ = _to_image(F32(x), F32(y), F32(vmax[2]), zero, origin, v, w, u)
        roof[index, 0] = int(np.float64(F32(image_x * F32(size))) / du)
        roof[index, 1] = int(np.float64(F32(image_y * F32(size))) / dv)
    return roof


def distance_step(direction_z, dv, size) -> float:
    """infinitise's depth increase per pixel of image row shift: tan(θ) dv / T (θ the beam's zenith angle)."""
    cosine = -np.float64(direction_z)
    angle = (0. if cosine > 0 else math.pi) if abs(cosine) > 1. else math.acos(cosine)
    return math.tan(angle) * dv / float(size)
