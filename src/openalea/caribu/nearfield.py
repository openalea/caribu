"""
The near field of radiosity: what each receiver sees of its neighbours (canestrad's calc_FF_Bfar, proj_ortho and NFF,
C.2–C.5 of the description). Each receiver's hemispheres are drawn on an NB × NB disc (the Nusselt analogue: a
direction's orthographic projection on the receiver's plane), the upper one, and the lower one for a leaf; every
emitter within the sphere is projected, the nearest kept per pixel. A pixel holds the form factor dFF = 4 / (π NB²);
a receiver's row counts, per emitter face, its pixels in the upper disc minus those in the lower one.

    rows = counts(points, labels, receivers, ...)       per receiver (the triangles, then the sensors), the emitter
                                                         faces seen and their net pixel counts (CSR)

The legacy's numerics are kept (float32 points and frames, double locals, its edge cases), to compare the counts
with canestrad's own matrix files.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from . import geometry, periodic

F32 = np.float32
UPPER_EMPTY = np.float32(99999999999.9)
LOWER_EMPTY = np.float32(999999999999.9)


# --- The disc ---

def disc_tables(size=52):
    """(i2stab, rhotab) of init_NFF: each pixel's coordinate on [−1, 1], and the height √(1 − x² − y²) of the
    hemisphere over the pixels of one octant (−1 outside the disc), folded by symmetry."""
    half = int(size / 2.0)
    pixel = np.array([(2 * i + 1) / float(size) - 1.0 for i in range(size)])
    heights = np.zeros(int(size / 4.0 * (size / 2.0 + 1)))
    index = 0
    for j in range(half):
        v2 = pixel[j] * pixel[j]
        for i in range(j, half):
            u2 = pixel[i] * pixel[i]
            heights[index] = np.sqrt(1 - u2 - v2) if (u2 + v2) < 1.0 else -1
            index += 1
    return pixel, heights


def disc_heights(size=52):
    """The hemisphere's height over every pixel of the NB × NB disc (sqrtab over init_NFF's folded table), −1
    outside the disc."""
    pixel, heights = disc_tables(size)
    half = int(size / 2.0)
    table = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a = i if i < half else size - i - 1
            b = j if j < half else size - j - 1
            if b > a:
                a, b = b, a
            table[i, j] = heights[int(a + b * (half - (b + 1) / 2.0))]
    return pixel, table


@njit(cache=True, nogil=True)
def _sqrt(value):
    """Msqrt: the square root, 0 for a slightly negative value."""
    return 0. if value < 0 else np.sqrt(value)


@njit(cache=True, nogil=True)
def _dot(a0, a1, a2, b0, b1, b2):
    """Vecteur::prod_scalaire, in float32."""
    return F32(F32(F32(a0 * b0) + F32(a1 * b1)) + F32(a2 * b2))


@njit(cache=True, nogil=True)
def _sphere(x, y, z):
    """loc2sph: a point to (x/r, y/r, r), float32, r the double square root of its float32 squared norm."""
    r = F32(_sqrt(np.float64(F32(F32(F32(x * x) + F32(y * y)) + F32(z * z)))))
    return F32(x / r), F32(y / r), r


@njit(cache=True, nogil=True)
def _crossing(a0, a1, a2, b0, b1, b2):
    """Pz0: where segment a–b (float32) crosses z = 0 (k in float32, as the C++ computes it)."""
    k = np.float64(F32(b2 / F32(b2 - a2)))
    return F32(k * a0 + (1. - k) * b0), F32(k * a1 + (1. - k) * b1), F32(0.)


@njit(cache=True, nogil=True)
def _determinant(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """solve3's determinant, its terms in its order."""
    return m00 * (m11 * m22 - (m21 * m12)) - m01 * (m10 * m22 - (m20 * m12)) + m02 * (m10 * m21 - (m20 * m11))


@njit(cache=True, nogil=True)
def _solve3(m00, m01, m02, m10, m11, m12, m20, m21, m22, b0, b1, b2):
    """solve3: Cramer's rule in double, (solvable, x0, x1, x2)."""
    determinant = _determinant(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if determinant == 0.0:
        return False, 0., 0., 0.
    return (True, _determinant(b0, m01, m02, b1, m11, m12, b2, m21, m22) / determinant,
            _determinant(m00, b0, m02, m10, b1, m12, m20, b2, m22) / determinant,
            _determinant(m00, m01, b0, m10, m11, b1, m20, m21, b2) / determinant)


@njit(cache=True, nogil=True)
def _edge_plane(a0, a1, a2, b0, b1, b2):
    """init_MPE: the plane through the origin and edge a–b (float32), normalised on its first non-zero coefficient,
    (degenerate, e0, e1, e2)."""
    d0 = np.float64(F32(F32(a1 * b2) - F32(b1 * a2)))
    d1 = np.float64(F32(-F32(b0 * a2) + F32(a0 * b2)))
    d2 = np.float64(F32(F32(a0 * b1) - F32(b0 * a1)))
    if d0 != 0.0:
        return False, 1., -d1 / d0, d2 / d0
    if d1 != 0.0:
        return False, -d0 / d1, 1., -d2 / d1
    if d2 != 0.0:
        return False, d0 / d2, -d1 / d2, 1.
    return True, 0., 0., 0.


# rows of _project's work arrays (workspace()): float32 ...
PROJECTED, MIDDLE, APEX, CORNERS, SPHERICAL = 0, 3, 5, 8, 11
# ... and float64
MATRIX, PLANE, EDGES, RECTANGLE, BOUNDS = 0, 3, 4, 7, 8


@njit(cache=True, nogil=True)
def workspace():
    """_project's work arrays: float32 (14, 3), the projected vertices, the plane's crossings, the parts' apexes and
    corners, their spherical coordinates; float64 (9, 4), the emitter plane's system and solution, the edges'
    planes, the bounding rectangle and an edge's bounds."""
    return np.zeros((14, 3), dtype=np.float32), np.zeros((9, 4))


@njit(cache=True, nogil=True)
def _copy(work, row, source_row):
    """work[row] = work[source_row], element by element (no array view)."""
    for k in range(3):
        work[row, k] = work[source_row, k]


@njit(cache=True, nogil=True)
def _project(emitter, emitters, frame, receiver_translucent, emitter_opaque, owners, depths, size, pixel, disc,
             work, work64, limit):
    """proj_ortho: the emitter triangle (float32 emitters[emitter], its vertices, centre and normal; translated by
    frame's shift in x, y) drawn on the receiver's discs (owners and depths, upper then lower; frame: the
    receiver's origin, u, v, w, then the shift), its index kept where nearest (a hit's d = −1 / distance: hits
    with d > *limit* dropped, −1 / R with `-B`); *work*, *work64*: workspace(). Few arrays, indexed, never sliced: each array passed or viewed costs atomic reference counts."""
    if emitter_opaque:
        er0 = F32(emitters[emitter, 3, 0] - frame[0, 0])
        er1 = F32(emitters[emitter, 3, 1] - frame[0, 1])
        er2 = F32(emitters[emitter, 3, 2] - frame[0, 2])
        if _dot(er0, er1, er2, emitters[emitter, 4, 0], emitters[emitter, 4, 1], emitters[emitter, 4, 2]) > 0:
            return
    above0 = above1 = below0 = below1 = on0 = -1
    side = 0
    for i in range(3):
        x = F32(F32(emitters[emitter, i, 0] + frame[4, 0]) - frame[0, 0])
        y = F32(F32(emitters[emitter, i, 1] + frame[4, 1]) - frame[0, 1])
        z = F32(emitters[emitter, i, 2] - frame[0, 2])
        for axis in range(3):
            work[PROJECTED + i, axis] = _dot(x, y, z, frame[1 + axis, 0], frame[1 + axis, 1], frame[1 + axis, 2])
        if work[PROJECTED + i, 2] > 1e-6:
            side += 1
            if above0 == -1:
                above0 = i
            else:
                above1 = i
        elif work[PROJECTED + i, 2] < -1e-6:
            side -= 1
            if below0 == -1:
                below0 = i
            else:
                below1 = i
        else:
            side += 4
            if on0 == -1:
                on0 = i
    if side == 4:  # one vertex on the plane: two triangles
        parts = 2
        _copy(work, MIDDLE, PROJECTED + on0)
        _copy(work, APEX, PROJECTED + above0)
        _copy(work, APEX + 1, PROJECTED + below0)
        mirrored = 0
    elif side == 1 or side == -1:  # one vertex below (above): three triangles
        parts = 3
        _copy(work, APEX, PROJECTED + (below0 if side == 1 else above0))
        _copy(work, APEX + 1, PROJECTED + (above0 if side == 1 else below0))
        _copy(work, APEX + 2, PROJECTED + (above1 if side == 1 else below1))
        mirrored = 1 if side == 1 else 0
    else:
        parts = 1
        mirrored = 0 if (side == 3 or side == 6 or side == 9) else 1
        _copy(work, MIDDLE, PROJECTED)
        _copy(work, MIDDLE + 1, PROJECTED + 1)
        _copy(work, APEX, PROJECTED + 2)
    if parts > 1:
        for crossing in range(parts - 1):  # middle[1] on apex 0–1 (two parts); middle[0], [1] on 0–1, 0–2 (three)
            row = MIDDLE + 1 if parts == 2 else MIDDLE + crossing
            b = APEX + 1 + crossing
            work[row, 0], work[row, 1], work[row, 2] = _crossing(work[APEX, 0], work[APEX, 1], work[APEX, 2],
                                                                 work[b, 0], work[b, 1], work[b, 2])
    for part in range(parts):
        if part == 0:
            _copy(work, CORNERS, MIDDLE)
            _copy(work, CORNERS + 1, MIDDLE + 1)
            _copy(work, CORNERS + 2, APEX)
        elif part == 1:
            _copy(work, CORNERS, MIDDLE + 1)
            _copy(work, CORNERS + 1, MIDDLE)
            _copy(work, CORNERS + 2, APEX + 1)
            mirrored = (mirrored + 1) % 2
        else:
            _copy(work, CORNERS, MIDDLE + 1)
            _copy(work, CORNERS + 1, APEX + 1)
            _copy(work, CORNERS + 2, APEX + 2)
        if not (mirrored == 0 or (receiver_translucent and mirrored == 1)):  # an opaque receiver's lower side
            continue
        if mirrored == 1:
            for i in range(3):
                work[CORNERS + i, 2] = abs(work[CORNERS + i, 2])
        zero_x = zero_y = 0  # the corner (from 1) with a zero x, y
        for i in range(3):
            c = CORNERS + i
            if work[c, 0] == 0 and work[c, 1] == 0.:
                work[c, 0] = work[c, 1] = F32(0.0000001)
            else:
                if work[c, 0] == 0:
                    if zero_x != 0:
                        work[c, 0] = F32(0.0000001)
                        work[CORNERS + zero_x - 1, 0] = F32(0.0000001)
                    else:
                        zero_x = i + 1
                if work[c, 1] == 0:
                    if zero_y != 0:
                        work[c, 1] = F32(0.0000001)
                        work[CORNERS + zero_y - 1, 1] = F32(0.0000001)
                    else:
                        zero_y = i + 1
            if work[c, 2] < 1e-6:
                work[c, 2] = F32(np.float64(work[c, 2]) + 0.000001)
        solvable, plane0, plane1, plane2 = _solve3(
            np.float64(work[CORNERS, 0]), np.float64(work[CORNERS, 1]), np.float64(work[CORNERS, 2]),
            np.float64(work[CORNERS + 1, 0]), np.float64(work[CORNERS + 1, 1]), np.float64(work[CORNERS + 1, 2]),
            np.float64(work[CORNERS + 2, 0]), np.float64(work[CORNERS + 2, 1]), np.float64(work[CORNERS + 2, 2]),
            -1., -1., -1.)
        if not solvable:
            continue
        centre_x = F32(np.float64(F32(F32(work[CORNERS, 0] + work[CORNERS + 1, 0]) + work[CORNERS + 2, 0])) / 3.)
        centre_y = F32(np.float64(F32(F32(work[CORNERS, 1] + work[CORNERS + 1, 1]) + work[CORNERS + 2, 1])) / 3.)
        centre_z = F32(np.float64(F32(F32(work[CORNERS, 2] + work[CORNERS + 1, 2]) + work[CORNERS + 2, 2])) / 3.)
        centre_x, centre_y, centre_z = _sphere(centre_x, centre_y, centre_z)
        height = _sqrt(np.float64(F32(F32(F32(1) - F32(centre_x * centre_x)) - F32(centre_y * centre_y))))
        for i in range(3):
            start, end = CORNERS + i, CORNERS + (i + 1) % 3
            degenerate, e0, e1, e2 = _edge_plane(work[start, 0], work[start, 1], work[start, 2], work[end, 0],
                                                 work[end, 1], work[end, 2])
            if degenerate:
                return
            work64[EDGES + i, 0], work64[EDGES + i, 1], work64[EDGES + i, 2] = e0, e1, e2
            work[SPHERICAL + i, 0], work[SPHERICAL + i, 1], work[SPHERICAL + i, 2] = _sphere(
                work[start, 0], work[start, 1], work[start, 2])
            work64[EDGES + i, 3] = e0 * centre_x + e1 * centre_y + e2 * height
        work64[RECTANGLE, 0] = work64[RECTANGLE, 2] = 1.5
        work64[RECTANGLE, 1] = work64[RECTANGLE, 3] = -1.5
        for i in range(3):
            e0, e1, e2 = work64[EDGES + i, 0], work64[EDGES + i, 1], work64[EDGES + i, 2]
            square0, square1, square2 = e0 * e0, e1 * e1, e2 * e2
            cross = e0 * e1
            total = square0 + square1 + square2
            for j in range(2):
                part_square = (square0 if j == 0 else square1) + square2
                extreme = _sqrt(part_square / total)
                if part_square == 0.:
                    return
                position = (-cross * extreme) / part_square
                high_index = 3 - j * 2
                if work[SPHERICAL + (i + 1) % 3, j] > work[SPHERICAL + i, j]:
                    first, second = SPHERICAL + i, SPHERICAL + (i + 1) % 3
                else:
                    second, first = SPHERICAL + i, SPHERICAL + (i + 1) % 3
                other = (j + 1) % 2
                highest = max(np.float64(work[first, other]), np.float64(work[second, other]))
                lowest = min(np.float64(work[first, other]), np.float64(work[second, other]))
                if (work[first, j] > 0) != (work[second, j] > 0):
                    a = (position - work[first, j]) / np.float64(F32(work[second, j] - work[first, j]))
                    ordinate = (1 - a) * work[first, other] + a * work[second, other]
                    if (ordinate > 0) != (extreme > 0):
                        position = -position
                        extreme = -extreme
                if position > work[first, j] and position < work[second, j]:
                    work64[BOUNDS, high_index] = extreme if extreme > highest else highest
                    work64[BOUNDS, high_index - 1] = extreme if extreme < lowest else lowest
                else:
                    work64[BOUNDS, high_index] = highest
                    work64[BOUNDS, high_index - 1] = lowest
                work64[RECTANGLE, high_index] = max(work64[BOUNDS, high_index], work64[RECTANGLE, high_index])
                work64[RECTANGLE, high_index - 1] = min(work64[BOUNDS, high_index - 1],
                                                        work64[RECTANGLE, high_index - 1])
        first_i = int(((work64[RECTANGLE, 0] + 1) * size - 1) / 2.0)
        first_i = 0 if first_i == 0 else first_i - 1
        last_i = int(((work64[RECTANGLE, 1] + 1) * size - 1) / 2.0)
        last_i = last_i + 1 if last_i < size - 1 else last_i
        first_j = int(((work64[RECTANGLE, 2] + 1) * size - 1) / 2.0)
        first_j = 0 if first_j == 0 else first_j - 1
        last_j = int(((work64[RECTANGLE, 3] + 1) * size - 1) / 2.0)
        last_j = last_j + 1 if last_j < size - 1 else last_j
        disc_index = 0 if mirrored == 0 else 1
        # a pixel inside: on each edge plane's side of the centre (its x coefficient taken as 1, as proj_ortho)
        y0, h0, centre_side0 = work64[EDGES, 1], work64[EDGES, 2], work64[EDGES, 3] > 0
        y1, h1, centre_side1 = work64[EDGES + 1, 1], work64[EDGES + 1, 2], work64[EDGES + 1, 3] > 0
        y2, h2, centre_side2 = work64[EDGES + 2, 1], work64[EDGES + 2, 2], work64[EDGES + 2, 3] > 0
        for i in range(first_i, last_i + 1):
            x = pixel[i]
            for j in range(first_j, last_j + 1):
                h = disc[i, j]
                if h == -1:
                    continue
                y = pixel[j]
                if ((x + y0 * y + h0 * h > 0) != centre_side0 or (x + y1 * y + h1 * h > 0) != centre_side1
                        or (x + y2 * y + h2 * h > 0) != centre_side2):
                    continue
                distance = plane0 * x + plane1 * y + plane2 * h
                if distance > limit:
                    continue
                if distance < depths[disc_index, i, j]:
                    depths[disc_index, i, j] = F32(distance)
                    owners[disc_index, i, j] = emitter


# --- The receivers ---

@njit(cache=True, nogil=True)
def _frame(receivers, receiver_normals, receiver_centres, r, frame):
    """init_proj: receiver r's frame into *frame*: its origin (centre), u = P0 − P1 normalised, v = w × u, w its
    normal (float32), then no shift."""
    u0 = F32(receivers[r, 0, 0] - receivers[r, 1, 0])
    u1 = F32(receivers[r, 0, 1] - receivers[r, 1, 1])
    u2 = F32(receivers[r, 0, 2] - receivers[r, 1, 2])
    norm = np.float64(np.sqrt(F32(F32(F32(u0 * u0) + F32(u1 * u1)) + F32(u2 * u2))))
    u0, u1, u2 = F32(u0 / norm), F32(u1 / norm), F32(u2 / norm)
    w0, w1, w2 = receiver_normals[r, 0], receiver_normals[r, 1], receiver_normals[r, 2]
    for axis in range(3):
        frame[0, axis] = receiver_centres[r, axis]
        frame[4, axis] = 0
    frame[1, 0], frame[1, 1], frame[1, 2] = u0, u1, u2
    frame[2, 0] = F32(F32(u2 * w1) - F32(u1 * w2))
    frame[2, 1] = F32(F32(u0 * w2) - F32(u2 * w0))
    frame[2, 2] = F32(F32(u1 * w0) - F32(u0 * w1))
    frame[3, 0], frame[3, 1], frame[3, 2] = w0, w1, w2


@njit(cache=True, nogil=True)
def _level(table, rows, level, column):
    """Tenv(level, column) of canestrad's column-major profile table (E−, E+, heights); beyond the table, the top
    level's."""
    index = level + rows * column
    return table[index] if index < table.shape[0] else table[rows - 1 + rows * column]


@njit(cache=True, nogil=True)
def _environment(i, j, sign, frame, pixel, disc, radius, table, rows, thickness, weights, weight_row):
    """env: the far field's flux seen through pixel (i, j) of the upper (*sign* 1) or lower (−1) hemisphere: the
    profile's E− (looking up) or E+ (looking down) where the pixel's direction leaves the sphere (float32 Point
    arithmetic, as canestrad), 0 below the soil level or above the top; its levels' weights added to
    weights[weight_row + column] (the -w files' bfc)."""
    m0, m1, m2 = F32(pixel[i]), F32(pixel[j]), F32(disc[i, j])
    tr = F32(sign)
    d2 = F32(F32(F32(m0 * frame[1, 2]) + F32(m1 * frame[2, 2])) + F32(F32(m2 * frame[3, 2]) * tr))
    height = F32(F32(np.float64(d2) * radius) + frame[0, 2])
    bottom, top = table[2 * rows], table[2 * rows + rows - 1]
    if height <= bottom or height > top:
        return 0.0
    column = 0 if d2 > 0 else 1
    layer = (np.float64(height) - bottom) / thickness
    level = int(layer) % 256  # an unsigned char
    layer -= level
    row = weight_row + column
    if layer == 0.0:
        if level < rows:
            weights[row, level] += 1
        return _level(table, rows, level, column)
    if level < rows:
        weights[row, level] += layer
    if level + 1 < rows:
        weights[row, level + 1] += 1. - layer
    return layer * _level(table, rows, level, column) + (1 - layer) * _level(table, rows, level + 1, column)


@njit(cache=True, nogil=True)
def _neighbourhood(centre, origin, side, voxels, box_sizes, infinite, period, radius, boxes, shifts):
    """The boxes (up to 8, de-duplicated per box and translation, empty ones skipped but the first) whose
    diffusers may lie within the sphere around *centre* (calc_FF_Bfar), with their translations in x, y; their
    count. The receiver's voxel as canestrad's: int(trunc(G − O) / side)."""
    shape = voxels.shape
    voxel = np.zeros(3, dtype=np.int64)
    step = np.zeros(3, dtype=np.int64)
    move = np.zeros(2, dtype=np.float32)
    corner = np.zeros(3)
    for i in range(3):
        voxel[i] = int(np.float64(int(F32(centre[i] - origin[i]))) / side)
        voxel[i] = min(max(voxel[i], 0), shape[i] - 1)  # beyond the grid (a sensor): canestrad reads anywhere
        middle = (voxel[i] + 0.5) * side + origin[i]
        wraps = infinite and i != 2
        if centre[i] < middle and (voxel[i] != 0 or wraps):
            if voxel[i] != 0:
                step[i] = -1
                if i != 2:
                    move[i] = 0.0
            else:
                step[i] = shape[i] - 1
                if i != 2:
                    move[i] = -period[i]
            corner[i] = voxel[i] * side + origin[i]
        elif centre[i] > middle and (voxel[i] != shape[i] - 1 or wraps):
            if voxel[i] != shape[i] - 1:
                step[i] = 1
                if i != 2:
                    move[i] = 0.0
            else:
                step[i] = -shape[i] + 1
                if i != 2:
                    move[i] = period[i]
            corner[i] = (voxel[i] + 1) * side + origin[i]
        else:
            step[i] = 0
            if i != 2:
                move[i] = 0.0
            corner[i] = 0.0
        translated = i != 2 and move[i] != 0
        if (step[i] != 0 or (infinite and shape[i] == 1 and translated)) and abs(centre[i] - corner[i]) > radius:
            step[i] = 0
            if i != 2:
                move[i] = 0.0
    count = 0
    count = _add_box(voxels[voxel[0], voxel[1], voxel[2]], 0., 0., box_sizes, boxes, shifts, count)
    axes = np.array([2, 2, 2])
    diagonal = 3
    if step[0] != 0 or (infinite and shape[0] == 1 and move[0] != 0):
        count = _add_box(voxels[voxel[0] + step[0], voxel[1], voxel[2]], move[0], 0., box_sizes, boxes, shifts,
                         count)
        axes[0] -= 1
        axes[2] -= 1
        diagonal -= 1
    if step[1] != 0 or (infinite and shape[1] == 1 and move[1] != 0):
        count = _add_box(voxels[voxel[0], voxel[1] + step[1], voxel[2]], 0., move[1], box_sizes, boxes, shifts,
                         count)
        axes[0] -= 1
        axes[1] -= 1
        diagonal -= 1
    if step[2] != 0:
        count = _add_box(voxels[voxel[0], voxel[1], voxel[2] + step[2]], 0., 0., box_sizes, boxes, shifts, count)
        axes[1] -= 1
        axes[2] -= 1
        diagonal -= 1
    distance = np.zeros(3)
    for i in range(3):
        distance[i] = (centre[i] - corner[i]) * (centre[i] - corner[i])
    square = radius * radius
    if axes[0] == 0 and distance[0] + distance[1] < square:
        count = _add_box(voxels[voxel[0] + step[0], voxel[1] + step[1], voxel[2]], move[0], move[1], box_sizes,
                         boxes, shifts, count)
    if axes[1] == 0 and distance[1] + distance[2] < square:
        count = _add_box(voxels[voxel[0], voxel[1] + step[1], voxel[2] + step[2]], 0., move[1], box_sizes, boxes,
                         shifts, count)
    if axes[2] == 0 and distance[2] + distance[0] < square:
        count = _add_box(voxels[voxel[0] + step[0], voxel[1], voxel[2] + step[2]], move[0], 0., box_sizes, boxes,
                         shifts, count)
    if diagonal == 0 and distance[0] + distance[1] + distance[2] < square:
        count = _add_box(voxels[voxel[0] + step[0], voxel[1] + step[1], voxel[2] + step[2]], move[0], move[1],
                         box_sizes, boxes, shifts, count)
    return count


@njit(cache=True, nogil=True)
def _add_box(box, dx, dy, box_sizes, boxes, shifts, count):
    """addbox: a box and its translation, unless empty (but the first) or already there."""
    if count != 0:
        if box < 0 or box_sizes[box] == 0:
            return count
        for k in range(count):
            if boxes[k] == box and shifts[k, 0] == F32(dx) and shifts[k, 1] == F32(dy):
                return count
    boxes[count] = box
    shifts[count, 0], shifts[count, 1] = F32(dx), F32(dy)
    return count + 1


@njit(cache=True, nogil=True)
def _receiver_rows(start, stop, receivers, receiver_normals, receiver_centres, receiver_translucent,
                   receiver_labels, receiver_optics, emitters, spreads, translucent, first_face, labels, n_faces,
                   radius, size, pixel, disc, origin, side, voxels, box_sizes, box_indptr, box_members, infinite,
                   period, bias, table, rows, thickness, dff):
    """The rows of receivers start..stop: (row lengths, emitter faces, net counts, far field (receivers, 2) of each
    receiver's upper and lower face, its levels' weights (receivers, 4, rows) as canestrad stores them, float32
    bfc · dFF), each row's faces ascending. *emitters*: the triangles' vertices, centre and normal (n, 5, 3);
    *spreads*: their centre's largest distance to a vertex (appart_sphere); *receiver_optics* (receivers, 4): ρ, τ
    of the upper face, ρ, τ of the lower one; the boxes' lists of diffusers (sensors after the triangles)."""
    owners = np.full((2, size, size), -1, dtype=np.int32)
    depths = np.empty((2, size, size), dtype=np.float32)
    frame = np.zeros((5, 3), dtype=np.float32)
    work, work64 = workspace()
    n_emitters = emitters.shape[0]
    square_radius = radius * radius
    limit = np.inf if bias else -1. / radius
    active_face = np.zeros(n_emitters, dtype=np.int64)
    flags = np.zeros(n_faces, dtype=np.uint8)
    count = np.zeros(n_faces, dtype=np.int32)
    lengths = np.zeros(stop - start, dtype=np.int64)
    far = np.zeros((stop - start, 2))
    weights = np.zeros((stop - start, 4, rows), dtype=np.float32)
    bfc = np.zeros((4, rows))
    boxes = np.zeros(8, dtype=np.int64)
    shifts = np.zeros((8, 2), dtype=np.float32)
    faces_out = []
    counts_out = []
    for r in range(start, stop):
        translucent_r = receiver_translucent[r]
        for i in range(size):
            for j in range(size):
                owners[0, i, j] = owners[1, i, j] = -1
                depths[0, i, j] = UPPER_EMPTY
                depths[1, i, j] = LOWER_EMPTY
        flags[:] = 0
        _frame(receivers, receiver_normals, receiver_centres, r, frame)
        origin0, origin1, origin2 = frame[0, 0], frame[0, 1], frame[0, 2]
        soil = receiver_labels[r] == 0
        box_count = _neighbourhood(receiver_centres[r], origin, side, voxels, box_sizes, infinite, period, radius,
                                   boxes, shifts)
        for b in range(box_count):
            dx, dy = shifts[b, 0], shifts[b, 1]
            frame[4, 0], frame[4, 1] = dx, dy
            flag = 2 if (dx == 0 and dy == 0) else 3
            box = boxes[b]
            if box < 0:
                continue
            for k in range(box_indptr[box], box_indptr[box + 1]):
                e = box_members[k]
                if e >= n_emitters or (r < n_emitters and e == r):  # a sensor, the receiver itself
                    continue
                if soil and labels[e] == 0:
                    continue
                t0, t1, t2 = F32(emitters[e, 3, 0] + dx), F32(emitters[e, 3, 1] + dy), emitters[e, 3, 2]
                if bias:
                    d0, d1, d2 = F32(origin0 - t0), F32(origin1 - t1), F32(origin2 - t2)
                    if not (np.float64(F32(F32(F32(d0 * d0) + F32(d1 * d1)) + F32(d2 * d2))) < square_radius):
                        continue
                else:
                    i0, i1 = F32(origin0 - dx), F32(origin1 - dy)
                    d0, d1 = F32(emitters[e, 3, 0] - i0), F32(emitters[e, 3, 1] - i1)
                    d2 = F32(emitters[e, 3, 2] - origin2)
                    distance = np.sqrt(np.float64(F32(F32(F32(d0 * d0) + F32(d1 * d1)) + F32(d2 * d2))))
                    if not (distance - np.float64(spreads[e]) < radius):
                        continue
                direction0, direction1, direction2 = F32(t0 - origin0), F32(t1 - origin1), F32(t2 - origin2)
                facing = _dot(direction0, direction1, direction2, emitters[e, 4, 0], emitters[e, 4, 1],
                              emitters[e, 4, 2])
                if facing == 0:
                    continue
                face = first_face[e] if (not translucent[e] or facing < 0) else first_face[e] + 1
                active_face[e] = face
                previous = flags[face]
                if ((previous * flag) // 2) % 3 != 0:  # the anti-doublon flags: once untranslated, once translated
                    continue
                flags[face] = previous + flag
                _project(e, emitters, frame, translucent_r, not translucent[e], owners, depths, size, pixel, disc,
                         work, work64, limit)
        frame[4, 0] = frame[4, 1] = 0
        rho_upper, tau_upper = receiver_optics[r, 0], receiver_optics[r, 1]
        rho_lower, tau_lower = receiver_optics[r, 2], receiver_optics[r, 3]
        far_upper = far_lower = 0.0
        bfc[:] = 0.
        touched = []
        for i in range(size):
            for j in range(size):
                if disc[i, j] < 0:
                    continue
                e = owners[0, i, j]
                upper = 0.0
                if e >= 0:
                    face = active_face[e]
                    if count[face] == 0:
                        touched.append(face)
                    count[face] += 1
                elif rows > 0:
                    upper = _environment(i, j, 1, frame, pixel, disc, radius, table, rows, thickness, bfc, 0) * dff
                    far_upper += upper * rho_upper
                if translucent_r:
                    if e < 0 and rows > 0:
                        far_lower += upper * tau_upper
                    e = owners[1, i, j]
                    if e >= 0:
                        face = active_face[e]
                        if count[face] == 0:
                            touched.append(face)
                        count[face] -= 1
                    elif rows > 0:
                        lower = _environment(i, j, -1, frame, pixel, disc, radius, table, rows, thickness, bfc, 2) * dff
                        far_lower += lower * rho_lower
                        far_upper += lower * tau_lower
        far[r - start, 0], far[r - start, 1] = far_upper, far_lower
        for row in range(4):
            for level in range(rows):
                weights[r - start, row, level] = F32(bfc[row, level] * dff)
        touched_array = np.empty(len(touched), dtype=np.int64)
        for index in range(len(touched)):
            touched_array[index] = touched[index]
        touched_array = np.sort(touched_array)
        length = 0
        previous_face = -1
        for face in touched_array:
            if face == previous_face:
                continue
            previous_face = face
            if count[face] != 0:
                faces_out.append(face)
                counts_out.append(count[face])
                length += 1
            count[face] = 0
        lengths[r - start] = length
    faces = np.zeros(len(faces_out), dtype=np.int64)
    values = np.zeros(len(counts_out), dtype=np.int64)
    for index in range(len(faces_out)):
        faces[index] = faces_out[index]
        values[index] = counts_out[index]
    return lengths, faces, values, far, weights


def grid(points, sensors, radius, pattern=None, threshold=50, ratio=1.) -> dict:
    """canestrad's grid of voxels and its BSP leaves (Voxel::construction): voxels of side 2 · ratio · *radius*
    over the accepted triangles' bounds (x, y the *pattern*'s in an infinite canopy), padded to whole voxels; the
    leaves, boxes split at a voxel face while they hold more than *threshold* diffusers (the triangles then the
    *sensors*), a straddling diffuser in both. {"origin", "side", "voxels" (each voxel's leaf), "sizes",
    "indptr", "members" (each leaf's diffusers in order), "infinite", "period"}."""
    points = np.asarray(points, dtype=F32)
    diffusers = points if sensors is None or not len(sensors) else np.concatenate([points, np.asarray(sensors, F32)])
    flat = points.reshape(-1, 3)
    low, high = flat.min(axis=0).astype(F32), flat.max(axis=0).astype(F32)
    period = np.zeros(2, dtype=F32)
    if pattern is not None:
        pattern_low, pattern_high = periodic.pattern_bounds(pattern)
        low[:2], high[:2] = pattern_low, pattern_high
        period = (pattern_high - pattern_low).astype(F32)
    side = 2.0 * ratio * radius
    shape = []
    for axis in range(3):
        extent = np.float64(F32(high[axis] - low[axis]))
        number = int(extent / side) + 1
        pad = (side * number - extent) / 2.0
        high[axis], low[axis] = F32(np.float64(high[axis]) + pad), F32(np.float64(low[axis]) - pad)
        shape.append(number)
    origin = low.copy()
    voxels = np.full(shape, -1, dtype=np.int64)
    leaves = []

    def number(box_low, box_high, members):
        limits = []
        for axis in range(3):
            bounds = []
            for value in (box_low[axis], box_high[axis]):
                index = int(np.float64(F32(value - origin[axis])) / side)
                if np.float64(value) - (index * side + np.float64(origin[axis])) > side / 2.0:
                    index += 1
                bounds.append(index)
            limits.append(bounds)
        voxels[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = len(leaves)
        leaves.append(members)

    def divide(axis, box_low, box_high, members):
        minimal = [abs(np.float64(F32(box_high[i] - box_low[i])) - side) < 1e-4 for i in range(3)]
        if all(minimal) or len(members) <= threshold:
            number(box_low, box_high, members)
            return
        frontier = np.float64(F32(box_high[axis] + box_low[axis])) / 2.0
        index = int((frontier - np.float64(origin[axis])) / side)
        base = index * side + np.float64(origin[axis])
        frontier = F32(base + side if frontier - base > side / 2.0 else base)
        first_high, second_low = box_high.copy(), box_low.copy()
        first_high[axis] = second_low[axis] = frontier
        coordinates = diffusers[members, :, axis] if members else np.zeros((0, 3), dtype=F32)
        below = np.all(coordinates <= frontier, axis=1)
        above = np.all(coordinates >= frontier, axis=1)
        first = [member for member, low_side, high_side in zip(members, below, above) if low_side or not high_side]
        second = [member for member, low_side, high_side in zip(members, below, above) if not low_side]
        following = (axis + 1) % 3
        while minimal[following]:
            following = (following + 1) % 3
        divide(following, box_low, first_high, first)
        divide(following, second_low, box_high, second)

    divide(2, low.copy(), high.copy(), list(range(len(diffusers))))
    sizes = np.array([len(members) for members in leaves], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    members = np.array([member for box in leaves for member in box], dtype=np.int64)
    return {"origin": origin, "side": side, "voxels": voxels, "sizes": sizes, "indptr": indptr, "members": members,
            "infinite": pattern is not None, "period": period}


def single_box(n_diffusers, radius) -> dict:
    """The grid of the full radiosity (the sphere's radius the scene's diagonal): one voxel, one box holding every
    diffuser in order."""
    return {"origin": np.zeros(3, dtype=F32), "side": 2.0 * radius, "voxels": np.zeros((1, 1, 1), dtype=np.int64),
            "sizes": np.array([n_diffusers]), "indptr": np.array([0, n_diffusers]),
            "members": np.arange(n_diffusers, dtype=np.int64), "infinite": False, "period": np.zeros(2, dtype=F32)}


def spreads(points) -> np.ndarray:
    """dGS of each triangle (appart_sphere): its centre's largest distance to a vertex (float32, as canestrad)."""
    points = np.asarray(points, dtype=F32)
    centres = geometry.legacy_centres(points)
    largest = np.full(len(points), -1, dtype=F32)
    for vertex in range(3):
        difference = (centres - points[:, vertex]).astype(F32)
        square = ((difference[:, 0] * difference[:, 0]).astype(F32) + (difference[:, 1] * difference[:, 1]).astype(F32)
                  ).astype(F32) + (difference[:, 2] * difference[:, 2]).astype(F32)
        largest = np.maximum(largest, square.astype(F32))
    return np.sqrt(largest.astype(np.float64)).astype(F32)


def counts(points, labels, translucent, first_face, n_faces, normals, centres, radius, sensors=None,
           sensor_normals=None, sensor_centres=None, size=52, workers=None, chunk=64, grid=None, profile=None,
           optics=None, bias=True):
    """The near-field rows of every receiver, the accepted triangles then the sensors: (indptr, faces, counts), a
    CSR matrix of net pixel counts (rows: receivers; columns: emitter faces). *radius*: the sphere's radius around
    each receiver (emitters whose centre lies farther are left to the far field); *grid*: the candidates' boxes
    (`grid`; one box holding all by default). With a *profile* (the far field's fluxes, as canestrad reads them)
    and the receivers' *optics* (receivers, 4: ρ, τ of the upper face, of the lower one), also each receiver's
    far field (receivers, 2) and its levels' weights (receivers, 4, levels). *bias* False: canestrad's `-B`."""
    pixel, disc = disc_heights(size)
    emitters = np.concatenate([points, centres[:, None], normals[:, None]], axis=1).astype(np.float32)
    receivers, receiver_normals, receiver_centres = points, normals, centres
    receiver_translucent, receiver_labels = translucent, labels
    sensor_count = 0 if sensors is None else len(sensors)
    if sensor_count:
        receivers = np.concatenate([points, sensors]).astype(np.float32)
        receiver_normals = np.concatenate([normals, sensor_normals]).astype(np.float32)
        receiver_centres = np.concatenate([centres, sensor_centres]).astype(np.float32)
        receiver_translucent = np.concatenate([translucent, np.zeros(sensor_count, dtype=bool)])
        receiver_labels = np.concatenate([labels, np.arange(1, sensor_count + 1, dtype=np.int64)])
    total = len(receivers)
    grid = single_box(total, radius) if grid is None else grid
    receiver_optics = np.zeros((total, 4)) if optics is None else np.asarray(optics, dtype=np.float64)
    if profile is None:
        table, rows, thickness = np.zeros(0), 0, 1.
    else:
        table = np.concatenate([profile["downward"], profile["upward"], profile["height"]]).astype(np.float64)
        rows, thickness = len(profile["height"]), float(profile["thickness"])
    starts = list(range(0, total, chunk))
    spread = spreads(points) if not bias else np.zeros(len(points), dtype=F32)

    def run(start):
        return _receiver_rows(start, min(start + chunk, total), receivers, receiver_normals, receiver_centres,
                              receiver_translucent, receiver_labels, receiver_optics, emitters, spread, translucent,
                              first_face, labels, n_faces, float(radius), size, pixel, disc, grid["origin"],
                              float(grid["side"]), grid["voxels"], grid["sizes"], grid["indptr"], grid["members"],
                              grid["infinite"], grid["period"], bias, table, rows, thickness, form_factor(size))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(run, starts))
    lengths = np.concatenate([part[0] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    faces = np.concatenate([part[1] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    values = np.concatenate([part[2] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    if profile is None:
        return indptr, faces, values
    far = np.concatenate([part[3] for part in parts]) if parts else np.zeros((0, 2))
    weights = np.concatenate([part[4] for part in parts]) if parts else np.zeros((0, 4, rows), dtype=F32)
    return (indptr, faces, values), far, weights


def form_factor(size=52) -> float:
    """dFF: a pixel's form factor, 4 / (π NB²) (the disc's area π over its NB² pixels of side 2 / NB, not
    renormalised to the pixels inside the disc)."""
    return 4 / float(size * size) / np.pi
