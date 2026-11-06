"""
Caribu's numba kernels, in one place (metafspm's conventions.md, "Performance rules"): the projections' rasters, the
direct light's budgets, the periodic copies, the near field's form factors, the SAIL layers and the radiosity solver,
each @njit(cache=True) (nogil where the projections run in threads). Called through their modules (projection,
direct, budgets, geometry, periodic, nearfield, nested, sail, solver; the near field on the GPU, gpu, calls its
helpers). warm_up() runs every one on a tiny scene, which fills numba's on-disk cache (metafspm's warm_up runs it with
every port's: python -m openalea.metafspm warm_up).
"""
import math

import numpy as np
from numba import njit

F32 = np.float32

# the corrections of the near field (corrections.resolve's names), as the kernels' option bits
VOXEL_FLOOR, FACE_COUNTS, PROJECTED_FACE, PROFILE_INTERPOLATION, PROFILE_BOUNDS = 1, 2, 4, 8, 16
# rows of _project's work arrays (workspace()): float32 ...
PROJECTED, MIDDLE, APEX, CORNERS, SPHERICAL = 0, 3, 5, 8, 11
# ... and float64
MATRIX, PLANE, EDGES, RECTANGLE, BOUNDS = 0, 3, 4, 7, 8


# ── Normals and centres (geometry) ──


@njit(cache=True, nogil=True)
def legacy_normals(points) -> np.ndarray:
    """(n, 3) float32: each triangle's unit normal as canestrad computes it (Polygone::calcul_normale_cst_equ):
    (P1 − P0) × (P2 − P0) in float32, divided by its float32 norm."""
    normal = np.zeros((points.shape[0], 3), dtype=np.float32)
    for t in range(points.shape[0]):
        first = points[t, 1] - points[t, 0]
        second = points[t, 2] - points[t, 0]
        x = F32(F32(first[1] * second[2]) - F32(first[2] * second[1]))
        y = F32(F32(first[2] * second[0]) - F32(first[0] * second[2]))
        z = F32(F32(first[0] * second[1]) - F32(first[1] * second[0]))
        norm = np.float64(np.sqrt(F32(F32(F32(x * x) + F32(y * y)) + F32(z * z))))
        if norm > 0:
            normal[t, 0], normal[t, 1], normal[t, 2] = F32(x / norm), F32(y / norm), F32(z / norm)
    return normal


@njit(cache=True, nogil=True)
def legacy_centres(points) -> np.ndarray:
    """(n, 3) float32: each triangle's isobarycentre as canestrad's (float32 sums, divided by 3 in double)."""
    centre = np.zeros((points.shape[0], 3), dtype=np.float32)
    for t in range(points.shape[0]):
        for q in range(3):
            total = F32(F32(F32(0.) + points[t, 0, q]) + points[t, 1, q]) + points[t, 2, q]
            centre[t, q] = F32(np.float64(F32(total)) / 3.0)
    return centre


# ── The projections' rasters (projection) ──


EMPTY_DEPTH = np.float32(99999999999.9)


@njit(cache=True, nogil=True, error_model="numpy")
def _normalise(x, y, z):
    """Vecteur::normalise: the norm, the float32 square root (C++'s sqrt(float)) of the float32 components' float32 sum
    of squares, each component divided by it in double and stored as float32."""
    norm = np.float64(np.sqrt(F32(x * x + y * y + z * z)))
    return F32(np.float64(x) / norm), F32(np.float64(y) / norm), F32(np.float64(z) / norm)


@njit(cache=True, nogil=True, error_model="numpy")
def legacy_screen(vmin, vmax, direction):
    """canestrad's screen for the propagation *direction* (float32 (3,)) over the scene's bounds *vmin*, *vmax*
    (float32 (3,)): (origin (3,) f32, v, w, u (3,) f32 the image axes and the beam, du, dv the screen's extents,
    cos θ). Image coordinates of a point P: (v·(P − origin), w·(P − origin), u·(P − origin))."""
    visee_x, visee_y, visee_z = _normalise(direction[0], direction[1], direction[2])
    costeta = -np.float64(visee_z)
    origin = np.zeros(3, dtype=np.float32)
    u, v, w = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    if abs(np.float64(visee_z) + 1.0) < 1e-6:
        origin[0], origin[1], origin[2] = vmin[0], vmax[1], F32(vmax[2] + F32(1))
        du = np.float64(F32(vmax[0] - vmin[0]))
        dv = np.float64(F32(vmax[1] - vmin[1]))
        u[0], u[1], u[2] = visee_x, visee_y, visee_z
        v[0], v[1], v[2] = 1, 0, 0
        w[0], w[1], w[2] = 0, -1, 0
        return origin, v, w, u, du, dv, costeta
    dz = np.float64(F32(vmax[2] - vmin[2]))
    corners = np.zeros((4, 3), dtype=np.float32)                   # Ecran, at z = vmin then translated
    points = np.zeros((8, 2))                                       # Pts, double
    corners[0, 0], corners[0, 1] = vmin[0], vmin[1]
    corners[1, 0], corners[1, 1] = vmax[0], vmin[1]
    corners[2, 0], corners[2, 1] = vmax[0], vmax[1]
    corners[3, 0], corners[3, 1] = vmin[0], vmax[1]
    for i in range(4):
        points[i, 0], points[i, 1] = corners[i, 0], corners[i, 1]
    shift = abs(dz / np.float64(visee_z))
    for i in range(4):
        corners[i, 2] = vmax[2]
        corners[i, 0] = F32(corners[i, 0] + F32(np.float64(visee_x) * shift))
        corners[i, 1] = F32(corners[i, 1] + F32(np.float64(visee_y) * shift))
        corners[i, 2] = F32(corners[i, 2] + F32(np.float64(visee_z) * shift))
        points[i + 4, 0], points[i + 4, 1] = corners[i, 0], corners[i, 1]
        corners[i, 2] = vmin[2]
    ux, uy, uz = _normalise(visee_x, visee_y, F32(0))
    vx, vy, vz = F32(-uy), ux, F32(0)
    # w = visee × v (Vecteur::prod_vectoriel: res0 = v1[2]*h1 - v1[1]*h2 ... with h = visee, v1 = v)
    wx = F32(F32(vz * visee_y) - F32(vy * visee_z))
    wy = F32(F32(vx * visee_z) - F32(vz * visee_x))
    wz = F32(F32(vy * visee_x) - F32(vx * visee_y))
    wx, wy, wz = _normalise(wx, wy, wz)
    if wz < 0:
        wx, wy, wz = F32(-wx), F32(-wy), F32(-wz)
        vx, vy, vz = F32(-vx), F32(-vy), F32(-vz)
    m00, m01, m10, m11 = np.float64(ux), np.float64(uy), np.float64(vx), np.float64(vy)
    mini0, mini1, maxi0, maxi1 = 999999.9, 999999.9, -999999.9, -999999.9
    for i in range(8):
        tmp = points[i, 0]
        points[i, 0] = m00 * tmp + m01 * points[i, 1]
        points[i, 1] = m10 * tmp + m11 * points[i, 1]
        mini0, maxi0 = min(mini0, points[i, 0]), max(maxi0, points[i, 0])
        mini1, maxi1 = min(mini1, points[i, 1]), max(maxi1, points[i, 1])
    corners[0, 0], corners[0, 1] = mini0, mini1
    corners[1, 0], corners[1, 1] = maxi0, mini1
    corners[2, 0], corners[2, 1] = maxi0, maxi1
    corners[3, 0], corners[3, 1] = mini0, maxi1
    m00, m01, m10, m11 = np.float64(ux), np.float64(vx), np.float64(uy), np.float64(vy)
    for i in range(4):
        tmp = np.float64(corners[i, 0])
        corners[i, 0] = m00 * tmp + m01 * np.float64(corners[i, 1])
        corners[i, 1] = m10 * tmp + m11 * np.float64(corners[i, 1])
    u_dot_w = np.float64(F32(F32(F32(wx * ux) + F32(wy * uy)) + F32(wz * uz)))
    dv = (maxi0 - mini0) * u_dot_w
    for (target, source) in ((1, 0), (2, 3)):
        corners[target, 0] = F32(corners[source, 0] + F32(np.float64(wx) * dv))
        corners[target, 1] = F32(corners[source, 1] + F32(np.float64(wy) * dv))
        corners[target, 2] = F32(corners[source, 2] + F32(np.float64(wz) * dv))
    lift = (dz + 1) / np.float64(visee_z)
    for i in range(4):
        corners[i, 0] = F32(corners[i, 0] + F32(np.float64(visee_x) * lift))
        corners[i, 1] = F32(corners[i, 1] + F32(np.float64(visee_y) * lift))
        corners[i, 2] = F32(corners[i, 2] + F32(np.float64(visee_z) * lift))
    du = maxi1 - mini1
    origin[0], origin[1], origin[2] = corners[0, 0], corners[0, 1], corners[0, 2]
    u[0], u[1], u[2] = visee_x, visee_y, visee_z
    v[0], v[1], v[2] = vx, vy, vz
    w[0], w[1], w[2] = wx, wy, wz
    return origin, v, w, u, du, dv, costeta


@njit(cache=True, nogil=True, error_model="numpy")
def _to_image(point_x, point_y, point_z, shift, origin, v, w, u):
    """Homogene::chgt_base of (point + shift − origin), float32, left to right."""
    x = F32(F32(point_x + shift[0]) - origin[0])
    y = F32(F32(point_y + shift[1]) - origin[1])
    z = F32(F32(point_z + shift[2]) - origin[2])
    return (F32(F32(F32(v[0] * x) + F32(v[1] * y)) + F32(v[2] * z)),
            F32(F32(F32(w[0] * x) + F32(w[1] * y)) + F32(w[2] * z)),
            F32(F32(F32(u[0] * x) + F32(u[1] * y)) + F32(u[2] * z)))


# pattern_cell's layout: on (1), the screen's origin, v, w, u, the pixel scales k0, k1, the cell's low and high corners
CELL = 19


@njit(cache=True, nogil=True, error_model="numpy")
def _in_cell(i, j, z, cell):
    """Whether pixel (i, j)'s point at depth *z* lies in the cell (pattern_cell's): the rasteriser's sample, column i
    at i / k0, row j on its scanline (j + 0.5) / k1."""
    x, y = i / cell[13], (j + 0.5) / cell[14]
    point_x = cell[1] + x * cell[4] + y * cell[7] + z * cell[10]
    point_y = cell[2] + x * cell[5] + y * cell[8] + z * cell[11]
    return cell[15] <= point_x < cell[17] and cell[16] <= point_y < cell[18]


@njit(cache=True, nogil=True, error_model="numpy")
def _colour_triangle(owner, depth, triangle, a, b, c, size, extent_x, extent_y, count_only, cell):
    """colorie_triangle: rows and columns whose pixel centre lies in the triangle (apex *a*, base *b*-*c*, a flat
    base), the nearest along the beam kept (strict <: the first drawn wins ties). a, b, c: double (3,). With
    *count_only* (colorie_capteur, a sensor) the pixels where it would be the nearest are counted, nothing drawn,
    and with a *cell* (pattern_cell's, its first value 1) only those whose point lies in the pattern. Returns that
    count."""
    k0 = (size - 1) / extent_x
    k1 = (size - 1) / extent_y
    slope_left = (b[0] - a[0]) / (b[1] - a[1])
    slope_right = (c[0] - a[0]) / (c[1] - a[1])
    if b[1] > a[1]:
        first, last = int(a[1] * k1 + 0.5), int(b[1] * k1 - 0.5)
    else:
        first, last = int(b[1] * k1 + 0.5), int(a[1] * k1 - 0.5)
    count = 0
    if last < 0 or first >= size or first > last:
        return count
    first, last = max(0, first), min(size - 1, last)
    vab = (b[2] - a[2]) / (b[1] - a[1])
    vbc = (c[2] - b[2]) / (b[1] - a[1])
    yrel = (np.float64(first) + 0.5) / k1 - a[1]
    x_left = slope_left * yrel + a[0]
    x_right = slope_right * yrel + a[0]
    z_left = a[2] + yrel * vab + ((c[2] - b[2]) / (c[0] - b[0]) * 0.5 / k0)
    dz = vbc / (slope_right - slope_left) / k0
    slope_left /= k1
    slope_right /= k1
    vab /= k1
    for j in range(first, last + 1):
        i_left = max(0, int(x_left * k0 + 0.5))
        i_right = min(size - 1, int(x_right * k0 - 0.5))
        if i_left <= i_right:
            z = z_left
            for i in range(i_left, i_right + 1):
                if z < depth[i, j]:
                    if count_only:
                        if cell[0] == 0. or _in_cell(i, j, z, cell):
                            count += 1
                    else:
                        depth[i, j] = F32(z)
                        owner[i, j] = triangle
                z = z + dz
        x_left += slope_left
        x_right += slope_right
        z_left += vab
    return count


@njit(cache=True, nogil=True, error_model="numpy")
def _draw(owner, depth, triangle, points, shift, origin, v, w, u, du, dv, size, count_only, cell):
    """One triangle (float32 (3, 3)) translated by *shift*, projected and split along its middle vertex into
    flat-based triangles as projplan, each coloured (or its visible pixels counted). Returns the count."""
    pp = np.zeros((4, 3), dtype=np.float32)
    a, b, c = np.zeros(3), np.zeros(3), np.zeros(3)
    for q in range(3):
        pp[q, 0], pp[q, 1], pp[q, 2] = _to_image(points[q, 0], points[q, 1], points[q, 2], shift, origin, v, w, u)
    j = 1 if pp[1, 1] > pp[2, 1] else 2
    k = 0 if pp[0, 1] > pp[j, 1] else j
    i = (k + 1) % 3
    j = (i + 1) % 3
    i = i if pp[i, 1] < pp[j, 1] else j
    j = 3 - i - k
    count = 0
    if i == k or (pp[i, 0] == pp[j, 0] and pp[j, 0] == pp[k, 0]) or (pp[i, 1] == pp[j, 1] and pp[j, 1] == pp[k, 1]):
        return count
    up = down = False
    last = 0
    if pp[i, 1] == pp[j, 1]:
        i = i if pp[i, 0] < pp[j, 0] else j
        j = 3 - i - k
        up = pp[i, 0] != pp[j, 0]
    elif pp[j, 1] == pp[k, 1]:
        k = k if pp[j, 0] < pp[k, 0] else j
        j = 3 - i - k
        down = pp[j, 0] != pp[k, 0]
    else:
        pp[3, 1] = pp[j, 1]
        slope = np.float64(F32(F32(pp[3, 1] - pp[i, 1]) / F32(pp[k, 1] - pp[i, 1])))
        pp[3, 0] = F32(slope * np.float64(F32(pp[k, 0] - pp[i, 0])) + np.float64(pp[i, 0]))
        pp[3, 2] = F32(slope * np.float64(F32(pp[k, 2] - pp[i, 2])) + np.float64(pp[i, 2]))
        up = down = pp[j, 0] != pp[3, 0]
        if pp[3, 0] > pp[j, 0]:
            last, i, j = i, j, 3
        else:
            last, i = i, 3
    if up:
        for q in range(3):
            a[q], b[q], c[q] = pp[i, q], pp[j, q], pp[k, q]
        count += _colour_triangle(owner, depth, triangle, c, a, b, size, du, dv, count_only, cell)
    if down:
        if up:
            k, j, i = j, i, last
        for q in range(3):
            a[q], b[q], c[q] = pp[i, q], pp[j, q], pp[k, q]
        count += _colour_triangle(owner, depth, triangle, a, b, c, size, du, dv, count_only, cell)
    return count


@njit(cache=True, nogil=True, error_model="numpy")
def rasterise(points, shifts, counts, origin, v, w, u, du, dv, size):
    """(T, T) the index of the triangle owning each pixel (-1: none), and the depth buffer: each triangle of *points*
    (float32 (n, 3, 3), in scene order) drawn once per copy, its *counts* (n,) first *shifts* (n, k, 3) float32 in
    order (the original, then its periodic border copies: periodic.copy_shifts)."""
    cell = np.zeros(CELL)
    depth = np.full((size, size), EMPTY_DEPTH, dtype=np.float32)
    owner = np.full((size, size), -1, dtype=np.int64)
    for t in range(points.shape[0]):
        for copy in range(counts[t]):
            _draw(owner, depth, t, points[t], shifts[t, copy], origin, v, w, u, du, dv, size, False, cell)
    return owner, depth


@njit(cache=True, nogil=True, error_model="numpy")
def sensor_areas(depth, points, seen, shifts, counts, origin, v, w, u, du, dv, size, pixel_area, cumulative, cell):
    """Each sensor's sunlit area (horizontal equivalent) on the scene's depth buffer: the pixels where it would be
    in front of every triangle (projplan's virtual sensors, after the scene), each copy of *shifts*, *counts*
    (rasterise's) counted. *seen*: whether the beam meets its upper face (only then is it counted). *cumulative*
    (canestrad's): the area added after each copy with the count so far (the first copies counted again). *cell*:
    pattern_cell's, the pixels counted only where the sensor's point lies in the pattern (none: zeros)."""
    areas = np.zeros(points.shape[0])
    owner = np.zeros((1, 1), dtype=np.int64)
    for s in range(points.shape[0]):
        if not seen[s]:
            continue
        pixels = 0
        for copy in range(counts[s]):
            drawn = _draw(owner, depth, s, points[s], shifts[s, copy], origin, v, w, u, du, dv, size, True, cell)
            if cumulative:
                pixels += drawn
                areas[s] += pixels * pixel_area
            else:
                areas[s] += drawn * pixel_area
    return areas


@njit(cache=True, nogil=True, error_model="numpy")
def projected_areas(owner, n_faces, first_face, translucent, lit_upper, pixel_area, depth, cell):
    """Per face, its sunlit area (horizontal equivalent): +pixel_area per pixel for the lit face, −pixel_area for a
    leaf's other face (its transmitted light), summed in projplan's pixel order. *first_face*: each triangle's first
    face number (a leaf's: its upper face, the next its lower one); *lit_upper*: whether the beam meets a leaf's upper
    face (direction · normal < 0). With a *cell* (pattern_cell's), only the pixels whose visible point (at its
    *depth*) lies in the pattern."""
    areas = np.zeros(n_faces)
    size = owner.shape[0]
    for i in range(size):
        for j in range(size):
            t = owner[i, j]
            if t < 0 or (cell[0] != 0. and not _in_cell(i, j, np.float64(depth[i, j]), cell)):
                continue
            face = first_face[t]
            if translucent[t]:
                lit = face if lit_upper[t] else face + 1
                other = face + 1 if lit_upper[t] else face
                areas[lit] += pixel_area
                areas[other] -= pixel_area
            else:
                areas[face] += pixel_area
    return areas


# ── Periodic copies (periodic) ──


@njit(cache=True, nogil=True, error_model="numpy")
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


@njit(cache=True, nogil=True, error_model="numpy")
def _periodise_centres(points):
    """periodise's Polygone::centre: float32 sums, divided by 3 in double, stored as float32."""
    centre = np.zeros((points.shape[0], 3), dtype=np.float32)
    for t in range(points.shape[0]):
        for q in range(3):
            total = F32(F32(F32(F32(0.) + points[t, 0, q]) + points[t, 1, q]) + points[t, 2, q])
            centre[t, q] = F32(np.float64(total) / 3.0)
    return centre


@njit(cache=True, nogil=True, error_model="numpy")
def _outside(x, y, step_x, step_y, size):
    """infinitise's out(): the walk stops once the translated image has left the screen along each axis it moves on."""
    return (step_x == 0 or abs(x) >= size) and (step_y == 0 or abs(y) >= size)


@njit(cache=True, nogil=True, error_model="numpy")
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


@njit(cache=True, nogil=True, error_model="numpy")
def tile(owner, depth, shifts, distance_step, own_copies=False):
    """infinitise without duplication: the image translated by each of *shifts* ((k, 2) pixels), its depth increased
    by *distance_step* per pixel of the second shift; where a translated pixel lies in front, the pixel is in a
    neighbouring pattern's shade (owner −1) and takes its depth. With *own_copies* (the correction "border_copies":
    each triangle's wrapped parts drawn by its copies, the pixels counted in the pattern's cell only) a translate
    never replaces its own triangle: it is the same surface, which the copy draws exactly (the steps are rounded to
    pixels). In place."""
    size = depth.shape[0]
    original = depth.copy()
    original_owner = owner.copy()
    # a pixel ends in the shade when any translate lies in front of the original image, whatever their order: the
    # rows are walked in memory order, each translate's row in turn
    for k in range(size):
        for index in range(shifts.shape[0]):
            i, j = shifts[index, 0], shifts[index, 1]
            if k - i < 0 or k - i >= size:
                continue
            offset = distance_step * j
            source, source_owner = original[k - i], original_owner[k - i]
            target, owners = depth[k], owner[k]
            for l in range(max(j, 0), size + min(j, 0)):
                value = np.float64(source[l - j]) + offset
                if value < target[l] and not (own_copies and source_owner[l - j] == owners[l]):
                    target[l] = F32(value)
                    owners[l] = -1


@njit(cache=True, nogil=True, error_model="numpy")
def tile_offsets(owner, depth, shifts, offsets, own_copies):
    """tile with each translate's own depth offset (lattice_translations')."""
    size = depth.shape[0]
    original = depth.copy()
    original_owner = owner.copy()
    for k in range(size):
        for index in range(shifts.shape[0]):
            i, j = shifts[index, 0], shifts[index, 1]
            if k - i < 0 or k - i >= size:
                continue
            offset = offsets[index]
            source, source_owner = original[k - i], original_owner[k - i]
            target, owners = depth[k], owner[k]
            for l in range(max(j, 0), size + min(j, 0)):
                value = np.float64(source[l - j]) + offset
                if value < target[l] and not (own_copies and source_owner[l - j] == owners[l]):
                    target[l] = F32(value)
                    owners[l] = -1


# ── Budgets of the lit pixels (budgets) ──


@njit(cache=True, nogil=True, error_model="numpy")
def triangle_budgets(radiosity, translucent, reflectance, transmittance):
    """Per accepted triangle (n,) arrays (Eabs, Ei, Ei_sup, Ei_inf) as float32, from the faces' radiosities
    (n_faces,): Ei the total incident (sup + inf for a leaf), Ei_inf −1 for an opaque triangle."""
    count = len(translucent)
    absorbed, incident = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    incident_sup, incident_inf = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    face = 0
    total = F32(0.)  # canestrad keeps the previous triangle's total for a singular leaf with r0 != t1
    for t in range(count):
        if not translucent[t]:
            rho, b = reflectance[face], radiosity[face]
            if rho == 0:
                absorbed[t] = incident[t] = incident_sup[t] = F32(-1)
            else:
                incident_sup[t] = total = F32(b / rho)
                incident[t] = total
                absorbed[t] = F32(np.float64(incident_sup[t]) - b)
            incident_inf[t] = F32(-1)
            face += 1
            continue
        r0, t0 = F32(reflectance[face]), F32(transmittance[face])
        r1, t1 = F32(reflectance[face + 1]), F32(transmittance[face + 1])
        b_sup, b_inf = radiosity[face], radiosity[face + 1]
        determinant = F32(F32(r0 * r1) - F32(t0 * t1))
        if determinant == 0:
            absorbed[t] = incident_sup[t] = incident_inf[t] = F32(-1)
            if r0 == t1:
                absorbed[t] = F32(b_sup * np.float64(F32(F32(1) / r0 - F32(1))) - b_inf)
                total = F32(absorbed[t] / F32(F32(F32(1) - r0) - t0))
        else:
            sup = F32(F32(b_sup * np.float64(r1) - b_inf * np.float64(t1)) / determinant)
            inf = F32(F32(np.float64(r0) * b_inf - np.float64(t0) * b_sup) / determinant)
            incident_sup[t], incident_inf[t] = sup, inf
            absorbed[t] = F32(np.float64(F32(sup + inf)) - (b_sup + b_inf))
            total = F32(sup + inf)
        incident[t] = total
        face += 2
    return absorbed, incident, incident_sup, incident_inf


@njit(cache=True, nogil=True, error_model="numpy")
def face_budgets(incident, translucent, reflectance, transmittance):
    """triangle_budgets from the faces' irradiances (n_faces,) (the correction "face_irradiance"): Ei_sup, Ei_inf the
    faces', Ei their sum, Eabs each face's absorptance times its irradiance (no 2 × 2 system: a leaf with
    ρ_sup ρ_inf = τ_sup τ_inf, ρ = τ for instance, or a black face are budgeted too); Ei_inf −1 for an opaque
    triangle."""
    count = len(translucent)
    absorbed, total = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    incident_sup, incident_inf = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    face = 0
    for t in range(count):
        upper = incident[face]
        incident_sup[t] = F32(upper)
        if not translucent[t]:
            absorbed[t] = F32((1. - reflectance[face]) * upper)
            total[t] = F32(upper)
            incident_inf[t] = F32(-1)
            face += 1
            continue
        lower = incident[face + 1]
        incident_inf[t] = F32(lower)
        total[t] = F32(upper + lower)
        absorbed[t] = F32((1. - reflectance[face] - transmittance[face]) * upper
                          + (1. - reflectance[face + 1] - transmittance[face + 1]) * lower)
        face += 2
    return absorbed, total, incident_sup, incident_inf


# ── The direct light (direct) ──


@njit(cache=True, nogil=True, error_model="numpy")
def _legacy_norm(x, y, z):
    return np.float64(np.sqrt(F32(x * x + y * y + z * z)))


@njit(cache=True, nogil=True, error_model="numpy")
def legacy_surfaces(points):
    """Each triangle's area as Polygone::surface computes it on float32 points: |P0P1| |P0P2| sin(angle) / 2, the
    angle the arc cosine of the unit edges' float32 dot product."""
    surfaces = np.zeros(points.shape[0])
    for t in range(points.shape[0]):
        first = points[t, 1] - points[t, 0]
        second = points[t, 2] - points[t, 0]
        first_norm = _legacy_norm(first[0], first[1], first[2])
        second_norm = _legacy_norm(second[0], second[1], second[2])
        dot = F32(0.)
        for q in range(3):
            dot = F32(dot + F32(F32(np.float64(first[q]) / first_norm) * F32(np.float64(second[q]) / second_norm)))
        cosine = np.float64(dot)
        if abs(cosine) > 1.:
            angle = 0. if cosine > 0 else math.pi
        else:
            angle = math.acos(cosine)
        surfaces[t] = first_norm * second_norm * math.sin(angle) / 2.0
    return surfaces


@njit(cache=True, nogil=True, error_model="numpy")
def accumulate(radiosity, incident, areas, energy, first_face, translucent, reflectance, transmittance, surfaces):
    """Adds a source's contribution to *radiosity* (n_faces,): E ρ A / S on the lit face, E τ(lit face) |A| / S on a
    leaf's other face (A < 0 there), in canestrad's order of operations; and to *incident* (n_faces,), each face's
    irradiance, E A / S on the lit face."""
    for t in range(first_face.shape[0]):
        faces = 2 if translucent[t] else 1
        for offset in range(faces):
            face = first_face[t] + offset
            area = areas[face]
            if area == 0.0:
                continue
            if area > 0:
                rho = reflectance[face] * area / surfaces[t]
                incident[face] += energy * area / surfaces[t]
            else:
                rho = -transmittance[first_face[t] + 1 - offset] * area / surfaces[t]
            radiosity[face] += energy * rho


# ── The near field (nearfield; gpu clones them as device functions) ──


UPPER_EMPTY = np.float32(99999999999.9)


LOWER_EMPTY = np.float32(999999999999.9)


@njit(cache=True, nogil=True, error_model="numpy")
def _sqrt(value):
    """Msqrt: the square root, 0 for a slightly negative value."""
    return 0. if value < 0 else math.sqrt(value)


@njit(cache=True, nogil=True, error_model="numpy")
def _dot(a0, a1, a2, b0, b1, b2):
    """Vecteur::prod_scalaire, in float32."""
    return F32(F32(F32(a0 * b0) + F32(a1 * b1)) + F32(a2 * b2))


@njit(cache=True, nogil=True, error_model="numpy")
def _sphere(x, y, z):
    """loc2sph: a point to (x/r, y/r, r), float32, r the double square root of its float32 squared norm."""
    r = F32(_sqrt(np.float64(F32(F32(F32(x * x) + F32(y * y)) + F32(z * z)))))
    return F32(x / r), F32(y / r), r


@njit(cache=True, nogil=True, error_model="numpy")
def _crossing(a0, a1, a2, b0, b1, b2):
    """Pz0: where segment a–b (float32) crosses z = 0 (k in float32, as the C++ computes it)."""
    k = np.float64(F32(b2 / F32(b2 - a2)))
    return F32(k * a0 + (1. - k) * b0), F32(k * a1 + (1. - k) * b1), F32(0.)


@njit(cache=True, nogil=True, error_model="numpy")
def _determinant(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """solve3's determinant, its terms in its order."""
    return m00 * (m11 * m22 - (m21 * m12)) - m01 * (m10 * m22 - (m20 * m12)) + m02 * (m10 * m21 - (m20 * m11))


@njit(cache=True, nogil=True, error_model="numpy")
def _solve3(m00, m01, m02, m10, m11, m12, m20, m21, m22, b0, b1, b2):
    """solve3: Cramer's rule in double, (solvable, x0, x1, x2)."""
    determinant = _determinant(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if determinant == 0.0:
        return False, 0., 0., 0.
    return (True, _determinant(b0, m01, m02, b1, m11, m12, b2, m21, m22) / determinant,
            _determinant(m00, b0, m02, m10, b1, m12, m20, b2, m22) / determinant,
            _determinant(m00, m01, b0, m10, m11, b1, m20, m21, b2) / determinant)


@njit(cache=True, nogil=True, error_model="numpy")
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


@njit(cache=True, nogil=True, error_model="numpy")
def workspace():
    """_project's work arrays: float32 (14, 3), the projected vertices, the plane's crossings, the parts' apexes and
    corners, their spherical coordinates; float64 (9, 4), the emitter plane's system and solution, the edges'
    planes, the bounding rectangle and an edge's bounds."""
    return np.zeros((14, 3), dtype=np.float32), np.zeros((9, 4))


@njit(cache=True, nogil=True, error_model="numpy")
def _copy(work, row, source_row):
    """work[row] = work[source_row], element by element (no array view)."""
    for k in range(3):
        work[row, k] = work[source_row, k]


@njit(cache=True, nogil=True, error_model="numpy")
def _project(emitter, emitters, frame, receiver_translucent, emitter_opaque, owners, depths, size, pixel, disc,
             work, work64, limit, stored):
    """proj_ortho: the emitter triangle (float32 emitters[emitter], its vertices, centre and normal; translated by
    frame's shift in x, y) drawn on the receiver's discs (owners and depths, upper then lower; frame: the
    receiver's origin, u, v, w, then the shift), *stored* (its index, or the face projected) kept where nearest (a
    hit's d = −1 / distance: hits with d > *limit* dropped, −1 / R with `-B`); *work*, *work64*: workspace(). Few
    arrays, indexed, never sliced: each array passed or viewed costs atomic reference counts."""
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
                    owners[disc_index, i, j] = stored


@njit(cache=True, nogil=True, error_model="numpy")
def _frame(receivers, receiver_normals, receiver_centres, r, frame):
    """init_proj: receiver r's frame into *frame*: its origin (centre), u = P0 − P1 normalised, v = w × u, w its
    normal (float32), then no shift."""
    u0 = F32(receivers[r, 0, 0] - receivers[r, 1, 0])
    u1 = F32(receivers[r, 0, 1] - receivers[r, 1, 1])
    u2 = F32(receivers[r, 0, 2] - receivers[r, 1, 2])
    norm = np.float64(F32(math.sqrt(np.float64(F32(F32(F32(u0 * u0) + F32(u1 * u1)) + F32(u2 * u2))))))  # sqrtf
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


@njit(cache=True, nogil=True, error_model="numpy")
def _level(table, rows, level, column):
    """Tenv(level, column) of canestrad's column-major profile table (E−, E+, heights); beyond the table, the top
    level's."""
    index = level + rows * column
    return table[index] if index < table.shape[0] else table[rows - 1 + rows * column]


@njit(cache=True, nogil=True, error_model="numpy")
def _environment(i, j, sign, frame, pixel, disc, radius, table, rows, thickness, weights, weight_row, options):
    """env: the far field's flux seen through pixel (i, j) of the upper (*sign* 1) or lower (−1) hemisphere: the
    profile's E− (looking up) or E+ (looking down) where the pixel's direction leaves the sphere (float32 Point
    arithmetic, as canestrad), 0 below the soil level or above the top (PROFILE_BOUNDS: the soil level's, the
    top's); its levels' weights added to weights[weight_row + column] (the -w files' bfc), inverted as canestrad's
    (PROFILE_INTERPOLATION: 1 − f on the level below, f on the one above)."""
    m0, m1, m2 = F32(pixel[i]), F32(pixel[j]), F32(disc[i, j])
    tr = F32(sign)
    d2 = F32(F32(F32(m0 * frame[1, 2]) + F32(m1 * frame[2, 2])) + F32(F32(m2 * frame[3, 2]) * tr))
    height = F32(F32(np.float64(d2) * radius) + frame[0, 2])
    bottom, top = table[2 * rows], table[2 * rows + rows - 1]
    column = 0 if d2 > 0 else 1
    if options & PROFILE_BOUNDS:
        clamped = min(max(np.float64(height), bottom), top)
        layer = (clamped - bottom) / thickness
        level = min(int(layer), rows - 1)
        fraction = layer - level if level < rows - 1 else 0.
        row = weight_row + column
        if fraction == 0.:
            weights[row, level] += 1
            return _level(table, rows, level, column)
        below, above = (1. - fraction, fraction) if options & PROFILE_INTERPOLATION else (fraction, 1. - fraction)
        weights[row, level] += below
        weights[row, level + 1] += above
        return below * _level(table, rows, level, column) + above * _level(table, rows, level + 1, column)
    if height <= bottom or height > top:
        return 0.0
    layer = (np.float64(height) - bottom) / thickness
    level = int(layer) % 256  # an unsigned char
    layer -= level
    row = weight_row + column
    if layer == 0.0:
        if level < rows:
            weights[row, level] += 1
        return _level(table, rows, level, column)
    below, above = (1. - layer, layer) if options & PROFILE_INTERPOLATION else (layer, 1. - layer)
    if level < rows:
        weights[row, level] += below
    if level + 1 < rows:
        weights[row, level + 1] += above
    return below * _level(table, rows, level, column) + above * _level(table, rows, level + 1, column)


@njit(cache=True, nogil=True, error_model="numpy")
def _neighbourhood(centre, origin, side, voxels, box_sizes, infinite, period, radius, boxes, shifts, cells, moves,
                   corners, options):
    """The boxes (up to 8, de-duplicated per box and translation, empty ones skipped but the first) whose
    diffusers may lie within the sphere around *centre* (calc_FF_Bfar), with their translations in x, y; their
    count. The receiver's voxel as canestrad's: int(trunc(G − O) / side) (VOXEL_FLOOR: floor((G − O) / side)).
    *cells* int64 (2, 3), *moves* float32 (2,), *corners* float64 (2, 3): work arrays."""
    shape0, shape1, shape2 = voxels.shape[0], voxels.shape[1], voxels.shape[2]
    for i in range(3):
        shape = shape0 if i == 0 else (shape1 if i == 1 else shape2)
        if options & VOXEL_FLOOR:
            voxel = int(math.floor(np.float64(F32(centre[i] - origin[i])) / side))
        else:
            voxel = int(np.float64(int(F32(centre[i] - origin[i]))) / side)
        voxel = min(max(voxel, 0), shape - 1)  # beyond the grid (a sensor): canestrad reads anywhere
        cells[0, i] = voxel
        middle = (voxel + 0.5) * side + origin[i]
        wraps = infinite and i != 2
        step, move, corner = 0, F32(0.), 0.0
        if centre[i] < middle and (voxel != 0 or wraps):
            if voxel != 0:
                step = -1
            else:
                step = shape - 1
                move = F32(-period[i]) if i != 2 else F32(0.)
            corner = voxel * side + origin[i]
        elif centre[i] > middle and (voxel != shape - 1 or wraps):
            if voxel != shape - 1:
                step = 1
            else:
                step = -shape + 1
                move = period[i] if i != 2 else F32(0.)
            corner = (voxel + 1) * side + origin[i]
        translated = i != 2 and move != 0
        if (step != 0 or (infinite and shape == 1 and translated)) and abs(centre[i] - corner) > radius:
            step, move = 0, F32(0.)
        cells[1, i] = step
        corners[0, i] = corner
        corners[1, i] = (centre[i] - corner) * (centre[i] - corner)
        if i != 2:
            moves[i] = move
    v0, v1, v2 = cells[0, 0], cells[0, 1], cells[0, 2]
    s0, s1, s2 = cells[1, 0], cells[1, 1], cells[1, 2]
    move0, move1 = moves[0], moves[1]
    count = _add_box(voxels[v0, v1, v2], F32(0.), F32(0.), box_sizes, boxes, shifts, 0)
    axis_xy = axis_yz = axis_zx = 2
    diagonal = 3
    if s0 != 0 or (infinite and shape0 == 1 and move0 != 0):
        count = _add_box(voxels[v0 + s0, v1, v2], move0, F32(0.), box_sizes, boxes, shifts, count)
        axis_xy -= 1
        axis_zx -= 1
        diagonal -= 1
    if s1 != 0 or (infinite and shape1 == 1 and move1 != 0):
        count = _add_box(voxels[v0, v1 + s1, v2], F32(0.), move1, box_sizes, boxes, shifts, count)
        axis_xy -= 1
        axis_yz -= 1
        diagonal -= 1
    if s2 != 0:
        count = _add_box(voxels[v0, v1, v2 + s2], F32(0.), F32(0.), box_sizes, boxes, shifts, count)
        axis_yz -= 1
        axis_zx -= 1
        diagonal -= 1
    square = radius * radius
    d0, d1, d2 = corners[1, 0], corners[1, 1], corners[1, 2]
    if axis_xy == 0 and d0 + d1 < square:
        count = _add_box(voxels[v0 + s0, v1 + s1, v2], move0, move1, box_sizes, boxes, shifts, count)
    if axis_yz == 0 and d1 + d2 < square:
        count = _add_box(voxels[v0, v1 + s1, v2 + s2], F32(0.), move1, box_sizes, boxes, shifts, count)
    if axis_zx == 0 and d2 + d0 < square:
        count = _add_box(voxels[v0 + s0, v1, v2 + s2], move0, F32(0.), box_sizes, boxes, shifts, count)
    if diagonal == 0 and d0 + d1 + d2 < square:
        count = _add_box(voxels[v0 + s0, v1 + s1, v2 + s2], move0, move1, box_sizes, boxes, shifts, count)
    return count


@njit(cache=True, nogil=True, error_model="numpy")
def _add_box(box, dx, dy, box_sizes, boxes, shifts, count):
    """addbox: a box and its translation, unless empty (but the first) or already there."""
    if count != 0:
        if box < 0 or box_sizes[box] == 0:
            return count
        for k in range(count):
            if boxes[k] == box and shifts[k, 0] == dx and shifts[k, 1] == dy:
                return count
    boxes[count] = box
    shifts[count, 0], shifts[count, 1] = dx, dy
    return count + 1


@njit(cache=True, nogil=True, error_model="numpy")
def _candidates(r, translucent_r, soil, frame, emitters, spreads, translucent, first_face, labels, radius, bias,
                limit, box_count, boxes, shifts, box_indptr, box_members, flags, active_face, owners, depths, size,
                pixel, disc, work, work64, options):
    """The receiver's candidates projected on its discs (calc_FF_Bfar's loop): per box, per diffuser of its list,
    real, not the receiver, not soil–soil, selected (its translated centre within the sphere, or appart_sphere
    without *bias*), not seen edge-on; its facing face activated (*active_face*), projected unless the anti-doublon
    *flags* of that face forbid (once untranslated, once translated). The discs keep the emitter (PROJECTED_FACE:
    the face of the copy projected)."""
    n_emitters = emitters.shape[0]
    square_radius = radius * radius
    origin0, origin1, origin2 = frame[0, 0], frame[0, 1], frame[0, 2]
    for b in range(box_count):
        box = boxes[b]
        if box < 0:
            continue
        dx, dy = shifts[b, 0], shifts[b, 1]
        frame[4, 0], frame[4, 1] = dx, dy
        flag = 2 if (dx == 0 and dy == 0) else 3
        for k in range(box_indptr[box], box_indptr[box + 1]):
            e = box_members[k]
            if e >= n_emitters or e == r:  # a sensor, the receiver itself
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
                distance = math.sqrt(np.float64(F32(F32(F32(d0 * d0) + F32(d1 * d1)) + F32(d2 * d2))))
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
            if ((previous * flag) // 2) % 3 != 0:
                continue
            flags[face] = previous + flag
            _project(e, emitters, frame, translucent_r, not translucent[e], owners, depths, size, pixel, disc,
                     work, work64, limit, face if options & PROJECTED_FACE else e)
    frame[4, 0] = frame[4, 1] = F32(0.)


@njit(cache=True, nogil=True, error_model="numpy")
def _faces(translucent_r, optics, frame, owners, active_face, faces, size, pixel, disc, radius, table, rows,
           thickness, dff, bfc, far, options):
    """NFF's pass over the discs: each pixel's emitter face (*faces* (2, size, size), its last activated face, −1
    where empty; PROJECTED_FACE: the face the disc holds), the far field through the empty ones into far[0] (upper
    face) and far[1] (lower face: ρ, τ in *optics* (4,)), its levels' weights into *bfc* (4, rows)."""
    projected = options & PROJECTED_FACE
    rho_upper, tau_upper, rho_lower, tau_lower = optics[0], optics[1], optics[2], optics[3]
    far_upper = far_lower = 0.0
    for row in range(4):
        for level in range(rows):
            bfc[row, level] = 0.
    for i in range(size):
        for j in range(size):
            faces[0, i, j] = faces[1, i, j] = -1
            if disc[i, j] < 0:
                continue
            e = owners[0, i, j]
            upper = 0.0
            if e >= 0:
                faces[0, i, j] = e if projected else active_face[e]
            elif rows > 0:
                upper = _environment(i, j, 1, frame, pixel, disc, radius, table, rows, thickness, bfc, 0,
                                     options) * dff
                far_upper += upper * rho_upper
            if translucent_r:
                if e < 0 and rows > 0:
                    far_lower += upper * tau_upper
                e = owners[1, i, j]
                if e >= 0:
                    faces[1, i, j] = e if projected else active_face[e]
                elif rows > 0:
                    lower = _environment(i, j, -1, frame, pixel, disc, radius, table, rows, thickness, bfc, 2,
                                         options) * dff
                    far_lower += lower * rho_lower
                    far_upper += lower * tau_lower
    far[0], far[1] = far_upper, far_lower


@njit(cache=True, nogil=True, error_model="numpy")
def _clear(owners, depths, size):
    for i in range(size):
        for j in range(size):
            owners[0, i, j] = owners[1, i, j] = -1
            depths[0, i, j] = UPPER_EMPTY
            depths[1, i, j] = LOWER_EMPTY


@njit(cache=True, nogil=True, error_model="numpy")
def _receiver_rows(start, stop, receivers, receiver_normals, receiver_centres, receiver_translucent,
                   receiver_labels, receiver_optics, emitters, spreads, translucent, first_face, labels, n_faces,
                   radius, size, pixel, disc, origin, side, voxels, box_sizes, box_indptr, box_members, infinite,
                   period, bias, table, rows, thickness, dff, options):
    """The rows of receivers start..stop: (row lengths, emitter faces, net counts, far field (receivers, 2) of each
    receiver's upper and lower face, its levels' weights (receivers, 4, rows): bfc · dFF, canestrad's `-w` files
    holding them as float32), each row's faces ascending (FACE_COUNTS: a face seen by both discs in two entries,
    its upper count then minus its lower one, instead of their difference). *emitters*: the triangles' vertices,
    centre and normal (n, 5, 3); *spreads*: their centre's largest distance to a vertex (appart_sphere);
    *receiver_optics* (receivers, 4): ρ, τ of the upper face, ρ, τ of the lower one; the boxes' lists of diffusers
    (sensors after the triangles)."""
    owners = np.full((2, size, size), -1, dtype=np.int32)
    depths = np.empty((2, size, size), dtype=np.float32)
    faces = np.empty((2, size, size), dtype=np.int64)
    frame = np.zeros((5, 3), dtype=np.float32)
    work, work64 = workspace()
    limit = np.inf if bias else -1. / radius
    active_face = np.zeros(emitters.shape[0], dtype=np.int64)
    flags = np.zeros(n_faces, dtype=np.uint8)
    count = np.zeros(n_faces, dtype=np.int32)
    count_lower = np.zeros(n_faces, dtype=np.int32)
    separate = options & FACE_COUNTS
    lengths = np.zeros(stop - start, dtype=np.int64)
    far = np.zeros((stop - start, 2))
    weights = np.zeros((stop - start, 4, rows))
    bfc = np.zeros((4, rows))
    boxes = np.zeros(8, dtype=np.int64)
    shifts = np.zeros((8, 2), dtype=np.float32)
    cells = np.zeros((2, 3), dtype=np.int64)
    moves = np.zeros(2, dtype=np.float32)
    corners = np.zeros((2, 3))
    faces_out = []
    counts_out = []
    for r in range(start, stop):
        _clear(owners, depths, size)
        flags[:] = 0
        _frame(receivers, receiver_normals, receiver_centres, r, frame)
        box_count = _neighbourhood(receiver_centres[r], origin, side, voxels, box_sizes, infinite, period, radius,
                                   boxes, shifts, cells, moves, corners, options)
        _candidates(r, receiver_translucent[r], receiver_labels[r] == 0, frame, emitters, spreads, translucent,
                    first_face, labels, radius, bias, limit, box_count, boxes, shifts, box_indptr, box_members,
                    flags, active_face, owners, depths, size, pixel, disc, work, work64, options)
        _faces(receiver_translucent[r], receiver_optics[r], frame, owners, active_face, faces, size, pixel, disc,
               radius, table, rows, thickness, dff, bfc, far[r - start], options)
        for row in range(4):
            for level in range(rows):
                weights[r - start, row, level] = bfc[row, level] * dff
        touched = []
        for i in range(size):
            for j in range(size):
                face = faces[0, i, j]
                if face >= 0:
                    if count[face] == 0:
                        touched.append(face)
                    count[face] += 1
                face = faces[1, i, j]
                if face >= 0:
                    if count[face] == 0 and count_lower[face] == 0:
                        touched.append(face)
                    if separate:
                        count_lower[face] += 1
                    else:
                        count[face] -= 1
        touched_array = np.empty(len(touched), dtype=np.int64)
        for index in range(len(touched)):
            touched_array[index] = touched[index]
        touched_array = np.sort(touched_array)
        length = 0
        previous_face = -1
        for face in touched_array:  # per face its pixels above minus those below, ascending, zeros dropped
            if face == previous_face:
                continue
            previous_face = face
            if count[face] != 0:
                faces_out.append(face)
                counts_out.append(count[face])
                length += 1
            if count_lower[face] != 0:
                faces_out.append(face)
                counts_out.append(-count_lower[face])
                length += 1
            count[face] = count_lower[face] = 0
        lengths[r - start] = length
    faces_array = np.zeros(len(faces_out), dtype=np.int64)
    values = np.zeros(len(counts_out), dtype=np.int64)
    for index in range(len(faces_out)):
        faces_array[index] = faces_out[index]
        values[index] = counts_out[index]
    return lengths, faces_array, values, far, weights


# ── Nested radiosity (nested) ──


@njit(cache=True, error_model="numpy")
def _reused(weights, receiver_optics, receiver_translucent, downward, upward, energy):
    count = weights.shape[0]
    flat = np.zeros(2 * count + 1)  # Cenv by face, after a slot for the slip's write before the first
    index = 1
    for r in range(count):
        rho_upper, tau_upper = receiver_optics[r, 0], receiver_optics[r, 1]
        rho_lower, tau_lower = receiver_optics[r, 2], receiver_optics[r, 3]
        for level in range(downward.shape[0]):
            flux = weights[r, 0, level] * downward[level] + weights[r, 1, level] * upward[level]
            flat[index] += rho_upper * flux
            if receiver_translucent[r]:
                flat[index + 1] += tau_upper * flux
        if receiver_translucent[r]:
            for level in range(downward.shape[0]):
                flux = weights[r, 2, level] * downward[level] + weights[r, 3, level] * upward[level]
                flat[index] += tau_lower * flux
                flat[index + 1] += rho_lower * flux
            index += 1
            flat[index] *= energy
        flat[index - 1] *= energy
        index += 1
    far = np.zeros((count, 2))
    index = 1
    for r in range(count):
        far[r, 0] = flat[index]
        if receiver_translucent[r]:
            far[r, 1] = flat[index + 1]
            index += 1
        index += 1
    return far


# ── SAIL's layers (sail) ──


SUBDIVISION_LEVELS = 6      # s2v's LevelMax: a triangle across cells split at most 5 times


DEGREES = 57.29577951       # s2v's 180 / π


@njit(cache=True, error_model="numpy")
def _layer(z, tops, layer_count):
    """classe: the layer of height z (0 the top; *tops* each layer's top, cumulated from the bottom), −10 below 0."""
    if z < 0:
        return -10
    for j in range(1, layer_count):
        if z > tops[j]:
            return j - 1
    return layer_count - 1


@njit(cache=True, error_model="numpy")
def _zenith_class(p, class_count):
    """normal: the zenith of the triangle's normal (0 for a horizontal triangle), its class of 90 / class_count
    degrees. The normal's cross product written out as s2v's, its float32 vertices (Patch) mixed with doubles: the
    edges and the products of two vertices' coordinates in float32, the rest in double."""
    f64 = np.float64
    edge0, edge1, edge2 = f64(F32(p[1, 0] - p[2, 0])), f64(F32(p[1, 1] - p[2, 1])), f64(F32(p[1, 2] - p[2, 2]))
    u = f64(p[0, 1]) * edge2 - f64(p[0, 2]) * edge1 + f64(F32(p[1, 1] * p[2, 2])) - f64(F32(p[2, 1] * p[1, 2]))
    v = f64(-p[0, 0]) * edge2 + f64(p[0, 2]) * edge0 - f64(F32(p[1, 0] * p[2, 2])) + f64(F32(p[2, 0] * p[1, 2]))
    w = f64(p[0, 0]) * edge1 - f64(p[0, 1]) * edge0 + f64(F32(p[1, 0] * p[2, 1])) - f64(F32(p[2, 0] * p[1, 1]))
    square = u * u + v * v
    zenith = 0. if square == 0 else math.acos(abs(w / math.sqrt(square + w * w))) * DEGREES
    return min(class_count - 1, int(zenith / (90.0 / class_count)))


@njit(cache=True, error_model="numpy")
def _structure(points, species, leaf, band_optics, period_x, period_y, tops, layer_count, class_count):
    """The areas per (species, layer, zenith class) and per band their sums × (ρ, τ) per (species, layer), the
    triangles in order, each split depth-first (repart) while its vertices fall in different cells; *points*
    float32 (s2v's Patch), its sums of coordinates in float32, the rest in double."""
    species_count, band_count = band_optics.shape[1], band_optics.shape[0]
    areas = np.zeros((species_count, layer_count, class_count))
    optical = np.zeros((species_count, layer_count, band_count, 2))
    stack = np.zeros((4 * SUBDIVISION_LEVELS + 1, 3, 3), dtype=np.float32)
    levels = np.zeros(4 * SUBDIVISION_LEVELS + 1, dtype=np.int64)
    cells = np.zeros((3, 3), dtype=np.int64)
    piece = np.zeros((3, 3), dtype=np.float32)
    middle = np.zeros((3, 3), dtype=np.float32)
    for t in range(points.shape[0]):
        if species[t] < 0:  # the soil
            continue
        je = species[t]
        area_factor = 0.5 if leaf[t] else 0.25  # a stem counts for half its area
        zenith = _zenith_class(points[t], class_count)
        stack[0] = points[t]
        levels[0] = 0
        top = 1
        while top > 0:
            top -= 1
            for i in range(3):
                for k in range(3):
                    piece[i, k] = stack[top, i, k]
            level = levels[top] + 1
            for i in range(3):
                cells[i, 0] = int(np.float64(piece[i, 0]) / period_x)
                cells[i, 1] = int(np.float64(piece[i, 1]) / period_y)
                cells[i, 2] = _layer(np.float64(piece[i, 2]), tops, layer_count)
            across = 0
            for i in range(3):
                across += abs(cells[0, i] - cells[1, i])
            for i in range(3):
                across += abs(cells[0, i] - cells[2, i])
            if across % 256 == 0:  # a char: in a single cell
                if np.float64(F32(F32(piece[0, 2] + piece[1, 2]) + piece[2, 2])) / 3. < 0:
                    continue
                jz = cells[0, 2]
            elif level == SUBDIVISION_LEVELS:  # the centroid's cell
                centre_z = np.float64(F32(F32(piece[0, 2] + piece[1, 2]) + piece[2, 2])) / 3.
                if centre_z < 0:
                    continue
                jz = _layer(centre_z, tops, layer_count)
            else:  # into 4, pushed so that the first is processed first
                for j in range(3):
                    for i in range(3):
                        middle[j, i] = F32(np.float64(F32(piece[j, i] + piece[(j + 1) % 3, i])) / 2.)
                for child in range(3, -1, -1):
                    for i in range(3):
                        if child == 0:
                            corners = piece[0, i], middle[0, i], middle[2, i]
                        elif child == 1:
                            corners = middle[0, i], piece[1, i], middle[1, i]
                        elif child == 2:
                            corners = middle[1, i], middle[2, i], piece[2, i]
                        else:
                            corners = middle[0, i], middle[1, i], middle[2, i]
                        stack[top, 0, i], stack[top, 1, i], stack[top, 2, i] = corners
                    levels[top] = level
                    top += 1
                continue
            a0 = np.float64(F32(piece[1, 0] - piece[0, 0]))
            a1 = np.float64(F32(piece[1, 1] - piece[0, 1]))
            a2 = np.float64(F32(piece[1, 2] - piece[0, 2]))
            b0 = np.float64(F32(piece[2, 0] - piece[0, 0]))
            b1 = np.float64(F32(piece[2, 1] - piece[0, 1]))
            b2 = np.float64(F32(piece[2, 2] - piece[0, 2]))
            c0, c1, c2 = a1 * b2 - a2 * b1, -a0 * b2 + a2 * b0, a0 * b1 - a1 * b0
            area = area_factor * math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
            areas[je, jz, zenith] += area
            for band in range(band_count):
                if leaf[t]:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 1]
                    optical[je, jz, band, 1] += area * band_optics[band, je, 2]
                else:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 0]
    return areas, optical


@njit(cache=True, error_model="numpy")
def _fraction_below(z, z0, z1, z2):
    """The fraction of a planar triangle's area below height *z*, its vertices' heights z0 ≤ z1 ≤ z2 (the height is
    linear over it: piecewise quadratic in z)."""
    if z <= z0:
        return 0. if z < z0 or z2 > z0 else 1.
    if z >= z2:
        return 1.
    if z <= z1:
        return (z - z0) * (z - z0) / ((z2 - z0) * (z1 - z0))
    return 1. - (z2 - z) * (z2 - z) / ((z2 - z0) * (z2 - z1))


@njit(cache=True, error_model="numpy")
def _exact_structure(points, species, leaf, band_optics, tops, layer_count, class_count):
    """_structure's sums with each triangle's area shared between the layers exactly (the correction
    "exact_layers"): the part between each layer's planes, the top layer holding what lies above the height, what
    lies below the soil (z < 0) left out; stems at half their area, as s2v."""
    species_count, band_count = band_optics.shape[1], band_optics.shape[0]
    areas = np.zeros((species_count, layer_count, class_count))
    optical = np.zeros((species_count, layer_count, band_count, 2))
    for t in range(points.shape[0]):
        if species[t] < 0:
            continue
        je = species[t]
        p = points[t].astype(np.float64)
        c0 = (p[1, 1] - p[0, 1]) * (p[2, 2] - p[0, 2]) - (p[1, 2] - p[0, 2]) * (p[2, 1] - p[0, 1])
        c1 = (p[1, 2] - p[0, 2]) * (p[2, 0] - p[0, 0]) - (p[1, 0] - p[0, 0]) * (p[2, 2] - p[0, 2])
        c2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
        whole = (0.5 if leaf[t] else 0.25) * math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
        zenith = _zenith_class(points[t], class_count)
        z = np.sort(p[:, 2])
        for jz in range(layer_count):
            high = 1. if jz == 0 else _fraction_below(tops[jz], z[0], z[1], z[2])
            low = _fraction_below(tops[jz + 1], z[0], z[1], z[2]) if jz < layer_count - 1 else \
                _fraction_below(0., z[0], z[1], z[2])
            area = whole * (high - low)
            if area <= 0.:
                continue
            areas[je, jz, zenith] += area
            for band in range(band_count):
                if leaf[t]:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 1]
                    optical[je, jz, band, 1] += area * band_optics[band, je, 2]
                else:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 0]
    return areas, optical


# ── The radiosity solver (solver) ──


MACHINE_EPSILON = 2.2204460492503130808e-16


@njit(cache=True, nogil=True, error_model="numpy")
def product(x, out, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff):
    """hd_mv_mlt: out = A x, row by row of the receivers (*receiver_faces*: each receiver's first face)."""
    for r in range(indptr.shape[0] - 1):
        face = receiver_faces[r]
        upper = x[face]
        rho_sup = -reflectance[face]
        lower, tau_sup, rho_inf, tau_inf = 0.0, 0.0, 0.0, 0.0
        if receiver_translucent[r]:
            lower = x[face + 1]
            tau_sup = -transmittance[face]
            rho_inf = reflectance[face + 1]
            tau_inf = transmittance[face + 1]
        for index in range(indptr[r], indptr[r + 1]):
            j, count = faces[index], values[index]
            upper += x[j] * count * dff * (rho_sup if count > 0 else tau_inf)
            if receiver_translucent[r]:
                lower += x[j] * count * dff * (tau_sup if count > 0 else rho_inf)
        out[face] = upper
        if receiver_translucent[r]:
            out[face + 1] = lower
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def incoming(x, indptr, faces, values, receiver_faces, receiver_translucent, dff, n_faces):
    """Each face's irradiance from the near field (n_faces,): Σ_j |c_ij| dFF B_j over the emitter faces its disc sees,
    the upper face's from the positive counts, a leaf's lower face's from the negative ones; receivers beyond
    *n_faces* (the sensors) left out."""
    out = np.zeros(n_faces)
    for r in range(indptr.shape[0] - 1):
        face = receiver_faces[r]
        if face >= n_faces:
            continue
        for index in range(indptr[r], indptr[r + 1]):
            j, count = faces[index], values[index]
            if count > 0:
                out[face] += x[j] * count * dff
            elif receiver_translucent[r]:
                out[face + 1] -= x[j] * count * dff
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def _norm(vector):
    total = 0.0
    for i in range(vector.shape[0]):
        total += vector[i] * vector[i]
    return np.sqrt(total)


@njit(cache=True, nogil=True, error_model="numpy")
def _inner(a, b):
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


@njit(cache=True, nogil=True, error_model="numpy")
def _add_scaled(out, vector, scale):
    """v_mltadd(out, vector, scale, out): out += scale · vector (a copy when scale is 0, a plain sum when 1)."""
    if scale == 0.0:
        return
    for i in range(out.shape[0]):
        out[i] = out[i] + vector[i] if scale == 1.0 else out[i] + scale * vector[i]


@njit(cache=True, nogil=True, error_model="numpy")
def _upper_solve(matrix, rhs, size):
    """Usolve on the leading size × size block of an upper triangular matrix."""
    out = np.zeros(size)
    last = size - 1
    while last >= 0 and rhs[last] == 0.0:
        out[last] = 0.0
        last -= 1
    for i in range(last, -1, -1):
        total = 0.0
        for j in range(i + 1, last + 1):
            total += matrix[i, j] * out[j]
        out[i] = (rhs[i] - total) / matrix[i, i]
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def mgcr(rhs, indptr, faces, values, receiver_faces, receiver_translucent, reflectance, transmittance, dff,
         tolerance=1e-6, krylov=100, limit=1000):
    """hd_mgcr from x = 0: (x, steps)."""
    dimension = rhs.shape[0]
    x = np.zeros(dimension)
    residual = np.zeros(dimension)
    directions = np.zeros((krylov, dimension))
    alpha = np.zeros(krylov)
    beta = np.zeros(krylov)
    upper = np.zeros((krylov, krylov))
    initial = 0.0
    steps = 0
    done = False
    while steps < limit:
        product(x, residual, indptr, faces, values, receiver_faces, receiver_translucent, reflectance,
                transmittance, dff)
        for k in range(dimension):
            residual[k] = rhs[k] - residual[k]
        norm = _norm(residual)
        dd = norm
        if steps == 0:
            initial = abs(norm)
        if norm == 0.0:
            break
        directions[0, :] = residual
        i = 0
        while i < krylov and steps < limit:
            steps += 1
            product(directions[i], residual, indptr, faces, values, receiver_faces, receiver_translucent,
                    reflectance, transmittance, dff)
            if i < krylov - 1:
                following = directions[i + 1]
                following[:] = residual
                for j in range(i):
                    beta[j] = _inner(directions[j + 1], following)
                    _add_scaled(following, directions[j + 1], -beta[j])
                beta[i] = norm = _norm(following)
                if norm <= MACHINE_EPSILON * initial:
                    i -= 1
                    done = True
                    break
                scale = 1.0 / norm
                if scale == 0.0:
                    following[:] = 0.0
                elif scale != 1.0:
                    for k in range(dimension):
                        following[k] = scale * following[k]
                alpha[i] = _inner(directions[0], following)
            else:
                for j in range(i):
                    beta[j] = _inner(directions[j + 1], residual)
                norm = _inner(residual, residual)
                for j in range(i):
                    norm -= beta[j] * beta[j]
                if np.sqrt(abs(norm)) <= MACHINE_EPSILON * initial:
                    i -= 1
                    done = True
                    break
                if norm < 0.0:
                    i -= 1
                    steps -= 1
                    break
                beta[i] = np.sqrt(norm)
                alpha[i] = _inner(directions[0], residual)
                for j in range(i):
                    alpha[i] -= beta[j] * alpha[j]
                alpha[i] /= beta[i]
            for j in range(krylov):
                upper[j, i] = beta[j]
            ratio = alpha[i] / dd
            if abs(ratio - 1.0) <= MACHINE_EPSILON * initial:
                dd = 0.0
            else:
                ratio = 1.0 - ratio * ratio
                if ratio < 0.0:
                    break
                dd *= np.sqrt(ratio)
            if dd <= initial * tolerance:
                done = True
                break
            i += 1
        if i >= krylov:
            i = krylov - 1
        coefficients = _upper_solve(upper, alpha, i + 1)
        for j in range(i + 1):
            _add_scaled(x, directions[j], coefficients[j])
        if done:
            break
    return x, steps


# ── The near field's rows on the GPU (gpu) ──


@njit(cache=True)
def _count_rows(faces, n_faces, size, separate):
    """The rows of a chunk's face images (receivers, 2, size, size): (lengths, faces, counts), as _receiver_rows
    (*separate*: its FACE_COUNTS)."""
    count = np.zeros(n_faces, dtype=np.int64)
    count_lower = np.zeros(n_faces, dtype=np.int64)
    lengths = np.zeros(faces.shape[0], dtype=np.int64)
    out_faces, out_counts = [0], [0]
    out_faces.pop()
    out_counts.pop()
    for k in range(faces.shape[0]):
        touched = [0]
        touched.pop()
        for i in range(size):
            for j in range(size):
                face = faces[k, 0, i, j]
                if face >= 0:
                    if count[face] == 0:
                        touched.append(face)
                    count[face] += 1
                face = faces[k, 1, i, j]
                if face >= 0:
                    if count[face] == 0 and count_lower[face] == 0:
                        touched.append(face)
                    if separate:
                        count_lower[face] += 1
                    else:
                        count[face] -= 1
        ordered = np.sort(np.array(touched, dtype=np.int64)) if len(touched) else np.zeros(0, dtype=np.int64)
        previous = -1
        for face in ordered:
            if face == previous:
                continue
            previous = face
            if count[face] != 0:
                out_faces.append(face)
                out_counts.append(count[face])
                lengths[k] += 1
            if count_lower[face] != 0:
                out_faces.append(face)
                out_counts.append(-count_lower[face])
                lengths[k] += 1
            count[face] = count_lower[face] = 0
    return lengths, np.array(out_faces, dtype=np.int64), np.array(out_counts, dtype=np.int64)


# ── Warm-up ──

def warm_up() -> None:
    """Every kernel compiled (or loaded from numba's cache) on a tiny scene of two triangles and a sensor: the direct
    light (an infinite canopy and a closed one, legacy and full precision, the corrections on and off), radiosity and
    mixed radiosity (the near field, SAIL's layers, the solver), on a small screen. metafspm's warm_up runs it
    (python -m openalea.metafspm warm_up)."""
    from openalea.caribu import api
    leaf = [(0., 0., 1.), (1., 0., 1.), (0., 1., 1.)]
    stem = [(0.5, 0.5, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.9)]
    triangles, materials = [leaf, stem], [(0.06, 0.07), (0.13,)]
    lights, domain, sensors = [(1., (0.3, 0.1, -0.9))], (-1., -1., 2., 2.), [[(0., 0., 0.2), (0.1, 0., 0.2),
                                                                                (0., 0.1, 0.2)]]
    for precision, corrections in (("legacy", None), ("full", None), ("full", "recommended")):
        options = dict(screen_size=64, precision=precision, corrections=corrections, workers=1)
        api.raycasting(triangles, materials, lights, domain=domain, sensors=sensors, **options)
        api.raycasting(triangles, materials, lights, **options)
        api.radiosity(triangles, materials, lights, sensors=sensors, **options)
        api.mixed_radiosity(triangles, materials, lights, domain, 0.15, 0.5, 2, 1.2, **options)
