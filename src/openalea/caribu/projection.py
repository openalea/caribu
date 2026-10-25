"""
Direct light: for each source, the scene seen along the beam on an orthographic z-buffer of T × T pixels; the pixels
each triangle owns give its sunlit area (canestrad's Canopy::projplan and colorie_triangle, B.6 of the description).

The legacy's numerics are reproduced, to compare pixel counts exactly: points and vectors in float32 (canestrad's
`reel`), its locals in double, the same operation order; the screen on the scene's bounds, lifted one length unit
above it; a pixel scale of (T − 1)/extent; a pixel owned by the triangle whose centre is nearest along the beam, the
first drawn winning ties; the sunlit (horizontal-equivalent) area of a pixel `Apix = du dv / T² / cos θ`.

    screen = legacy_screen(vmin, vmax, direction)
    owner = rasterise(points, screen, size)                   (T, T) triangle index, -1 where no triangle
    areas = projected_areas(owner, ...)                       per face, +Apix lit, -Apix for a leaf's other face
"""
import math

import numpy as np
from numba import njit

F32 = np.float32
EMPTY_DEPTH = np.float32(99999999999.9)


@njit(cache=True)
def _normalise(x, y, z):
    """Vecteur::normalise: the norm, the float32 square root (C++'s sqrt(float)) of the float32 components' float32 sum
    of squares, each component divided by it in double and stored as float32."""
    norm = np.float64(np.sqrt(F32(x * x + y * y + z * z)))
    return F32(np.float64(x) / norm), F32(np.float64(y) / norm), F32(np.float64(z) / norm)


@njit(cache=True)
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


@njit(cache=True)
def _to_image(point_x, point_y, point_z, shift, origin, v, w, u):
    """Homogene::chgt_base of (point + shift − origin), float32, left to right."""
    x = F32(F32(point_x + shift[0]) - origin[0])
    y = F32(F32(point_y + shift[1]) - origin[1])
    z = F32(F32(point_z + shift[2]) - origin[2])
    return (F32(F32(F32(v[0] * x) + F32(v[1] * y)) + F32(v[2] * z)),
            F32(F32(F32(w[0] * x) + F32(w[1] * y)) + F32(w[2] * z)),
            F32(F32(F32(u[0] * x) + F32(u[1] * y)) + F32(u[2] * z)))


@njit(cache=True)
def _colour_triangle(owner, depth, triangle, a, b, c, size, extent_x, extent_y, count_only):
    """colorie_triangle: rows and columns whose pixel centre lies in the triangle (apex *a*, base *b*-*c*, a flat
    base), the nearest along the beam kept (strict <: the first drawn wins ties). a, b, c: double (3,). With
    *count_only* (colorie_capteur, a sensor) the pixels where it would be the nearest are counted, nothing drawn.
    Returns that count."""
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
                        count += 1
                    else:
                        depth[i, j] = F32(z)
                        owner[i, j] = triangle
                z = z + dz
        x_left += slope_left
        x_right += slope_right
        z_left += vab
    return count


@njit(cache=True)
def _draw(owner, depth, triangle, points, shift, origin, v, w, u, du, dv, size, count_only):
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
        count += _colour_triangle(owner, depth, triangle, c, a, b, size, du, dv, count_only)
    if down:
        if up:
            k, j, i = j, i, last
        for q in range(3):
            a[q], b[q], c[q] = pp[i, q], pp[j, q], pp[k, q]
        count += _colour_triangle(owner, depth, triangle, a, b, c, size, du, dv, count_only)
    return count


@njit(cache=True)
def rasterise(points, copies, shifts, origin, v, w, u, du, dv, size):
    """(T, T) the index of the triangle owning each pixel (-1: none), and the depth buffer: each triangle of *points*
    (float32 (n, 3, 3), in scene order) drawn once per copy of *copies* (n, 3) bool (the original, +dx, +dy:
    periodic border copies) translated by *shifts* (3, 3) float32."""
    depth = np.full((size, size), EMPTY_DEPTH, dtype=np.float32)
    owner = np.full((size, size), -1, dtype=np.int64)
    for t in range(points.shape[0]):
        for copy in range(3):
            if copies[t, copy]:
                _draw(owner, depth, t, points[t], shifts[copy], origin, v, w, u, du, dv, size, False)
    return owner, depth


@njit(cache=True)
def sensor_areas(depth, points, seen, copies, shifts, origin, v, w, u, du, dv, size, pixel_area):
    """Each sensor's sunlit area (horizontal equivalent) on the scene's depth buffer: the pixels where it would be
    in front of every triangle (projplan's virtual sensors, after the scene). *seen*: whether the beam meets its
    upper face (only then is it counted). As canestrad, the area is added after each copy with the count so far."""
    areas = np.zeros(points.shape[0])
    owner = np.zeros((1, 1), dtype=np.int64)
    for s in range(points.shape[0]):
        if not seen[s]:
            continue
        pixels = 0
        for copy in range(3):
            if copies[s, copy]:
                pixels += _draw(owner, depth, s, points[s], shifts[copy], origin, v, w, u, du, dv, size, True)
                areas[s] += pixels * pixel_area
    return areas


@njit(cache=True)
def projected_areas(owner, n_faces, first_face, translucent, lit_upper, pixel_area):
    """Per face, its sunlit area (horizontal equivalent): +pixel_area per pixel for the lit face, −pixel_area for a
    leaf's other face (its transmitted light), summed in projplan's pixel order. *first_face*: each triangle's first
    face number (a leaf's: its upper face, the next its lower one); *lit_upper*: whether the beam meets a leaf's upper
    face (direction · normal < 0)."""
    areas = np.zeros(n_faces)
    size = owner.shape[0]
    for i in range(size):
        for j in range(size):
            t = owner[i, j]
            if t < 0:
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
