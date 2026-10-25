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


@njit(cache=True, nogil=True)
def _height(i, j, size, heights):
    """sqrtab: the hemisphere's height over pixel (i, j), −1 outside the disc."""
    half = int(size / 2.0)
    i = i if i < half else size - i - 1
    j = j if j < half else size - j - 1
    if j > i:
        i, j = j, i
    return heights[int(i + j * (half - (j + 1) / 2.0))]


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
def _crossing(a, b, out):
    """Pz0: where segment a–b crosses z = 0 (k in float32, as the C++ computes it)."""
    k = np.float64(F32(b[2] / F32(b[2] - a[2])))
    out[0] = F32(k * a[0] + (1. - k) * b[0])
    out[1] = F32(k * a[1] + (1. - k) * b[1])
    out[2] = F32(0.)


@njit(cache=True, nogil=True)
def _solve3(matrix, rhs):
    """solve3: Cramer's rule in double, the solution written into rhs; False when singular."""
    m = matrix
    determinant = 0.0
    determinant += m[0, 0] * (m[1, 1] * m[2, 2] - (m[2, 1] * m[1, 2]))
    determinant -= m[0, 1] * (m[1, 0] * m[2, 2] - (m[2, 0] * m[1, 2]))
    determinant += m[0, 2] * (m[1, 0] * m[2, 1] - (m[2, 0] * m[1, 1]))
    if determinant == 0.0:
        return False
    solution = np.zeros(3)
    saved = np.zeros(3)
    for j in range(3):
        for i in range(3):
            saved[i] = m[i, j]
            m[i, j] = rhs[i]
        solution[j] += m[0, 0] * (m[1, 1] * m[2, 2] - (m[2, 1] * m[1, 2]))
        solution[j] -= m[0, 1] * (m[1, 0] * m[2, 2] - (m[2, 0] * m[1, 2]))
        solution[j] += m[0, 2] * (m[1, 0] * m[2, 1] - (m[2, 0] * m[1, 1]))
        solution[j] /= determinant
        for i in range(3):
            m[i, j] = saved[i]
    for i in range(3):
        rhs[i] = solution[i]
    return True


@njit(cache=True, nogil=True)
def _edge_plane(projected, i, j, plane):
    """init_MPE: the plane through the origin and edge i–j, normalised on its first non-zero coefficient; True when
    degenerate."""
    d0 = np.float64(F32(F32(projected[i, 1] * projected[j, 2]) - F32(projected[j, 1] * projected[i, 2])))
    d1 = np.float64(F32(-F32(projected[j, 0] * projected[i, 2]) + F32(projected[i, 0] * projected[j, 2])))
    d2 = np.float64(F32(F32(projected[i, 0] * projected[j, 1]) - F32(projected[j, 0] * projected[i, 1])))
    if d0 != 0.0:
        plane[0], plane[1], plane[2] = 1., -d1 / d0, d2 / d0
        return False
    if d1 != 0.0:
        plane[1], plane[0], plane[2] = 1., -d0 / d1, -d2 / d1
        return False
    if d2 != 0.0:
        plane[2], plane[0], plane[1] = 1., d0 / d2, -d1 / d2
        return False
    return True


@njit(cache=True, nogil=True)
def _project(emitter, triangle, shift, origin, u, v, w, receiver_translucent, emitter_opaque, emitter_centre,
             emitter_normal, upper_owner, upper_depth, lower_owner, lower_depth, size, pixel, heights):
    """proj_ortho: the emitter triangle (float32 (3, 3), translated by *shift* in x, y) drawn on the receiver's
    discs, its index *emitter* kept where nearest."""
    if emitter_opaque:
        er0 = F32(emitter_centre[0] - origin[0])
        er1 = F32(emitter_centre[1] - origin[1])
        er2 = F32(emitter_centre[2] - origin[2])
        if _dot(er0, er1, er2, emitter_normal[0], emitter_normal[1], emitter_normal[2]) > 0:
            return
    projected = np.zeros((3, 3), dtype=np.float32)
    above, below = np.full(2, -1, dtype=np.int64), np.full(2, -1, dtype=np.int64)
    on = np.full(2, -1, dtype=np.int64)
    side = 0
    for i in range(3):
        x = F32(F32(triangle[i, 0] + shift[0]) - origin[0])
        y = F32(F32(triangle[i, 1] + shift[1]) - origin[1])
        z = F32(triangle[i, 2] - origin[2])
        projected[i, 0] = _dot(x, y, z, u[0], u[1], u[2])
        projected[i, 1] = _dot(x, y, z, v[0], v[1], v[2])
        projected[i, 2] = _dot(x, y, z, w[0], w[1], w[2])
        if projected[i, 2] > 1e-6:
            side += 1
            if above[0] == -1:
                above[0] = i
            else:
                above[1] = i
        elif projected[i, 2] < -1e-6:
            side -= 1
            if below[0] == -1:
                below[0] = i
            else:
                below[1] = i
        else:
            side += 4
            if on[0] == -1:
                on[0] = i
            else:
                on[1] = i
    middle = np.zeros((2, 3), dtype=np.float32)
    apex = np.zeros((3, 3), dtype=np.float32)
    if side == 4:  # one vertex on the plane: two triangles
        parts = 2
        middle[0] = projected[on[0]]
        apex[0] = projected[above[0]]
        apex[1] = projected[below[0]]
        _crossing(apex[0], apex[1], middle[1])
        mirrored = 0
    elif side == 1:  # one vertex below: three triangles
        parts = 3
        apex[0] = projected[below[0]]
        apex[1] = projected[above[0]]
        apex[2] = projected[above[1]]
        _crossing(apex[0], apex[1], middle[0])
        _crossing(apex[0], apex[2], middle[1])
        mirrored = 1
    elif side == -1:  # one vertex above: three triangles
        parts = 3
        apex[0] = projected[above[0]]
        apex[1] = projected[below[0]]
        apex[2] = projected[below[1]]
        _crossing(apex[0], apex[1], middle[0])
        _crossing(apex[0], apex[2], middle[1])
        mirrored = 0
    else:
        parts = 1
        mirrored = 0 if (side == 3 or side == 6 or side == 9) else 1
        middle[0] = projected[0]
        middle[1] = projected[1]
        apex[0] = projected[2]
    points = np.zeros((3, 3), dtype=np.float32)
    matrix = np.zeros((3, 3))
    plane = np.zeros(3)
    edges = np.zeros((3, 4))
    spherical = np.zeros((3, 3), dtype=np.float32)
    rectangle = np.zeros(4)
    bounds = np.zeros(4)
    for part in range(parts):
        if part == 0:
            points[0], points[1], points[2] = middle[0], middle[1], apex[0]
        elif part == 1:
            points[0], points[1], points[2] = middle[1], middle[0], apex[1]
            mirrored = (mirrored + 1) % 2
        else:
            points[0], points[1], points[2] = middle[1], apex[1], apex[2]
        if mirrored == 1:
            for i in range(3):
                points[i, 2] = abs(points[i, 2])
        axis_zero = np.zeros(2, dtype=np.int64)
        for i in range(3):
            if points[i, 0] == 0 and points[i, 1] == 0.:
                points[i, 0] = points[i, 1] = F32(0.0000001)
            else:
                for j in range(2):
                    if points[i, j] == 0:
                        if axis_zero[j] != 0:
                            points[i, j] = F32(0.0000001)
                            points[axis_zero[j] - 1, j] = F32(0.0000001)
                        else:
                            axis_zero[j] = i + 1
            if points[i, 2] < 1e-6:
                points[i, 2] = F32(np.float64(points[i, 2]) + 0.000001)
            for j in range(3):
                matrix[i, j] = points[i, j]
            plane[i] = -1
        if not ((mirrored == 0 or (receiver_translucent and mirrored == 1)) and _solve3(matrix, plane)):
            continue
        centre_x = F32(np.float64(F32(F32(points[0, 0] + points[1, 0]) + points[2, 0])) / 3.)
        centre_y = F32(np.float64(F32(F32(points[0, 1] + points[1, 1]) + points[2, 1])) / 3.)
        centre_z = F32(np.float64(F32(F32(points[0, 2] + points[1, 2]) + points[2, 2])) / 3.)
        centre_x, centre_y, centre_z = _sphere(centre_x, centre_y, centre_z)
        height = _sqrt(np.float64(F32(F32(F32(1) - F32(centre_x * centre_x)) - F32(centre_y * centre_y))))
        for i in range(3):
            if _edge_plane(points, i, (i + 1) % 3, edges[i]):
                return
            spherical[i, 0], spherical[i, 1], spherical[i, 2] = _sphere(points[i, 0], points[i, 1], points[i, 2])
            edges[i, 3] = edges[i, 0] * centre_x + edges[i, 1] * centre_y + edges[i, 2] * height
        rectangle[0] = rectangle[2] = 1.5
        rectangle[1] = rectangle[3] = -1.5
        for i in range(3):
            square0, square1, square2 = edges[i, 0] * edges[i, 0], edges[i, 1] * edges[i, 1], edges[i, 2] * edges[i, 2]
            cross = edges[i, 0] * edges[i, 1]
            total = square0 + square1 + square2
            for j in range(2):
                part_square = (square0 if j == 0 else square1) + square2
                extreme = _sqrt(part_square / total)
                if part_square == 0.:
                    return
                position = (-cross * extreme) / part_square
                high_index = 3 - j * 2
                if spherical[(i + 1) % 3, j] > spherical[i, j]:
                    first, second = i, (i + 1) % 3
                else:
                    second, first = i, (i + 1) % 3
                other = (j + 1) % 2
                highest = max(np.float64(spherical[first, other]), np.float64(spherical[second, other]))
                lowest = min(np.float64(spherical[first, other]), np.float64(spherical[second, other]))
                if (spherical[first, j] > 0) != (spherical[second, j] > 0):
                    a = (position - spherical[first, j]) / np.float64(F32(spherical[second, j] - spherical[first, j]))
                    ordinate = (1 - a) * spherical[first, other] + a * spherical[second, other]
                    if (ordinate > 0) != (extreme > 0):
                        position = -position
                        extreme = -extreme
                if position > spherical[first, j] and position < spherical[second, j]:
                    bounds[high_index] = extreme if extreme > highest else highest
                    bounds[high_index - 1] = extreme if extreme < lowest else lowest
                else:
                    bounds[high_index] = highest
                    bounds[high_index - 1] = lowest
                rectangle[high_index] = max(bounds[high_index], rectangle[high_index])
                rectangle[high_index - 1] = min(bounds[high_index - 1], rectangle[high_index - 1])
        first_i = int(((rectangle[0] + 1) * size - 1) / 2.0)
        first_i = 0 if first_i == 0 else first_i - 1
        last_i = int(((rectangle[1] + 1) * size - 1) / 2.0)
        last_i = last_i + 1 if last_i < size - 1 else last_i
        first_j = int(((rectangle[2] + 1) * size - 1) / 2.0)
        first_j = 0 if first_j == 0 else first_j - 1
        last_j = int(((rectangle[3] + 1) * size - 1) / 2.0)
        last_j = last_j + 1 if last_j < size - 1 else last_j
        for i in range(first_i, last_i + 1):
            x = pixel[i]
            for j in range(first_j, last_j + 1):
                h = _height(i, j, size, heights)
                if h == -1:
                    continue
                y = pixel[j]
                inside = 3
                for e in range(3):
                    inside -= 1 if ((x + edges[e, 1] * y + edges[e, 2] * h > 0) != (edges[e, 3] > 0)) else 0
                if inside != 3:
                    continue
                distance = plane[0] * x + plane[1] * y + plane[2] * h
                if mirrored == 0:
                    if distance < upper_depth[i, j]:
                        upper_depth[i, j] = F32(distance)
                        upper_owner[i, j] = emitter
                elif distance < lower_depth[i, j]:
                    lower_depth[i, j] = F32(distance)
                    lower_owner[i, j] = emitter


# --- The receivers ---

@njit(cache=True, nogil=True)
def _frame(triangle, normal):
    """init_proj: the receiver's frame, u = P0 − P1 normalised, w its normal, v = w × u (float32)."""
    u0 = F32(triangle[0, 0] - triangle[1, 0])
    u1 = F32(triangle[0, 1] - triangle[1, 1])
    u2 = F32(triangle[0, 2] - triangle[1, 2])
    norm = np.float64(np.sqrt(F32(F32(F32(u0 * u0) + F32(u1 * u1)) + F32(u2 * u2))))
    u = np.array([F32(u0 / norm), F32(u1 / norm), F32(u2 / norm)], dtype=np.float32)
    w = normal.copy()
    v = np.array([F32(F32(u[2] * w[1]) - F32(u[1] * w[2])), F32(F32(u[0] * w[2]) - F32(u[2] * w[0])),
                  F32(F32(u[1] * w[0]) - F32(u[0] * w[1]))], dtype=np.float32)
    return u, v, w


@njit(cache=True, nogil=True)
def _receiver_rows(start, stop, receivers, receiver_normals, receiver_centres, receiver_translucent,
                   receiver_labels, points, normals, centres, translucent, first_face, labels, n_faces, square_radius,
                   size, pixel, heights):
    """The rows of receivers start..stop: (row lengths, emitter faces, net counts), each row's faces ascending."""
    upper_owner = np.full((size, size), -1, dtype=np.int64)
    lower_owner = np.full((size, size), -1, dtype=np.int64)
    upper_depth = np.full((size, size), UPPER_EMPTY, dtype=np.float32)
    lower_depth = np.full((size, size), LOWER_EMPTY, dtype=np.float32)
    active_face = np.zeros(points.shape[0], dtype=np.int64)
    projected_face = np.zeros(n_faces, dtype=np.int64)
    count = np.zeros(n_faces, dtype=np.int64)
    lengths = np.zeros(stop - start, dtype=np.int64)
    faces_out = []
    counts_out = []
    no_shift = np.zeros(2, dtype=np.float32)
    for r in range(start, stop):
        upper_owner[:] = -1
        upper_depth[:] = UPPER_EMPTY
        if receiver_translucent[r]:
            lower_owner[:] = -1
            lower_depth[:] = LOWER_EMPTY
        projected_face[:] = 0
        origin = receiver_centres[r]
        u, v, w = _frame(receivers[r], receiver_normals[r])
        soil = receiver_labels[r] == 0
        for e in range(points.shape[0]):
            if r < points.shape[0] and e == r:
                continue
            if soil and labels[e] == 0:
                continue
            t0, t1, t2 = centres[e, 0], centres[e, 1], centres[e, 2]
            d0, d1, d2 = F32(origin[0] - t0), F32(origin[1] - t1), F32(origin[2] - t2)
            if not (np.float64(F32(F32(F32(d0 * d0) + F32(d1 * d1)) + F32(d2 * d2))) < square_radius):
                continue
            direction0, direction1, direction2 = F32(t0 - origin[0]), F32(t1 - origin[1]), F32(t2 - origin[2])
            facing = _dot(direction0, direction1, direction2, normals[e, 0], normals[e, 1], normals[e, 2])
            if facing == 0:
                continue
            face = first_face[e] if (not translucent[e] or facing < 0) else first_face[e] + 1
            active_face[e] = face
            if projected_face[face] != 0:  # the anti-doublon flags: once untranslated
                continue
            projected_face[face] = 2
            _project(e, points[e], no_shift, origin, u, v, w, receiver_translucent[r], not translucent[e],
                     centres[e], normals[e], upper_owner, upper_depth, lower_owner, lower_depth, size, pixel, heights)
        touched = []
        for i in range(size):
            for j in range(size):
                if _height(i, j, size, heights) < 0:
                    continue
                e = upper_owner[i, j]
                if e >= 0:
                    face = active_face[e]
                    if count[face] == 0:
                        touched.append(face)
                    count[face] += 1
                if receiver_translucent[r]:
                    e = lower_owner[i, j]
                    if e >= 0:
                        face = active_face[e]
                        if count[face] == 0:
                            touched.append(face)
                        count[face] -= 1
        touched_array = np.empty(len(touched), dtype=np.int64)
        for index in range(len(touched)):
            touched_array[index] = touched[index]
        touched_array = np.sort(touched_array)
        length = 0
        previous = -1
        for face in touched_array:
            if face == previous:
                continue
            previous = face
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
    return lengths, faces, values


def counts(points, labels, translucent, first_face, n_faces, normals, centres, radius, sensors=None,
           sensor_normals=None, sensor_centres=None, size=52, workers=None, chunk=64):
    """The near-field rows of every receiver, the accepted triangles then the sensors: (indptr, faces, counts), a
    CSR matrix of net pixel counts (rows: receivers; columns: emitter faces). *radius*: the sphere's radius around
    each receiver (emitters whose centre lies farther are left to the far field)."""
    pixel, heights = disc_tables(size)
    receivers, receiver_normals, receiver_centres = points, normals, centres
    receiver_translucent, receiver_labels = translucent, labels
    if sensors is not None and len(sensors):
        receivers = np.concatenate([points, sensors]).astype(np.float32)
        receiver_normals = np.concatenate([normals, sensor_normals]).astype(np.float32)
        receiver_centres = np.concatenate([centres, sensor_centres]).astype(np.float32)
        receiver_translucent = np.concatenate([translucent, np.zeros(len(sensors), dtype=bool)])
        receiver_labels = np.concatenate([labels, np.arange(1, len(sensors) + 1, dtype=np.int64)])
    total = len(receivers)
    starts = list(range(0, total, chunk))

    def rows(start):
        return _receiver_rows(start, min(start + chunk, total), receivers, receiver_normals, receiver_centres,
                              receiver_translucent, receiver_labels, points, normals, centres, translucent,
                              first_face, labels, n_faces, radius * radius, size, pixel, heights)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(rows, starts))
    lengths = np.concatenate([part[0] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    faces = np.concatenate([part[1] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    values = np.concatenate([part[2] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    return indptr, faces, values


def form_factor(size=52) -> float:
    """dFF: a pixel's form factor, 4 / (π NB²) (the disc's area π over its NB² pixels of side 2 / NB, not
    renormalised to the pixels inside the disc)."""
    return 4 / float(size * size) / np.pi
