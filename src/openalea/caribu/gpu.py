"""
The near field on a GPU (numba.cuda), for the full and the nested radiosity: one thread per receiver fills its
discs, projects its candidates and sums its far field with the CPU kernels' own functions (nearfield's), compiled
as device functions: the same arithmetic in the same order, sequential within a receiver, so the counts and far
field are the CPU's (CUDA without fast math: no fused multiply-add). The discs' faces are counted into rows on the
CPU.

    if gpu.available():
        rows = nearfield.counts(..., device="cuda")

Without a GPU, numba's CUDA simulator (NUMBA_ENABLE_CUDASIM=1, before numba is imported) runs the same kernel in
Python, slowly: the tests use it on small scenes.
"""
import types

import numpy as np
from numba import cuda, njit

from . import nearfield

F32 = np.float32
DEVICE_FUNCTIONS = ("_sqrt", "_dot", "_sphere", "_crossing", "_determinant", "_solve3", "_edge_plane", "_copy",
                    "_project", "_frame", "_level", "_environment", "_add_box", "_neighbourhood", "_candidates",
                    "_faces", "_clear")
_compiled = {}


def available() -> bool:
    """Whether a CUDA GPU, or numba's CUDA simulator, is there."""
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _receivers(start, count, receivers, receiver_normals, receiver_centres, receiver_translucent, receiver_labels,
               receiver_optics, emitters, spreads, translucent, first_face, labels, radius, size, pixel, disc, origin,
               side, voxels, box_sizes, box_indptr, box_members, infinite, period, bias, limit, table, rows, thickness,
               dff, owners, depths, faces, frames, works, works64, active_faces, flags, boxes, shifts, cells, moves,
               corners, bfcs, far, weights):
    """A receiver per thread: its discs' faces, far field and levels' weights (nearfield._receiver_rows' body),
    in its own slices of the work arrays."""
    k = cuda.grid(1)
    if k >= count:
        return
    r = start + k
    _clear(owners[k], depths[k], size)
    flags_k = flags[k]
    for face in range(flags_k.shape[0]):
        flags_k[face] = 0
    frame = frames[k]
    _frame(receivers, receiver_normals, receiver_centres, r, frame)
    box_count = _neighbourhood(receiver_centres[r], origin, side, voxels, box_sizes, infinite, period, radius,
                               boxes[k], shifts[k], cells[k], moves[k], corners[k])
    _candidates(r, receiver_translucent[r], receiver_labels[r] == 0, frame, emitters, spreads, translucent,
                first_face, labels, radius, bias, limit, box_count, boxes[k], shifts[k], box_indptr, box_members,
                flags_k, active_faces[k], owners[k], depths[k], size, pixel, disc, works[k], works64[k])
    _faces(receiver_translucent[r], receiver_optics[r], frame, owners[k], active_faces[k], faces[k], size, pixel,
           disc, radius, table, rows, thickness, dff, bfcs[k], far[k])
    for row in range(4):
        for level in range(rows):
            weights[k, row, level] = bfcs[k, row, level] * dff


def _kernel():
    """nearfield's functions cloned into a namespace of device functions, and the kernel compiled in it."""
    if "kernel" not in _compiled:
        namespace = dict(vars(nearfield))
        namespace["cuda"] = cuda
        for name in DEVICE_FUNCTIONS:
            function = getattr(nearfield, name).py_func
            clone = types.FunctionType(function.__code__, namespace, name, function.__defaults__,
                                       function.__closure__)
            namespace[name] = cuda.jit(device=True)(clone)
        clone = types.FunctionType(_receivers.__code__, namespace, "_receivers")
        _compiled["kernel"] = cuda.jit(clone)
    return _compiled["kernel"]


@njit(cache=True)
def _count_rows(faces, n_faces, size):
    """The rows of a chunk's face images (receivers, 2, size, size): (lengths, faces, counts), as _receiver_rows."""
    count = np.zeros(n_faces, dtype=np.int64)
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
                    if count[face] == 0:
                        touched.append(face)
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
            count[face] = 0
    return lengths, np.array(out_faces, dtype=np.int64), np.array(out_counts, dtype=np.int64)


def receiver_rows(total, chunk, receivers, receiver_normals, receiver_centres, receiver_translucent, receiver_labels,
                  receiver_optics, emitters, spreads, translucent, first_face, labels, n_faces, radius, size, pixel,
                  disc, origin, side, voxels, box_sizes, box_indptr, box_members, infinite, period, bias, table, rows,
                  thickness, dff, threads=64):
    """nearfield's rows of every receiver computed on the GPU, *chunk* receivers per launch: [(lengths, faces,
    counts, far, weights)] per chunk, as nearfield._receiver_rows'."""
    kernel = _kernel()
    limit = np.inf if bias else -1. / radius
    rows_count = max(rows, 1)
    constant = [cuda.to_device(np.ascontiguousarray(array)) for array in (
        receivers, receiver_normals, receiver_centres, receiver_translucent, receiver_labels, receiver_optics,
        emitters, spreads, translucent, first_face, labels)]
    discs = [cuda.to_device(np.ascontiguousarray(array)) for array in (pixel, disc, origin)]
    cells_arrays = [cuda.to_device(np.ascontiguousarray(array)) for array in (voxels, box_sizes, box_indptr,
                                                                                box_members)]
    profile = [cuda.to_device(np.ascontiguousarray(period)), cuda.to_device(np.ascontiguousarray(
        table if len(table) else np.zeros(1)))]
    parts = []
    for start in range(0, total, chunk):
        count = min(chunk, total - start)
        work = [cuda.device_array(shape, dtype=dtype) for shape, dtype in (
            ((count, 2, size, size), np.int32), ((count, 2, size, size), np.float32),
            ((count, 2, size, size), np.int32), ((count, 5, 3), np.float32), ((count, 14, 3), np.float32),
            ((count, 9, 4), np.float64), ((count, emitters.shape[0]), np.int32), ((count, n_faces), np.uint8),
            ((count, 8), np.int64), ((count, 8, 2), np.float32), ((count, 2, 3), np.int64), ((count, 2), np.float32),
            ((count, 2, 3), np.float64), ((count, 4, rows_count), np.float64), ((count, 2), np.float64),
            ((count, 4, rows_count), np.float64))]
        blocks = (count + threads - 1) // threads
        kernel[blocks, threads](start, count, *constant[:9], constant[9], constant[10], float(radius), size,
                                discs[0], discs[1], discs[2], float(side), *cells_arrays, bool(infinite), profile[0],
                                bool(bias), float(limit), profile[1], rows, float(thickness), float(dff), *work)
        faces = work[2].copy_to_host()
        lengths, row_faces, values = _count_rows(faces, n_faces, size)
        parts.append((lengths, row_faces, values, work[14].copy_to_host(), work[15].copy_to_host()[:, :, :rows]))
    return parts
