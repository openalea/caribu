"""
The near field of radiosity: what each receiver sees of its neighbours (canestrad's calc_FF_Bfar, proj_ortho and NFF,
doc/legacy.rst). Each receiver's hemispheres are drawn on an NB × NB disc (the Nusselt analogue: a
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

from . import kernels, periodic
from .kernels import FACE_COUNTS, PROFILE_BOUNDS, PROFILE_INTERPOLATION, PROJECTED_FACE, VOXEL_FLOOR, _receiver_rows

F32 = np.float32

# the corrections of the near field (corrections.resolve's names), as the kernels' option bits (kernels.VOXEL_FLOOR...)
OPTION_NAMES = {"voxel_floor": VOXEL_FLOOR, "face_counts": FACE_COUNTS, "projected_face": PROJECTED_FACE,
                "profile_interpolation": PROFILE_INTERPOLATION, "profile_bounds": PROFILE_BOUNDS}


def options(corrections) -> int:
    """The kernels' option bits of a set of correction names."""
    return sum(bit for name, bit in OPTION_NAMES.items() if name in corrections)


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


# --- The receivers ---


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
    centres = kernels.legacy_centres(points)
    largest = np.full(len(points), -1, dtype=F32)
    for vertex in range(3):
        difference = (centres - points[:, vertex]).astype(F32)
        square = ((difference[:, 0] * difference[:, 0]).astype(F32) + (difference[:, 1] * difference[:, 1]).astype(F32)
                  ).astype(F32) + (difference[:, 2] * difference[:, 2]).astype(F32)
        largest = np.maximum(largest, square.astype(F32))
    return np.sqrt(largest.astype(np.float64)).astype(F32)


def counts(points, labels, translucent, first_face, n_faces, normals, centres, radius, sensors=None,
           sensor_normals=None, sensor_centres=None, size=52, workers=None, chunk=64, grid=None, profile=None,
           optics=None, bias=True, device="cpu", options=0):
    """The near-field rows of every receiver, the accepted triangles then the sensors: (indptr, faces, counts), a
    CSR matrix of net pixel counts (rows: receivers; columns: emitter faces). *radius*: the sphere's radius around
    each receiver (emitters whose centre lies farther are left to the far field); *grid*: the candidates' boxes
    (`grid`; one box holding all by default). With a *profile* (the far field's fluxes, as canestrad reads them)
    and the receivers' *optics* (receivers, 4: ρ, τ of the upper face, of the lower one), also each receiver's
    far field (receivers, 2) and its levels' weights (receivers, 4, levels). *bias* False: canestrad's `-B`.
    *device* "cuda": the receivers computed on the GPU (gpu.receiver_rows, the same results), in chunks of
    *chunk* × 64 receivers. *options*: the corrections' bits (options())."""
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
                              grid["infinite"], grid["period"], bias, table, rows, thickness, form_factor(size),
                              options)

    if device == "cuda":
        from . import gpu
        parts = gpu.receiver_rows(total, chunk * 64, receivers, receiver_normals, receiver_centres,
                                  receiver_translucent, receiver_labels, receiver_optics, emitters, spread,
                                  translucent, first_face, labels, n_faces, float(radius), size, pixel, disc,
                                  grid["origin"], float(grid["side"]), grid["voxels"], grid["sizes"], grid["indptr"],
                                  grid["members"], grid["infinite"], grid["period"], bias, table, rows, thickness,
                                  form_factor(size), options)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(run, starts))
    lengths = np.concatenate([part[0] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    faces = np.concatenate([part[1] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    values = np.concatenate([part[2] for part in parts]) if parts else np.zeros(0, dtype=np.int64)
    if profile is None:
        return indptr, faces, values
    far = np.concatenate([part[3] for part in parts]) if parts else np.zeros((0, 2))
    weights = np.concatenate([part[4] for part in parts]) if parts else np.zeros((0, 4, rows))
    return (indptr, faces, values), far, weights


def form_factor(size=52) -> float:
    """dFF: a pixel's form factor, 4 / (π NB²) (the disc's area π over its NB² pixels of side 2 / NB, not
    renormalised to the pixels inside the disc)."""
    return 4 / float(size * size) / np.pi
