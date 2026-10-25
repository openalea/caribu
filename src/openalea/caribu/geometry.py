"""
The triangles' geometry: normals, areas, centroids, and the triangles canestrad rejects. A scene is an array of
vertices (n, 3, 3): per triangle its three points, x y z, counter-clockwise seen from the side its normal points to
(the upper face, `sup`).

The legacy reads the scene from text written with 6 decimals into float32 points (as_legacy_vertices); rejected
reproduces its validity tests on those points (Polygone::init): coincident vertices, collinear ones, a null normal.
Rejected triangles keep their place in the outputs (area 0, energies NaN).
"""
import numpy as np


def as_legacy_vertices(vertices) -> np.ndarray:
    """The vertices as canestrad reads them: written with 6 decimals (caribu's `%.6f`), parsed as doubles, stored as
    float32."""
    vertices = np.asarray(vertices, dtype=np.float64)
    written = np.array([float(f"{value:.6f}") for value in vertices.reshape(-1)], dtype=np.float64)
    return written.astype(np.float32).reshape(vertices.shape)


def normals(vertices) -> np.ndarray:
    """(n, 3): each triangle's unit normal, (P1 − P0) × (P2 − P0) normalised (zero for a degenerate one)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    return np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 0)


def areas(vertices) -> np.ndarray:
    """Each triangle's area."""
    vertices = np.asarray(vertices, dtype=np.float64)
    return 0.5 * np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1)


def centroids(vertices) -> np.ndarray:
    """(n, 3): each triangle's isobarycentre."""
    return np.asarray(vertices, dtype=np.float64).mean(axis=1)


def rejected(vertices) -> np.ndarray:
    """Whether canestrad rejects each triangle (Polygone::init's tests, on float32 points): vertices 0 and 1 or 0 and
    2 equal, vertices 1 and 2 equal, the three collinear (|û·v̂| > 0.9999999999), or a null normal."""
    points = np.asarray(vertices, dtype=np.float32)
    first, second = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    first_norm = np.sqrt(np.sum(first * first, axis=1, dtype=np.float32))
    second_norm = np.sqrt(np.sum(second * second, axis=1, dtype=np.float32))
    out = (first_norm == 0) | (second_norm == 0) | np.all(points[:, 1] == points[:, 2], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        first_unit = (first / first_norm[:, None].astype(np.float64)).astype(np.float32)
        second_unit = (second / second_norm[:, None].astype(np.float64)).astype(np.float32)
    dot = np.sum(first_unit * second_unit, axis=1, dtype=np.float32).astype(np.float64)
    out |= np.abs(dot) > 0.9999999999
    cross = np.cross(first_unit, second_unit).astype(np.float32)
    out |= ~(np.sqrt(np.sum(cross * cross, axis=1, dtype=np.float32)) > 0)
    return out
