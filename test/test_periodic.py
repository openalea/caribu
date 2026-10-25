"""The infinite canopy's pieces: periodise, the border flags and copies, the image tiling."""
import numpy as np

from openalea.caribu import periodic

PATTERN = (0., 0., 10., 10.)


def triangle(x, y, z=1., side=1.):
    return [(x, y, z), (x + side, y, z), (x, y + side, z)]


def test_periodise_moves_centres_into_the_pattern():
    points = np.array([triangle(12., 3.), triangle(-4., -13.), triangle(2., 2., z=-1.)], dtype=np.float32)
    moved = periodic.periodise(points, PATTERN)
    centres = moved.mean(axis=1)
    assert np.all((centres[:, :2] > 0) & (centres[:, :2] <= 10))
    np.testing.assert_allclose(moved[0, 0], (2., 3., 2.), rtol=1e-6)  # moved by -10 in x, lifted by 1
    np.testing.assert_allclose(moved[1, 0], (6., 7., 2.), rtol=1e-6)
    assert abs(moved[:, :, 2].min()) < 1e-6  # the lowest point at the ground (to rounding)


def test_border_flags():
    points = np.array([triangle(5., 5.), triangle(9.5, 5.), triangle(5., 9.5), triangle(9.5, 9.5),
                       triangle(-0.2, 5.)], dtype=np.float32)
    accepted, flags = periodic.border_flags(points, PATTERN)
    assert flags.tolist() == [0, 1, 2, 3, 1]
    assert accepted.tolist() == [True, True, True, True, True]
    outside = np.array([triangle(10.5, 5.)], dtype=np.float32)
    assert not periodic.border_flags(outside, PATTERN)[0][0]


def test_border_copies_keep_the_legacy_switch():
    copies = periodic.border_copies(np.array([0, 1, 2, 3, 0, 3]))
    assert copies.tolist() == [[True, False, False], [True, True, False], [True, False, True], [True, True, True],
                               [True, False, False], [True, False, False]]


def test_tiling_steps_round_as_c():
    roof = np.array([[0, 0], [5, 0], [5, 3], [0, 3]])
    assert periodic.tiling_steps(roof).tolist() == [0, 3, 5, 0, -5, 0, 0, -3]
    assert periodic._c_round(2.5) == 3 and periodic._c_round(-2.5) == -3


def test_tile_shades_from_neighbours():
    """An image whose left column is a triangle in front: tiled by one column, every pixel but those of the original
    column is shaded by a translate in front of it."""
    size = 4
    depth = np.full((size, size), periodic.F32(10.), dtype=np.float32)
    owner = np.zeros((size, size), dtype=np.int64)
    depth[0, :] = 1.
    periodic.tile(owner, depth, np.array([[1, 0], [2, 0], [3, 0]]), 0.)
    assert (owner[0] == 0).all() and (owner[1:] == -1).all()
