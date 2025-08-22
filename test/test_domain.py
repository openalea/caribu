from tools import assert_almost_equal

from alinea.caribu.caribu import green_leaf_PAR, raycasting


def test_triangle_inside():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]
    domain = (-2, -2, 2, 2)

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 3)


def test_triangle_outside():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]
    lights = [(100, (0, 0, -1))]

    # just on the right
    domain = (-2, -2, 0, 2)
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 3)

    # just on the left
    domain = (2, -2, 4, 2)
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 3)

    # further on the left
    domain = (12, -2, 14, 2)
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 3)

    # both directions
    domain = (12, 12, 14, 14)
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 3)


# def test_with_splitting():
#     pts1 = [(-1, 1, 0), (0, 0, 0), (1, 1, 0)]
#     triangles = [pts1]
#     mats = [green_leaf_PAR]
#     domain = (-2, -2, 0, 2)
#
#     # vertical light
#     lights = [(100, (0, 0, -1))]
#     res = raycasting(triangles, mats, lights, domain)
#     print res
#
#     assert_almost_equal(res['area'][0], 1, 3)
#     assert_almost_equal(res['Ei_sup'][0], 100, 0)
#     assert_almost_equal(res['Ei_inf'][0], 0, 3)
