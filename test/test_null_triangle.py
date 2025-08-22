from math import isnan, sqrt
from tools import assert_almost_equal

from alinea.caribu.caribu import green_leaf_PAR, raycasting, radiosity, mixed_radiosity


def test_raycasting_null_triangle_superposed_points():
    pts1 = [(0, 0, 0), (1, 0, 0), (1, 0, 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]
    lights = [(100, (0, 0, -1))]

    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 0, 3)
    assert isnan(res['Ei'][0])
    assert isnan(res['Eabs'][0])
    assert isnan(res['Ei_sup'][0])
    assert isnan(res['Ei_inf'][0])


def test_raycasting_null_triangle_aligned_points():
    pts1 = [(0, 0, 0), (1, 0, 0), (0.5, 0, 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]
    lights = [(100, (0, 0, -1))]

    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 0, 3)
    assert isnan(res['Ei'][0])
    assert isnan(res['Eabs'][0])
    assert isnan(res['Ei_sup'][0])
    assert isnan(res['Ei_inf'][0])


def test_raycasting_null_triangle_too_small():
    pts1 = [(0, 0, 0), (1, 0, 0), (0, 1e-7, 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]
    lights = [(100, (0, 0, -1))]

    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 0, 3)
    assert isnan(res['Ei'][0])
    assert isnan(res['Eabs'][0])
    assert isnan(res['Ei_sup'][0])
    assert isnan(res['Ei_inf'][0])


def test_raycasting_two_triangles_one_null():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    pts2 = [(10, 10, 0), (10 - sqrt(2), 10, 0), (10, 10, 0)]
    triangles = [pts1, pts2]
    mats = [green_leaf_PAR] * 2

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    assert_almost_equal(res['area'][1], 0, 3)
    assert isnan(res['Ei'][1])


def test_raycasting_infinite_null_triangle_inside_domain():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    pts2 = [(0, 1, 0), (1, 0, 0), (1, 0, 0)]
    triangles = [pts1, pts2]
    mats = [green_leaf_PAR] * 2
    lights = [(100, (0, 0, -1))]

    domain = (-2, -2, 2, 2)
    res = raycasting(triangles, mats, lights, domain)
    print(res)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    assert_almost_equal(res['area'][1], 0, 3)
    assert isnan(res['Ei'][1])


def test_raycasting_infinite_null_triangle_outside_domain():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    pts2 = [(2, 1, 0), (3, 0, 0), (3, 0, 0)]
    triangles = [pts1, pts2]
    mats = [green_leaf_PAR] * 2
    lights = [(100, (0, 0, -1))]
    domain = (-2, -2, 2, 2)

    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    assert_almost_equal(res['area'][1], 0, 3)
    assert isnan(res['Ei'][1])


def test_radiosity_two_triangles_one_null():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    pts2 = [(2, 1, 0), (3, 0, 0), (3, 0, 0)]
    triangles = [pts1, pts2]
    mats = [green_leaf_PAR] * 2
    lights = [(100, (0, 0, -1))]

    res = radiosity(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    assert_almost_equal(res['area'][1], 0, 3)
    assert isnan(res['Ei'][1])


def test_mixed_radiosity_two_triangles_one_null():
    pts1 = [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
    pts2 = [(2, 1, 0.5), (3, 0, 0.5), (3, 0, 0.5)]
    triangles = [pts1, pts2]
    mats = [green_leaf_PAR] * 2
    lights = [(100, (0, 0, -1))]
    domain = (-2, -2, 2, 2)
    soil_reflectance = 0.2
    diameter = 1
    layers, height = 2, 1

    res = mixed_radiosity(triangles, mats, lights, domain, soil_reflectance, diameter, layers, height)

    assert_almost_equal(res['area'][0], 0.5, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    assert_almost_equal(res['area'][1], 0, 3)
    assert isnan(res['Ei'][1])