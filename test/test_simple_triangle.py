from math import sqrt
from tools import assert_almost_equal

from alinea.caribu.caribu import (mixed_radiosity, radiosity,
                                  raycasting)


def test_raycasting_single_triangle():
    points = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    triangles = [points]
    materials = [(0.06, 0.04)]

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 90, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)


def test_raycasting_two_triangles_no_occlusion():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    pts2 = [(10, 10, 0), (10 - sqrt(2), 10, 0), (10, 10 - sqrt(2), 0)]
    triangles = [pts1, pts2]
    materials = [(0.06, 0.04)] * 2

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, materials, lights)
    for i in (0, 1):
        assert_almost_equal(res['area'][i], 1, 3)
        assert_almost_equal(res['Eabs'][i], 90, 0)
        assert_almost_equal(res['Ei'][i], 100, 0)


def test_two_triangle_confounded():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    triangles = [pts1, pts1]
    materials = [(0.06, 0.04)] * 2
    lights = [(100, (0, 0, -1))]

    # raycasting
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei'][1], 0, 0)

    #radiosity
    res = radiosity(triangles, materials, lights)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei'][1], 0, 0)


def test_raycasting_two_triangles_full_occlusion():
    lower_pts = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    upper_pts = [(0, 0, 1e-5), (sqrt(2), 0, 1e-5), (0, sqrt(2), 1e-5)]
    triangles = [lower_pts, upper_pts]
    lower, upper = 0, 1
    materials = [(0.06, 0.04)] * 2

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, materials, lights)

    assert_almost_equal(res['area'][lower], 1, 3)
    assert_almost_equal(res['Eabs'][lower], 0, 0)
    assert_almost_equal(res['Ei'][lower], 0, 0)

    assert_almost_equal(res['area'][upper], 1, 3)
    assert_almost_equal(res['Eabs'][upper], 90, 0)
    assert_almost_equal(res['Ei'][upper], 100, 0)


def test_radiosity_two_triangles_full_occlusion():
    lower_pts = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    upper_pts = [(0, 0, 1e-5), (sqrt(2), 0, 1e-5), (0, sqrt(2), 1e-5)]
    triangles = [lower_pts, upper_pts]
    lower, upper = 0, 1
    materials = [(0.06, 0.04)] * 2

    # vertical light
    lights = [(100, (0, 0, -1))]
    res = radiosity(triangles, materials, lights)

    assert_almost_equal(res['area'][lower], 1, 3)
    assert_almost_equal(res['Eabs'][lower], 3.6, 1)
    assert_almost_equal(res['Ei'][lower], 4, 0)

    assert_almost_equal(res['area'][upper], 1, 3)
    assert_almost_equal(res['Eabs'][upper], 90, 0)
    assert_almost_equal(res['Ei'][upper], 100, 0)


def test_mixed_radiosity_four_triangles_full_occlusion():
    # a two time two stack of sticked triangles
    dz = 1e-5
    z1 = 0.25
    z2 = 0.75
    lower_pts_layer1 = [(0, 0, z1), (sqrt(2), 0, z1), (0, sqrt(2), z1)]
    upper_pts_layer1 = [(0, 0, z1 + dz), (sqrt(2), 0, z1 + dz), (0, sqrt(2), z1 + dz)]
    lower_pts_layer2 = [(0, 0, z2), (sqrt(2), 0, z2), (0, sqrt(2), z2)]
    upper_pts_layer2 = [(0, 0, z2 + dz), (sqrt(2), 0, z2 + dz), (0, sqrt(2), z2 + dz)]
    triangles = [lower_pts_layer1, upper_pts_layer1, lower_pts_layer2, upper_pts_layer2]
    lower1, upper1, lower2, upper2 = list(range(4))

    lights = [(100, (0, 0, -1))]
    materials = [(0.06, 0.04)] * 4
    layers, height = 2, 1
    soil_reflectance = 0.2

    # pure radiosity
    res = radiosity(triangles, materials, lights)
    assert_almost_equal(res['Ei'][upper2], 100, 0)
    assert_almost_equal(res['Ei'][lower2], 4, 0)
    assert_almost_equal(res['Ei'][upper1], 0.1, 1)
    assert_almost_equal(res['Ei'][lower1], 0, 0)

    # direct + pure layer, dense canopy
    domain = (0, 0, sqrt(2), sqrt(2))
    diameter = 0
    res = mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)
    assert_almost_equal(res['Ei'][upper2], 103, 0)
    assert_almost_equal(res['Ei'][lower2], 4, 0)
    assert_almost_equal(res['Ei'][upper1], 4, 0)
    assert_almost_equal(res['Ei'][lower1], 4, 0)

    # direct + mixed radiosity, dense canopy (20% soil reflectance)
    domain = (0, 0, sqrt(2), sqrt(2))
    diameter = 0.1
    res = mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)
    assert_almost_equal(res['Ei'][upper2], 101, 0)
    assert_almost_equal(res['Ei'][lower2], 7, 0)
    assert_almost_equal(res['Ei'][upper1], 2, 0)
    assert_almost_equal(res['Ei'][lower1], 3, 0)

    # direct + pure layer, sparse canopy (20% soil reflectance)
    domain = (-10, -10, 10, 10)
    diameter = 0
    res = mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)
    assert_almost_equal(res['Ei'][upper2], 120, 0)
    assert_almost_equal(res['Ei'][lower2], 20, 0)
    assert_almost_equal(res['Ei'][upper1], 20, 0)
    assert_almost_equal(res['Ei'][lower1], 20, 0)

    # direct + mixed radiosity, sparse canopy
    domain = (-10, -10, 10, 10)
    diameter = 0.1
    res = mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)
    assert_almost_equal(res['Ei'][upper2], 101, 0)
    assert_almost_equal(res['Ei'][lower2], 24, 0)
    assert_almost_equal(res['Ei'][upper1], 1, 0)
    assert_almost_equal(res['Ei'][lower1], 20, 0)
