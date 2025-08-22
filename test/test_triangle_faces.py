from math import sqrt
from tools import assert_almost_equal

from alinea.caribu.caribu import radiosity, raycasting


def test_unresolved_faces():
    points = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    lights = [(100, (0, 0, -1))]
    triangles = [points]

    # reflectance == transmittance
    materials = [(0.05, 0.05)]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 90, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei_sup'][0], -1, 0)
    assert_almost_equal(res['Ei_inf'][0], -1, 0)

    # reflectance_product == transmittance_product
    materials = [(0.05, 0.01, 0.01, 0.05)]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 94, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei_sup'][0], -1, 0)
    assert_almost_equal(res['Ei_inf'][0], -1, 0)


def test_raycasting_opaque_flip():
    points = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    lights = [(100, (0, 0, -1))]
    materials = [(0.1,)]

    # pointing up to source
    triangles = [points]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 90, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], -1, 0)

    # pointing back from source (bug: the triangle is reversed back by Caribu)
    # triangles = [reversed(points)]
    # res = raycasting(triangles, materials, lights)
    # assert_almost_equal(res['area'][0], 1, 3)
    # assert_almost_equal(res['Eabs'][0], 90, 0)
    # assert_almost_equal(res['Ei'][0], 100, 0)    
    # assert_almost_equal(res['Ei_sup'][0], 100, 0)
    # assert_almost_equal(res['Ei_inf'][0], -1, 0)    


def test_raycasting_translucent_flip():
    points = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    lights = [(100, (0, 0, -1))]
    materials = [(0.06, 0.04)]

    # pointing facesup toward source
    triangles = [points]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 90, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei_sup'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 0, 0)

    # pointing faceinf toward source
    triangles = [reversed(points)]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Eabs'][0], 90, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)
    assert_almost_equal(res['Ei_inf'][0], 100, 0)
    assert_almost_equal(res['Ei_sup'][0], 0, 0)


def test_radiosity_translucent_flip():
    lower_pts = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    upper_pts = [(0, 0, 1e-5), (sqrt(2), 0, 1e-5), (0, sqrt(2), 1e-5)]
    lower, upper = 0, 1
    lights = [(100, (0, 0, -1))]
    materials = [(0.1, 0.2)] * 2

    # upper / lower pointing facesup toward source
    triangles = [lower_pts, upper_pts]
    res = radiosity(triangles, materials, lights)
    assert_almost_equal(res['area'][upper], 1, 3)
    assert_almost_equal(res['Ei'][upper], 102, 0)
    assert_almost_equal(res['Eabs'][upper], 71, 0)
    assert_almost_equal(res['Ei_sup'][upper], 100, 0)
    assert_almost_equal(res['Ei_inf'][upper], 2, 0)

    assert_almost_equal(res['area'][lower], 1, 3)
    assert_almost_equal(res['Ei'][lower], 20, 0)
    assert_almost_equal(res['Eabs'][lower], 14, 0)
    assert_almost_equal(res['Ei_sup'][lower], 20, 0)
    assert_almost_equal(res['Ei_inf'][lower], 0, 0)

    # flip upper
    triangles = [lower_pts, reversed(upper_pts)]
    res = radiosity(triangles, materials, lights)
    assert_almost_equal(res['area'][upper], 1, 3)
    assert_almost_equal(res['Ei'][upper], 102, 0)
    assert_almost_equal(res['Eabs'][upper], 71, 0)
    assert_almost_equal(res['Ei_sup'][upper], 2, 0)
    assert_almost_equal(res['Ei_inf'][upper], 100, 0)

    assert_almost_equal(res['area'][lower], 1, 3)
    assert_almost_equal(res['Ei'][lower], 20, 0)
    assert_almost_equal(res['Eabs'][lower], 14, 0)
    assert_almost_equal(res['Ei_sup'][lower], 20, 0)
    assert_almost_equal(res['Ei_inf'][lower], 0, 0)

    # flip upper and lower
    triangles = [reversed(lower_pts), reversed(upper_pts)]
    res = radiosity(triangles, materials, lights)
    assert_almost_equal(res['area'][upper], 1, 3)
    assert_almost_equal(res['Ei'][upper], 102, 0)
    assert_almost_equal(res['Eabs'][upper], 71, 0)
    assert_almost_equal(res['Ei_sup'][upper], 2, 0)
    assert_almost_equal(res['Ei_inf'][upper], 100, 0)

    assert_almost_equal(res['area'][lower], 1, 3)
    assert_almost_equal(res['Ei'][lower], 20, 0)
    assert_almost_equal(res['Eabs'][lower], 14, 0)
    assert_almost_equal(res['Ei_sup'][lower], 0, 0)
    assert_almost_equal(res['Ei_inf'][lower], 20, 0)
