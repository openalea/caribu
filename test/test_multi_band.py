from math import sqrt
from tools import assert_almost_equal

from alinea.caribu.caribu import x_radiosity, x_mixed_radiosity, x_raycasting

def test_raycasting_two_triangles_no_occlusion():
    lower_pts = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    upper_pts = [(0, 0, 0), (-sqrt(2), 0, 0), (0, -sqrt(2), 0)]
    triangles = [lower_pts, upper_pts]
    lower, upper = 0, 1

    # vertical light, opaque material
    lights = [(100, (0, 0, -1))]
    materials = dict(band1=[(0.1, )] * 2,
                     band2=[(0.2, )] * 2)
    x_res = x_raycasting(triangles, materials, lights)

    for band in ("band1", "band2"):
        res = x_res[band]
        assert_almost_equal(res['area'][lower], 1, 3)
        assert_almost_equal(res['Ei'][lower], 100, 0)

        assert_almost_equal(res['area'][upper], 1, 3)
        assert_almost_equal(res['Ei'][upper], 100, 0)

        if band == "band1":
            assert_almost_equal(res['Eabs'][upper], 90, 0)
        else:
            assert_almost_equal(res['Eabs'][upper], 80, 0)



def test_radiosity_two_triangles_full_occlusion():
    lower_pts = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    upper_pts = [(0, 0, 1e-5), (sqrt(2), 0, 1e-5), (0, sqrt(2), 1e-5)]
    triangles = [lower_pts, upper_pts]
    lower, upper = 0, 1

    # vertical light, opaque material
    lights = [(100, (0, 0, -1))]
    materials = dict(band1=[(0.1, )] * 2,
                     band2=[(0.2, )] * 2)
    x_res = x_radiosity(triangles, materials, lights)

    for band in ("band1", "band2"):
        res = x_res[band]
        assert_almost_equal(res['area'][lower], 1, 3)
        assert_almost_equal(res['Ei_sup'][lower], 0, 0)
        assert_almost_equal(res['Ei_inf'][lower], -1, 3)

        assert_almost_equal(res['area'][upper], 1, 3)
        assert_almost_equal(res['Ei_sup'][upper], 100, 0)
        assert_almost_equal(res['Ei_inf'][upper], -1, 3)
        if band == "band1":
            assert_almost_equal(res['Eabs'][upper], 90, 0)
        else:
            assert_almost_equal(res['Eabs'][upper], 80, 0)

            
def test_mixed_radiosity_three_triangles_full_occlusion():
    # mixed radiosity needs at least two triangle
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    pts2 = [(0, 0, 0.5), (sqrt(2), 0, 0.5), (0, sqrt(2), 0.5)]
    pts3 = [(0, 0, 1), (sqrt(2), 0, 1), (0, sqrt(2), 1)]
    triangles = [pts1, pts2, pts3]
    materials = dict(band1=[(0.1, )] * 3,
                     band2=[(0.2, )] * 3)
    domain = (0, 0, 2, 2)
    soil_reflectance = dict(band1=0.2, band2=0.1)

    # vertical light
    lights = [(100, (0, 0, -1))]
    diameter, layers, height = 0.6, 3, 1.2
    x_res = x_mixed_radiosity(triangles, materials, lights, domain, soil_reflectance, diameter, layers, height)

    for band in ("band1", "band2"):
        res = x_res[band]
        assert_almost_equal(res['area'][0], 1, 3)
        # assert_almost_equal(res['Ei_sup'][0], -1, 0)
        # assert_almost_equal(res['Ei_inf'][0], -1, 3)

        assert_almost_equal(res['area'][2], 1, 3)
        # assert_almost_equal(res['Ei_sup'][2], -1, 0)
        # assert_almost_equal(res['Ei_inf'][2], -1, 3)
        # TODO radiosity result