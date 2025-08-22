from math import sqrt, radians, cos
from tools import assert_almost_equal

from alinea.caribu.caribu import green_leaf_PAR, raycasting
from alinea.caribu.light import light_sources
import numpy


def test_vertical_light():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]

    # vertical light no intensity
    lights = [(0, (0, 0, -1))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)

    # vertical light full intensity
    lights = [(100, (0, 0, -1))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    # vertical light full intensity, vertical triangle
    pts1 = [(0, 0, 0), (0, sqrt(2), 0), (0, 0, sqrt(2))]
    triangles = [pts1]
    lights = [(100, (-1, 0, 0))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)


def test_horizontal_light():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    triangles = [pts1]
    mats = [green_leaf_PAR]

    # horizontal light no intensity
    lights = [(0, (-1, 0, 0))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)

    # horizontal light full intensity
    lights = [(100, (-1, 0, 0))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)

    # horizontal light full intensity, infinite canopy
    lights = [(100, (-1, 0, 0))]
    domain = (-2, -2, 2, 2)
    res = raycasting(triangles, mats, lights, domain)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)

    # horizontal light full intensity, vertical triangle (impossible case)
    pts1 = [(0, 0, 0), (0, sqrt(2), 0), (0, 0, sqrt(2))]
    triangles = [pts1]
    lights = [(100, (-1, 0, 0))]
    res = raycasting(triangles, mats, lights)

    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 0, 0)


def test_diagonal_light():
    pts1 = [(0, 0, 0), (sqrt(2), 0, 0), (0, sqrt(2), 0)]
    triangles = [pts1]
    materials = [green_leaf_PAR]

    # full intensity on horizontal surface
    lights = [(100, (-1, 0, -1))]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 100, 0)

    # full intensity perpendicular to the source
    emission = 100
    apparent_emission = emission * cos(radians(45))
    lights = [(apparent_emission, (-1, 0, -1))]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 3)
    assert_almost_equal(res['Ei'][0], 100 * cos(radians(45)), 0)

    # full intensity perpendicular to the source, inclined triangle
    emission = 100
    apparent_emission = emission * cos(radians(45))
    lights = [(apparent_emission, (-1, 0, -1))]
    proj = cos(radians(45))
    pts1 = [(0, 0, 0), (0, sqrt(2), 0), (-sqrt(2) * proj, 0, proj * sqrt(2))]
    triangles = [pts1]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 0)
    assert_almost_equal(res['Ei'][0], 100, 0)

    # full intensity on horizontal surface, inclined triangle
    lights = [(100, (-1, 0, -1))]
    proj = cos(radians(45))
    pts1 = [(0, 0, 0), (0, sqrt(2), 0), (-sqrt(2) * proj, 0, proj * sqrt(2))]
    triangles = [pts1]
    res = raycasting(triangles, materials, lights)
    assert_almost_equal(res['area'][0], 1, 0)
    assert_almost_equal(res['Ei'][0], 141, 0)


def test_sources():
    light = light_sources(90, 0, 1)
    irr, (x,y,z) = light[0]
    numpy.testing.assert_array_equal([irr, x, y, z], [1, 0, 0, -1])
    light = light_sources(1, 0, 1)
    irr, (x,y,z) = light[0]
    numpy.testing.assert_allclose([x, y, z], [-0.99, 0, -0.01], atol=0.01)
    #
    # Cardinal positions of elevation=45 deg sources when x+ = North
    # azimuth of input is from north, positive clockwise
    north, south, east, west = light_sources([45] * 4, [0, 180, 90, -90],
                                             [1] * 4)
    # North source has a X- look_at vector, South a X+
    numpy.testing.assert_allclose(north[1], [-0.71, 0, -0.71], atol=0.01)
    numpy.testing.assert_allclose(south[1], [0.71, 0, -0.71], atol=0.01)
    # East source has Y+ look_at vector, west source a Y-
    numpy.testing.assert_allclose(east[1], [0, 0.71, -0.71], atol=0.01)
    numpy.testing.assert_allclose(west[1], [0, -0.71, -0.71], atol=0.01)
    #
    # Cardinal positions of elevation=45 deg sources when y+ = North
    # azimuth of input is from north, positive clockwise
    # orientation is from X+ to north, positive clockwise
    north, south, east, west = light_sources([45] * 4, [0, 180, 90, -90],
                                             [1] * 4, orientation=-90)
    # North source has a Y- look at vector, South a Y+
    numpy.testing.assert_allclose(north[1], [0, -0.71, -0.71], atol=0.01)
    numpy.testing.assert_allclose(south[1], [0, 0.71, -0.71], atol=0.01)
    # East source has X- look_at vector, west source a X+
    numpy.testing.assert_allclose(east[1], [-0.71, 0, -0.71], atol=0.01)
    numpy.testing.assert_allclose(west[1], [0.71, 0, -0.71], atol=0.01)
