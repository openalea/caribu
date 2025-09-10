from openalea.caribu.caribu_star import *
from openalea.adel.astk_interface import AdelWheat
from openalea.astk.plant_interface import *
from openalea.adel.data_samples import adel_one_leaf
from openalea.adel.mtg_interpreter import mtg_interpreter
from openalea.caribu.lightString import lightString
from openalea.plantgl.all import * 
from math import pi
from openalea.caribu.CaribuScene import CaribuScene
from nose.tools import assert_almost_equal, assert_not_almost_equals

# broken : xysr not there
# def test_inclin():
    # g = adel_one_leaf()
    # flaten the leaf
    # blade = g.node(10)
    # x,y,s,r = blade.shape_xysr
    # y *= 0
    # blade.shape_xysr = (x,y,s,r)
    # stars = []
    # for inclin in (0,0.5,1): #relative inclination compare to flat leaf
        # blade.inclination = inclin
        # mtg_interpreter(g)
        # geom = g.property('geometry')
        # star, exposed_area = caribu_star(geom)
        # stars.append(star)
    # return g, stars

def test_star():
    wheat = AdelWheat()
    g,_ = new_canopy(wheat, age=100)
    geom = g.property('geometry')
    rain_star, rain_exposed_area = caribu_star(geom)
    light_star, light_exposed_area = caribu_star(geom, '16')
    return rain_star, rain_exposed_area, light_star, light_exposed_area
 
def test_rain_and_light():
    wheat = AdelWheat()
    g,_ = new_canopy(wheat, age=100)
    rain_and_light_star(g)
    return g
    
def test_unit_square():
    

    # A list of the coordinates of the square
    points = [(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,.5,0)]

    # A list of directions for the normals
    normals = [(0,0,1) for i in range(4)]
    # A list of indices that set the indices for the quads
    indices = [(0, 1, 2, 3)]
    # Creation of the quadset
    carre = AxisRotated((1,0,0), pi/2., QuadSet(points,indices,normals,indices))
    scene = Scene([Shape(carre)])
    sources = '\n'.join([lightString(emission_inv(10,1),80,90),lightString(emission_inv(10,1),80,-90)])
    return scene, sources, run_caribu(sources, scene, opticals='leaf')

def test_exposed_area():
    '''Test the validity of exposed_area according to the number of slices for cylinder reconstruction.
    '''
    radius = 0.5
    height = 3.0
    projected_area = radius * 2 * height
    
    def compute_exposed_area(slices):
        cyl = Cylinder(radius, height, False, slices)
        cyl_rotated = AxisRotated((0,1,0), pi/2., cyl)
        scene = {1: cyl_rotated}
        sources = diffuse_source(1)
        c_scene = CaribuScene()
        idmap = c_scene.add_Shapes(scene)
        c_scene.addSources(sources)
        output = c_scene.runCaribu(infinity=False)
        out = c_scene.output_by_id(output, idmap)
        star = out['Ei'].values()[0]
        area = out['Area'].values()[0]
        exposed_area = star * area
        return exposed_area
    
    exposed_area = compute_exposed_area(3) #3 slices (too few)
    assert_not_almost_equals(exposed_area, projected_area, delta=0.01) # to low
    
    exposed_area = compute_exposed_area(8) #8 slices (enough)
    assert_almost_equal(exposed_area, projected_area, delta=0.01)
    
   