from openalea.plantgl import all as pgl
from alinea.caribu.CaribuScene import CaribuScene
import alinea.caribu.sky_tools.turtle as turtle
from math import radians, sin , cos
from six.moves import zip



def vector_direction(elevation,azimuth):
    theta = radians(90 - elevation)
    phi = radians(azimuth)
    return sin(theta) * cos(phi),sin(theta) * sin(phi),  -cos(theta)

vecteur_direction = vector_direction

def emission_inv(elevation, energy):
    """ return energy of emission for a source of a given direction and of a given energy received on a horizontal surface """
    theta = radians(90 - elevation)
    received_energy = energy * abs(cos(theta))
    return received_energy


def geom2shape(vid, mesh):
    """ Create a shape """
    shape = pgl.Shape(mesh)
    shape.id = vid
    return shape


def run_caribu(sources, scene_geometry, output_by_triangle = False):
    """ 
    Calls Caribu for different energy sources

    :Parameters:
    ------------
    - `sources` (int)
    - `scene_geometry`
    - `output_by_triangle` (bool) 
        Default 'False'. Return is done by id of geometry. If 'True', return is done by triangle. 

    :Returns:
    ---------
    - 'out_moy' (dict)
        A dict of intercepted variable (energy) per id
    - 'out_tri' (dict) only if output_by_triangle = True, return a tuple (out_moy, out_tri)
        A dict of intercepted variable (energy) per triangle
    """
    c_scene = CaribuScene()
    shapes=[geom2shape(k,v) for k,v in scene_geometry.items()]
    idmap = c_scene.add_Shapes(shapes)    
    c_scene.addSources(sources)
    output = c_scene.runCaribu(infinity=False)
    out_moy = c_scene.output_by_id(output, idmap)
    if output_by_triangle:
        out_tri = c_scene.output_by_id(output, idmap, aggregate = False)
        indices = c_scene.scene_ids
        return out_moy, out_tri, indices
    else:
        return out_moy


def turtle_interception(sectors, scene_geometry, energy, output_by_triangle = False, convUnit = 0.01):
    """ 
    Calls Caribu for different energy sources

    :Parameters:
    ------------
    - `sectors` (int)
    - `scene_geometry`
    - `energy` (float)
        e.g. Meteorological mean variables at the global scale, per square meter. Could be:
            - 'PAR' : Quantum PAR (ppfd) in micromol.m-2.sec-1
            - 'Pluie' : Precipitation (mm)
    - 'output_by_triangle' (bool)
        Default 'False'. Choose if return is made by id of geometry or by triangle
    - 'convUnit' (float)
        Default '0.01'. Conversion factor to get meter from scene length unit.

    :Returns:
    ---------
    - 'out_moy' (dict)
        Meteorological variable at the leaf scale
    - 'out_tri' (dict) [optional]
        Meteorological variable at the leaf scale
    """
    energy_scaled = float(energy) * convUnit**2
    energie, emission, direction, elevation, azimuth = turtle.turtle(sectors=sectors, energy=energy_scaled) 
    sources = list(zip(energie,direction))
    return run_caribu(sources, scene_geometry, output_by_triangle=output_by_triangle)

