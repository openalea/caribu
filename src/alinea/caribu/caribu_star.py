# -*- python -*-
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       WebSite : https://github.com/openalea-incubator/caribu
#
# ==============================================================================

from alinea.caribu.CaribuScene import CaribuScene
from alinea.caribu.light import turtle


def _caribu_call(scene, directions, opt=None, domain=None, convUnit=None):
    """ adaptor for backward compatibility of macros calling caribu
    """
    energie, emission, direction, elevation, azimuth = turtle(sectors=str(directions), energy=1)
    sources = list(zip(energie, direction))

    c_scene = CaribuScene(scene=scene, light=sources, opt=opt, pattern=domain)
    if convUnit is not None:
        c_scene.conv_unit = convUnit

    ifty = False
    if domain is not None:
        ifty = True

    raw, aggregated = c_scene.run(direct=True, infinite=ifty, simplify=True)

    return c_scene, raw, aggregated


def caribu_lighted_scene(scene, directions=1, domain=None, minval=None, maxval=None):
    """ generate a per-triangle colored lighted scene  (like ViewMapOnCan)
    """
    c_scene, out, _ = _caribu_call(scene, str(directions), domain=domain)
    ei = out['Ei']
    return c_scene.plot(ei, minval=minval, maxval=maxval, display=False)


def caribu_star(scene_geometry, directions=1, output_by_triangle=False, domain=None, convUnit=0.01):
    """Compute exposition ('surface_viewed-to_area ratio and exposed surface(m2)') of scene elements from a given number
        of direction

    Args:
        scene_geometry: scene geometry
        directions (int): number of directions (1, 16 or 46)
        output_by_triangle (bool): should results be aggregated ?
        domain: the domain bounding the scene
        convUnit: (float) Default '0.01'. Conversion factor to get meter from scene length unit.

    Returns:
        a dict id:star (ratio viewed area / area) and id:exposed_surface (m2)
    """

    # TODO : check that sources are okay for star (energy of emission should be the one normalised)

    c_scene, raw, aggregated = _caribu_call(scene_geometry, str(directions), domain=domain, convUnit=convUnit)
    if output_by_triangle:
        star = raw['Ei']
        areas = raw['area']
        exposed_area = {vid: [star[vid][i] * areas[vid][i] for i in range(len(areas[vid]))] for vid in areas}
    else:
        star = aggregated['Ei']
        areas = aggregated['area']
        exposed_area = {vid: star[vid] * areas[vid] for vid in areas}

    return star, exposed_area


def caribu_rain_star(g, output_by_triangle=False, domain=None, convUnit=0.01, dt=1):
    geom = g.property('geometry')
    rain_star, rain_exposed_area = caribu_star(geom, directions=1, output_by_triangle=output_by_triangle,
                                               convUnit=convUnit, domain=domain)
    if 'rain_exposed_area' not in g.properties():
        g.add_property('rain_exposed_area')
    if 'rain_star' not in g.properties():
        g.add_property('rain_star')
    g.property('rain_exposed_area').update(rain_exposed_area)
    g.property('rain_star').update(rain_star)
    return g


def caribu_light_star(g, light_sectors=16, output_by_triangle=False, domain=None, convUnit=0.01, trigger=1):
    geom = g.property('geometry')
    light_star, light_exposed_area = caribu_star(geom, directions=light_sectors, output_by_triangle=output_by_triangle,
                                                 convUnit=convUnit, domain=domain)
    if 'light_exposed_area' not in g.properties():
        g.add_property('light_exposed_area')
    if 'light_star' not in g.properties():
        g.add_property('light_star')
    g.property('light_exposed_area').update(light_exposed_area)
    g.property('light_star').update(light_star)
    return g


def rain_and_light_star(g, light_sectors=16, output_by_triangle=False, domain=None, convUnit=0.01, trigger=1):
    geom = g.property('geometry')
    rain_star, rain_exposed_area = caribu_star(geom, directions=light_sectors, output_by_triangle=output_by_triangle,
                                               convUnit=convUnit, domain=domain)

    if 'rain_exposed_area' not in g.properties():
        g.add_property('rain_exposed_area')
    if 'rain_star' not in g.properties():
        g.add_property('rain_star')
    if 'light_exposed_area' not in g.properties():
        g.add_property('light_exposed_area')
    if 'light_star' not in g.properties():
        g.add_property('light_star')
    g.property('rain_exposed_area').update(rain_exposed_area)
    g.property('light_exposed_area').update(light_exposed_area)
    g.property('rain_star').update(rain_star)
    g.property('light_star').update(light_star)
    return g


def run_caribu(sources, scene, opticals='stem', optical_properties=None, output_by_triangle=False, domain=None,
               zsoil=None):
    """ 
    Calls Caribu for differents energy sources

    :Parameters:
    ------------
    - `sources` (int)
    - `scene` : any scene format accepted by CaribuScene. This will be cast to a list of plantGL shapes
    - opticals : a list of optical property labels ('leaf', 'soil', 'stem', 'ear' or 'awn') for all shapes in scene. If a shorter opticale list is provided, it will be recycled to match eength of shape list
    - `optical_properties`: a filename (*.opt).
    - `output_by_triangle` (bool)
        Default 'False'. Return is done by id of geometry. If 'True', return is done by triangle. 

    :Returns:
    ---------
    - 'out_moy' (dict)
        A dict of intercepted variable (energy) per id
    - 'out_tri' (dict) only if output_by_triangle = True, return a tuple (out_moy, out_tri)
        A dict of intercepted variable (energy) per triangle
    """

    raise DeprecationWarning('This function is deprecated, use CaribuScene / caribu_star instead')

    from alinea.caribu.label import simple_canlabel
    from alinea.caribu.file_adaptor import read_opt, build_materials

    n, soil_reflectance, po = read_opt(optical_properties)
    if not isinstance(opticals, list):
        opticals = [opticals]
    if len(opticals) < len(scene):
        opticals = opticals * (len(scene) / len(opticals)) + [opticals[i] for i in range(len(scene) % len(opticals))]
    labels = [simple_canlabel(opt) for opt in opticals]
    materials = build_materials(labels, po, soil_reflectance)
    band = CaribuScene.default_band

    c_scene = CaribuScene(scene=scene, light=sources, pattern=domain, opt={band: materials},
                          soil_reflectance={band: soil_reflectance})

    ifty = False
    if domain is not None:
        ifty = True
        # if zsoil is not None:
        #    idmap_soil = c_scene.addSoil(zsoil=zsoil)
        #    idmap.update(idmap_soil)

    raw, aggregated = c_scene.run(direct=True, infinite=ifty, simplify=True)

    if output_by_triangle:
        out = raw
    else:
        out = aggregated
    return out
