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
""" Adaptor for PlantGL object and derived
"""
import openalea.plantgl.all as pgl



def pgl_to_triangles(pgl_object, tesselator=None):
    triangles = []
    if tesselator is None:
        tesselator = pgl.Tesselator()
    pgl_object.apply(tesselator)
    mesh = tesselator.triangulation
    if mesh:
        indices = mesh.indexList
        pts = list(map(tuple,mesh.pointList))
        triangles = [(pts[itri[0]],pts[itri[1]],pts[itri[2]]) for itri in indices]
    return triangles

def scene_to_cscene(scene):
    """ Build a caribu-compatible scene from a PlantGl scene

    Args:
        scene: an openalea.plantgl.all.Scene instance

    Returns:
        a {primitive_id: [triangles,]} dict.A triangle is a 3-tuple of 3-tuples points coordinates
        primitive_id is taken as the index of the shape in the scene shape list.

    """
    import itertools
    cscene = {}
    tesselator = pgl.Tesselator()
    for pid, pgl_objects in scene.todict().items():
        tri_list = list(itertools.chain(*[pgl_to_triangles(pgl_object, tesselator) for pgl_object in pgl_objects]))
        if len(tri_list) > 0:
            cscene[pid] = tri_list
    return cscene


def mtg_to_cscene(g, property_name='geometry'):
    """Build a caribu-compatible scene from a mtg encoding geometries

    Args:
        g: an openalea.mtg.mtg.MTG instance
        property_name: (str) the name of the property in g where plantGL geometries are encoded

    Returns:
        a {primitive_id: [triangles,]} dict.A triangle is a 3-tuple of 3-tuples points coordinates
        primitive_id is the vertex id.

    """
    geometry = g.property(property_name)
    tesselator = pgl.Tesselator()
    cscene = {}
    for pid in geometry:
        cscene[pid] = pgl_to_triangles(geometry[pid], tesselator)

    return cscene
