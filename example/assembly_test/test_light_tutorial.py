#!/usr/bin/env
"""
Test the generic light tutorial with caribu.
"""

from openalea.adel.data_samples import adel_two_metamers
from openalea.adel.astk_interface import AdelWheat
from openalea.caribu.caribu_star import caribu_star
from mtg.plantframe.color import colormap

def plot3d(g):
    """
    TODO: move to vplants.newmtg? 
    
    Returns a plantgl scene from an mtg.
    """
    import openalea.plantgl.all as pgl
    
    Material = pgl.Material
    Color3 = pgl.Color3
    Shape = pgl.Shape
    Scene = pgl.Scene
    
    colors = g.property('color')    
    geometries = g.property('geometry')

    scene = Scene()

    def geom2shape(vid, mesh, scene):
        shape = None
        if isinstance(mesh, list):
            for m in mesh:
                geom2shape(vid, m, scene)
            return
        if mesh is None:
            return
        if isinstance(mesh, Shape):
            shape = mesh
            mesh = mesh.geometry

        if colors:
            shape = Shape(mesh, Material(Color3(* colors.get(vid, [0,0,0]) )))

        shape.id = vid
        scene.add(shape)

    for vid, mesh in geometries.iteritems():
        geom2shape(vid, mesh, scene)
    pgl.Viewer.display(scene)
    return scene


# 1. Get a sample scene from adel and create a mtg
adel = AdelWheat()
g = adel.setup_canopy(600)

# 2. Run light with default PARs (quickest call)
geom = g.property('geometry')
light_star, light_exposed_area = caribu_star(geom)

# 3. Update mtg with light_star
g.properties()['light_star'] = light_star

# 4. Add a color property to the mtg and view the result on a plot 3D
g = colormap(g, 'light_star', cmap='jet', lognorm=True)
plot3d(g)

# 5. Configure the light sources (zenith, side, diffuse)

# 6. Interface with meteo data

# 7. Update mtg

# 8. Configure the organ properties: 
# outil mtg en entree -> modele 'easy' -> mtg avec property 'optical property' (format libre a homogeneiser)

# 8. Configure the soil

# 9. Explore model option : infinity, grid, outil 'star'..... (advanced feature)
