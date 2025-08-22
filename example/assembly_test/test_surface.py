from alinea.caribu.caribu_star import *
from alinea.adel.astk_interface import AdelWheat
from alinea.astk.plant_interface import *
from alinea.astk.plantgl_utils import *


wheat = AdelWheat()
g,_ = new_canopy(wheat, age=550)
geom = g.property('geometry')
areas, normals = get_area_and_normal(geom)
energie, emission, direction, elevation, azimuth = turtle.turtle(sectors=1, energy=1) 
sources = zip(energie,direction)
out = run_caribu(sources, geom, output_by_triangle=False)

rain_fraction, rain_exposition = exposed_surface(geom)
light_fraction, light_exposition = exposed_surface(geom, '16')

#one mature length
leaf_id=32
elt_id = 34
#leaf_id=8
#elt_id = 10
length = g[leaf_id]['length']
vlength = g[leaf_id]['visible_length']
w=g[leaf_id]['shape_max_width']
area_adel = g[elt_id]['area']
area_pgl = sum(areas[elt_id])
area_caribu = out['Area'][elt_id]
area_exposition = rain_exposition[elt_id] / rain_fraction[elt_id] /0.01**2
print(area_adel, area_pgl, area_caribu, area_exposition)


#one newly emerged leaf
leaf_id=396
elt_id=398
length = g[leaf_id]['length']
vlength = g[leaf_id]['visible_length']
w=g[leaf_id]['shape_max_width']
area_adel = g[elt_id]['area']
area_pgl = sum(areas[elt_id])
area_caribu = out['Area'][elt_id]
area_exposition = None
if rain_fraction[elt_id] > 0:
    area_exposition = rain_exposition[elt_id] / rain_fraction[elt_id] /0.01**2
print(area_adel, area_pgl, area_caribu, area_exposition)

#input blade elements
l=g[396]['length']
lvis=g[396]['visible_length']
lrolled=g[396]['rolled_length']
lsen=g[396]['senesced_length']
Lshape=g[396]['shape_mature_length']
Lwshape = g[396]['shape_max_width']
xysr_shape = g[396]['shape_xysr']