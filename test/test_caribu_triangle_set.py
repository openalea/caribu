import openalea.caribu.data as data
from openalea.caribu.caribu_triangle_set import CaribuTriangleSet



def test_instantiation_from_file():
    cts = CaribuTriangleSet.fromfile(data.get('f331s1_100plantes.can'))
    plants = cts.scales['plant']
    assert len(plants) == max(plants) == 100
    areas = cts.sum_at_scale(cts.triangle_areas(), scale='plant')
    assert all([0.95 < a / 0.36 < 1.05 for a in areas.values()])