from openalea.caribu.file_adaptor import read_light, read_pattern, read_opt, read_can, build_materials
from openalea.caribu.data_samples import data_path


def test_light():
    sky = data_path('zenith.light')
    lights = read_light(sky)
    assert len(lights) == 1
    assert lights[0][0] == 1
    assert lights[0][1] == (0, 0, -1)

    sky = data_path('Turtle16soc.light')
    lights = read_light(sky)
    assert len(lights) == 16



def test_pattern():
    path = data_path('filter.8')
    domain = read_pattern(path)
    assert domain == (0, 0, 20, 20)


def test_opt():
    path = data_path('par.opt')
    n, s, opts = read_opt(path)

    assert n == 2
    assert s == 0.15
    assert opts[1] == (0.1, 0.1, 0.05, 0.1, 0.05)


def test_can():
    can = data_path('filterT.can')
    cscene = read_can(can)
    assert len(cscene) == 1
    triangles = cscene[list(cscene.keys())[0]]
    assert len(triangles[0]) == 3
    assert len(triangles[0][0]) == 3



def test_materials():
    can = data_path('filterT.can')
    cscene = read_can(can)
    path = data_path('par.opt')
    n, s, opts = read_opt(path)
    materials = build_materials(list(cscene.keys()), opts, s)
    assert len(materials) == len(cscene)
    assert materials[list(materials.keys())[0]] == (0.1, 0.05)

