run_test = True
try:
    import openalea.plantgl.all as pgl
except ImportError:
    run_test = False

if run_test:
    from tools import assert_almost_equal

    import openalea.plantgl.all as pgl
    from alinea.caribu.CaribuScene import CaribuScene
    from alinea.caribu.data_samples import data_path


    def test_instantiation_from_files():
        can = str(data_path('filterT.can'))
        sky = str(data_path('Turtle16soc.light'))
        opts = list(map(str, [data_path('par.opt'), data_path('nir.opt')]))
        pattern = str(data_path('filter.8'))

        # defaults
        cs = CaribuScene()
        assert cs.scene is None
        assert cs.light == [cs.default_light]
        assert cs.pattern is None
        assert cs.material is None
        assert cs.soil_reflectance[
                   cs.default_band] == cs.default_soil_reflectance

        # complete set of files
        cs = CaribuScene(scene=can, light=sky, opt=opts, pattern=pattern)
        assert len(cs.scene) == 1
        assert len(cs.scene[list(cs.scene.keys())[0]]) == 192
        assert len(cs.light) == 16
        for band in ('nir', 'par'):
            assert band in cs.material
            assert len(cs.material[band]) == len(cs.scene)
        assert cs.soil_reflectance['par'] != cs.soil_reflectance['nir']
        assert len(cs.pattern) == 4

        # incomplete set of files
        cs = CaribuScene(scene=can)
        assert len(cs.material[cs.default_band]) == len(cs.scene)


    def test_instantiation_from_python():
        s = pgl.Scene()
        s.add(pgl.Sphere())
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        triangles = [points]
        pyscene = {'t1': triangles}
        sky = [(0, (0, 0, -1)), (1, (0, 0, -1))]
        pattern = (0, 0, 20, 20)
        pattern_old_style = ((0, 0), (20, 20))
        materials = {'par': {'t1': (0.3,)}}
        simplified_materials = {'par': (0.3,)}
        soil_reflectance = {'par': 0}

        cs = CaribuScene(scene=s, light=sky, pattern=pattern)
        assert len(cs.scene) == 1
        assert len(list(cs.scene.values())[0]) == 112
        assert len(cs.light) == 2
        assert len(cs.pattern) == 4
        assert list(cs.material[cs.default_band].values())[0] == cs.default_material
        assert cs.soil_reflectance[
                   cs.default_band] == cs.default_soil_reflectance

        cs = CaribuScene(scene=pyscene, opt=materials,
                         pattern=pattern_old_style)
        assert len(cs.scene) == 1
        assert len(cs.scene['t1']) == 1
        assert cs.material == materials
        assert cs.soil_reflectance['par'] == cs.default_soil_reflectance
        assert len(cs.pattern) == 4

        cs = CaribuScene(scene=pyscene, soil_reflectance=soil_reflectance)
        assert 'par' in cs.material
        assert cs.soil_reflectance == soil_reflectance

        cs = CaribuScene(scene=pyscene, opt=materials,
                         soil_reflectance=soil_reflectance)
        assert 'par' in cs.material
        assert cs.soil_reflectance == soil_reflectance

        cs = CaribuScene(scene=pyscene, opt=simplified_materials,
                         soil_reflectance=soil_reflectance)
        assert 'par' in cs.material
        assert cs.soil_reflectance == soil_reflectance


    def test_bbox():
        s = pgl.Scene()
        s.add(pgl.Sphere())
        cs = CaribuScene(scene=s)
        bbox = cs.bbox()
        assert bbox == ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        triangles = [points]
        pyscene = {'t1': triangles}
        cs = CaribuScene(scene=pyscene)
        bbox = cs.bbox()
        assert bbox == ((0, 0, 0), (1, 1, 0))


    def test_autoscreen():
        s = pgl.Scene()
        s.add(pgl.Sphere())
        cs = CaribuScene(scene=s)
        npix = cs.auto_screen(0.01)
        assert npix == 173
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        triangles = [points]
        pyscene = {'t1': triangles}
        cs = CaribuScene(scene=pyscene)
        npix = cs.auto_screen(0.01)
        assert npix == 141


    def test_aggregation():
        # simple case
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(agg['area']['lower'], 1, 0)
        assert_almost_equal(agg['Ei']['lower'], 0.5, 1)
        assert_almost_equal(agg['Ei']['upper'], 1, 0)

        # pts3 now define a null triangle
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 0, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(agg['area']['lower'], 0.5, 0)
        assert_almost_equal(agg['Ei']['lower'], 0, 1)
        assert_almost_equal(agg['Ei']['upper'], 1, 0)


    def test_soil():
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain, soil_mesh=1)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(agg['area']['lower'], 1, 0)
        assert_almost_equal(agg['Ei']['lower'], 0.5, 1)
        assert_almost_equal(agg['Ei']['upper'], 1, 0)
        assert len(cscene.soil) == 2
        assert len(cscene.soil_raw[cscene.default_band]['Ei']) == 2
        soil = cscene.soil_aggregated[cscene.default_band]
        assert_almost_equal(soil['Ei'], 0, 1)
        assert_almost_equal(soil['area'], 1, 1)

        # test with numeric ids
        pyscene = {0: [pts_1, pts_3], 1: [pts_2]}
        cscene = CaribuScene(pyscene, pattern=domain, soil_mesh=1)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(agg['area'][0], 1, 0)
        assert_almost_equal(agg['Ei'][0], 0.5, 1)
        assert_almost_equal(agg['Ei'][1], 1, 0)
        assert len(cscene.soil) == 2
        assert len(cscene.soil_raw[cscene.default_band]['Ei']) == 2
        soil = cscene.soil_aggregated[cscene.default_band]
        assert_almost_equal(soil['Ei'], 0, 1)
        assert_almost_equal(soil['area'], 1, 1)

    def test_display():
        # standard scene
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)
        cscene.plot(display=False)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        cscene.plot(agg['Ei'], display=False)
        cscene.plot(out['Ei'], display=False)

        # include null triangle
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 0, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        cscene.plot(out['Ei'], display=False)


    def test_unit():
        # scene in meter
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(sum(out['area']['lower']), 1, 0)
        assert_almost_equal(sum(out['Ei']['upper']), 1, 0)

        # same scene but now in centimeter
        pts_1 = [(0, 0, 0), (100, 0, 0), (0, 100, 0)]
        pts_2 = [(0, 0, 1e-3), (1, 0, 1e-3), (0, 1, 1e-3)]
        pts_3 = [(100, 0, 0), (100, 100, 0), (0, 100, 0)]
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 100, 100)
        cscene = CaribuScene(pyscene, pattern=domain, scene_unit='cm')
        out, agg = cscene.run(direct=True, infinite=False, simplify=True)
        assert_almost_equal(sum(out['area']['lower']), 1, 0)
        assert_almost_equal(sum(out['Ei']['upper']), 1, 0)


    def test_run_monochrome():
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        sensors = {'solem': [[(0, 0, 2), (1, 0, 2), (0, 1, 2)], [(1, 0, 2), (1, 1, 2), (0, 1, 2)]],
                   'kipp': [[(0, 0, 3), (1, 0, 3), (0, 1, 3)]]}
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain)

        # raycasting
        out, agg = cscene.run(direct=True, infinite=False, sensors=sensors)
        assert len(out) == 1
        assert len(out[cscene.default_band]['Eabs']) == 2
        assert len(out[cscene.default_band]['Eabs']['lower']) == 2
        assert len(out[cscene.default_band]['Eabs']['upper']) == 1
        assert len(out[cscene.default_band]['sensors']['Ei']) == 2

        # radiosity
        out, agg = cscene.run(direct=False, infinite=False)
        assert len(out) == 1
        assert len(out[cscene.default_band]['Eabs']) == 2
        assert len(out[cscene.default_band]['Eabs']['lower']) == 2
        assert len(out[cscene.default_band]['Eabs']['upper']) == 1

        # mixed radiosity
        out, agg = cscene.run(direct=False, infinite=True)
        assert len(out) == 1
        assert len(out[cscene.default_band]['Eabs']) == 2
        assert len(out[cscene.default_band]['Eabs']['lower']) == 2
        assert len(out[cscene.default_band]['Eabs']['upper']) == 1


    def test_run_polychrome():
        pts_1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pts_2 = [(0, 0, 1e-5), (1, 0, 1e-5), (0, 1, 1e-5)]
        pts_3 = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        sensors = {'solem': [[(0, 0, 2), (1, 0, 2), (0, 1, 2)], [(1, 0, 2), (1, 1, 2), (0, 1, 2)]],
                   'kipp': [[(0, 0, 3), (1, 0, 3), (0, 1, 3)]]}
        pyscene = {'lower': [pts_1, pts_3], 'upper': [pts_2]}
        opt = {'par': {'lower': (0.1,), 'upper': (0.1,)},
               'nir': {'lower': (0.5,), 'upper': (0.5,)}}
        domain = (0, 0, 1, 1)
        cscene = CaribuScene(pyscene, pattern=domain, opt=opt)

        # raycasting
        out, agg = cscene.run(direct=True, infinite=False,sensors=sensors)
        assert 'par' in list(out.keys())
        assert 'nir' in list(out.keys())
        assert len(out['par']['Eabs']) == 2
        assert len(out['par']['Eabs']['lower']) == 2
        assert len(out['nir']['Eabs']) == 2
        assert out['par']['Eabs']['upper'][0] != out['nir']['Eabs']['upper'][0]

        # radiosity
        out, agg = cscene.run(direct=False, infinite=False)
        assert 'par' in list(out.keys())
        assert 'nir' in list(out.keys())
        assert len(out['par']['Eabs']) == 2
        assert len(out['par']['Eabs']['lower']) == 2
        assert len(out['nir']['Eabs']) == 2
        assert out['par']['Eabs']['upper'][0] != out['nir']['Eabs']['upper'][0]

        # mixed radiosity
        out, agg = cscene.run(direct=False, infinite=True)
        assert 'par' in list(out.keys())
        assert 'nir' in list(out.keys())
        assert len(out['par']['Eabs']) == 2
        assert len(out['par']['Eabs']['lower']) == 2
        assert len(out['nir']['Eabs']) == 2
        assert out['par']['Eabs']['upper'][0] != out['nir']['Eabs']['upper'][0]
