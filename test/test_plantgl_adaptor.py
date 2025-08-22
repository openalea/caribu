run_test = True
try:
    import openalea.plantgl.all as pgl
    from openalea.mtg.mtg import MTG
except ImportError:
    run_test = False

if run_test:

    from alinea.caribu.plantgl_adaptor import scene_to_cscene, mtg_to_cscene

    def test_scene():
        s = pgl.Scene()
        cs = scene_to_cscene(s)
        assert cs == {}

        s.add(pgl.Sphere())
        cs = scene_to_cscene(s)
        assert len(cs) == 1
        css = list(cs.values())[0]
        assert len(css[0]) == 3
        assert len(css[0][0]) == 3

        s.add(pgl.Shape(pgl.Sphere()))
        cs = scene_to_cscene(s)
        assert len(cs) == 2

        css = list(cs.values())[1]
        assert len(css[0]) == 3
        assert len(css[0][0]) == 3


    def test_mtg():
        g = MTG()
        cs = mtg_to_cscene(g)
        assert cs == {}
        g.add_property('geometry')
        cs = mtg_to_cscene(g)
        assert cs == {}
        geom = g.property('geometry')
        geom[0] = pgl.Sphere()
        cs = mtg_to_cscene(g)
        assert len(cs) == 1
        assert len(list(cs.values())[0][0]) == 3
        assert len(list(cs.values())[0][0][0]) == 3
