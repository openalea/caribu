""" Unit Tests for caribu_shell module """

from openalea.caribu.caribu_shell import Caribu, CaribuOptionError, vcaribu
from openalea.caribu.data_samples import data_path


# Original test of caribu.csh script by M. Chelle
def test_case_1_projection_non_toric_scene(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None, debug=debug)

    sim.infinity = False
    sim.direct = True  # direct light only
    sim.nb_layers = None
    sim.can_height = None
    sim.sphere_diameter = -1
    sim.pattern = None
    sim.run()
    assert isinstance(sim.nrj, dict)
    assert 'par' in sim.nrj
    assert 'nir' in sim.nrj
    for w in ('par', 'nir'):
        assert 'data' in sim.nrj[w]
        assert 'Eabs' in sim.nrj[w]['data']



def test_case_2_projection_toric_scene(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None, debug=debug)

    sim.infinity = True
    sim.direct = True  # direct light only
    sim.nb_layers = None
    sim.can_height = None
    sim.sphere_diameter = -1
    sim.pattern = data_path('filter.8')

    sim.run()
    assert isinstance(sim.nrj, dict)
    assert 'par' in sim.nrj
    assert 'nir' in sim.nrj
    for w in ('par', 'nir'):
        assert 'data' in sim.nrj[w]
        assert 'Eabs' in sim.nrj[w]['data']


def test_case_3_radiosity_non_toric_scene(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None, debug=debug)

    sim.infinity = False
    sim.direct = False  # NOT direct light only
    sim.nb_layers = None
    sim.can_height = None
    sim.sphere_diameter = -1
    sim.pattern = None

    sim.run()
    assert isinstance(sim.nrj, dict)
    assert 'par' in sim.nrj
    assert 'nir' in sim.nrj
    for w in ('par', 'nir'):
        assert 'data' in sim.nrj[w]
        assert 'Eabs' in sim.nrj[w]['data']


def test_case_4_projection_sail_toric_scene(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None, debug=debug)

    sim.infinity = True
    sim.direct = False  # NOT direct light only
    sim.sphere_diameter = 0
    sim.nb_layers = 6
    sim.can_height = 21
    sim.pattern = data_path('filter.8')

    sim.run()
    assert isinstance(sim.nrj, dict)
    assert 'par' in sim.nrj
    assert 'nir' in sim.nrj
    for w in ('par', 'nir'):
        assert 'data' in sim.nrj[w]
        assert 'Eabs' in sim.nrj[w]['data']


def test_case_5_nested_radiosity_toric_scene(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]
    opts = data_path('par.opt')
    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None, debug=debug)

    sim.infinity = True
    sim.direct = False  # NOT direct light only
    sim.sphere_diameter = 1  # 1, 2, 5, 10, 20 make canestra crash !!!
    sim.nb_layers = 6
    sim.can_height = 21
    sim.pattern = data_path('filter.8')

    sim.run()
    assert isinstance(sim.nrj, dict)
    assert 'par' in sim.nrj
    assert 'data' in sim.nrj['par']
    assert 'Eabs' in sim.nrj['par']['data']


def test_caribu_inconsistent_case():
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, resdir=None, resfile=None)

    sim.direct = False  # NOT direct light only
    sim.nb_layers = None
    sim.can_height = None
    sim.sphere_diameter = -1
    sim.pattern = None

    # radiosity toric scene
    sim.infinity = True
    sim.pattern = data_path('filter.8')
    try:
        sim.run()
        assert False, "This test uses inconsistent options, it should raise an CaribuOptionError"
    except CaribuOptionError:
        assert True

    # nested radiosity non toric scene
    sim.infinity = False
    sim.pattern = None
    sim.sphere_diameter = 1
    sim.nb_layers = 6
    sim.can_height = 21
    try:
        sim.run()
        assert False, "This test uses inconsistent options, it should raise an CaribuOptionError"
    except CaribuOptionError:
        assert True


def test_vcaribu():
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = data_path('par.opt')
    pattern = None
    options = {'infinity': False}
    nrj, status = vcaribu(can, sky, opts, pattern, options)
    assert isinstance(nrj, dict)
    assert 'par' in nrj
    assert 'data' in nrj['par']
    assert 'Eabs' in nrj['par']['data']


def test_sensor(debug=False):
    can = data_path('filterT.can')
    sky = data_path('zenith.light')
    opts = [data_path('par.opt'), data_path('nir.opt')]
    sensor = data_path('sensor.can')

    sim = Caribu(canfile=can, skyfile=sky,
                 optfiles=opts, sensorfile=sensor, resdir=None, resfile=None, debug=debug)

    sim.infinity = False
    sim.direct = True  # direct light only
    sim.nb_layers = None
    sim.can_height = None
    sim.sphere_diameter = -1
    sim.pattern = None
    sim.run()
    assert isinstance(sim.measures, dict)
    assert 'par' in sim.measures

if __name__ == '__main__':
    tests = [(fname,func) for fname, func in globals().items() if 'test_' in fname]
    for fname,func in tests:
            print(fname)
            func(True)