"""
Test caribu legacy test suite, using libcaribu.algos
"""
import pytest
from numpy.testing import assert_array_equal
from importlib.resources import files
import openalea.libcaribu.io as lcio
from openalea.caribu.caribu_shell import set_scene, caribu

data_dir = files('openalea.libcaribu.data')


@pytest.fixture(scope="module")  # tmp_path is kept between tests
def caribu_test_scene(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("caribu_test_scene")
    set_scene(tmp_path,
              canopy=data_dir / "filterT.can",
              pattern=data_dir / "filter.8",
              lights=data_dir / "zenith.light",
              sensors=data_dir / "filterT.sensor",
              opts=data_dir / "par.opt")
    return tmp_path


def test_projection_non_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=True, toric=False)
    assert 'par' in out
    res, _ = out['par']
    expected = lcio.read_results(data_dir / 'projection_non_toric_scene.vec0')
    for field in res:
        assert_array_equal(res[field], expected[field])


def test_sensor_non_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=True, toric=False, sensors=True)
    assert 'par' in out
    res, mes = out['par']
    expected_res = lcio.read_results(data_dir / 'projection_non_toric_scene.vec0')
    expected_mes = lcio.read_measures(data_dir / 'sensor_non_toric_scene.dat')
    for field in res:
        assert_array_equal(res[field], expected_res[field])
    for field in mes:
        assert_array_equal(mes[field], expected_mes[field])


def test_projection_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=True, toric=True)
    assert 'par' in out
    res, _ = out['par']
    expected = lcio.read_results(data_dir / 'projection_toric_scene.vec0')
    for field in res:
        assert_array_equal(res[field], expected[field])


def test_radiosity_non_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=False, mixed_radiosity=-1)
    assert 'par' in out
    res, _ = out['par']
    expected = lcio.read_results(data_dir / 'radiosity_non_toric_scene.vec0')
    for field in res:
        assert_array_equal(res[field], expected[field])


def test_projection_sail_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=False, mixed_radiosity=0, layers=6, height=21)
    assert 'par' in out
    res, _ = out['par']
    expected = lcio.read_results(data_dir / 'projection_sail_toric_scene.vec0')
    for field in res:
        assert_array_equal(res[field], expected[field])


def test_nested_radiosity_toric_scene(caribu_test_scene):
    out = caribu(caribu_test_scene, direct_only=False, mixed_radiosity=1, layers=6, height=21)
    assert 'par' in out
    res, _ = out['par']
    expected = lcio.read_results(data_dir / 'nested_radiosity_toric_scene.vec0')
    for field in res:
        assert_array_equal(res[field], expected[field])
