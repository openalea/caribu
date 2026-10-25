"""lights against the legacy's light.py and sky_tools: equal values on a grid of inputs."""
import math

import numpy as np
import pytest

from openalea.caribu import lights
from openalea.caribu.legacy import light as legacy_light
from openalea.caribu.legacy.sky_tools import GenSky, GetLight, GetLightsSun, Gensun, Sky as legacy_sky
from openalea.caribu.legacy.sky_tools import Sun as legacy_sun
from openalea.caribu.legacy.sky_tools import mergeNskies, spitters_horaire


def parsed(text):
    """The legacy's light strings read back."""
    out = []
    for line in text.split("\n"):
        if line:
            values = [float(value) for value in line.split(" ")]
            out.append((values[0], tuple(values[1:4])))
    return out


@pytest.mark.parametrize("sectors, sky_type", [("46", "soc"), ("46", "uoc"), ("16", "soc"), ("16", "uoc"), ("1", "soc")])
def test_turtle(sectors, sky_type):
    port, legacy = lights.turtle(sectors, sky_type, 2.5), legacy_light.turtle(sectors, sky_type, 2.5)
    for a, b in zip(port, legacy):
        np.testing.assert_array_equal(np.array(a, dtype=float), np.array(b, dtype=float))
    assert lights.diffuse_source(16) == legacy_light.diffuse_source(16)


def test_light_sources():
    elevation, azimuth, irradiance = [10., 45., 80.], [0., 135., 290.], [1., 2., 3.]
    for orientation in (0, 30):
        port = lights.light_sources(elevation, azimuth, irradiance, orientation)
        legacy = legacy_light.light_sources(elevation, azimuth, irradiance, orientation)
        assert [(float(e), tuple(map(float, v))) for e, v in port] == [(float(e), tuple(map(float, v))) for e, v in legacy]
    assert lights.light_source(1., 30., 60.) == list(legacy_light.light_source(1., 30., 60.)[0])


@pytest.mark.parametrize("azimuths, zeniths, sky_type", [(4, 5, "soc"), (4, 5, "uoc"), (24, 9, "soc")])
def test_sector_sky(azimuths, zeniths, sky_type):
    """GenSky and GetLight as Wheat-BRIDGES uses them."""
    port = lights.Sky(azimuths, zeniths).diffuse(1., sky_type)
    legacy = GenSky.GenSky()(1., sky_type, azimuths, zeniths)
    assert port.lights() == parsed(GetLight.GetLight(legacy))
    assert port.total() == legacy.get_Rs()
    assert port.normalised().lights() == parsed(GetLight.GetLight(legacy.Noralised_Sky()))


@pytest.mark.parametrize("day, hour, latitude", [(1, 12, 48.85), (172, 7.5, 43.6), (300, 15, -30.), (80, 3, 48.85)])
def test_sun(day, hour, latitude):
    legacy = Gensun.Gensun()(1.7, day, hour, latitude)
    assert lights.sun_light(1.7, day, hour, latitude) == parsed(GetLightsSun.GetLightsSun(legacy))
    port = lights.Sun(1.7).place(day, hour, 3.14 / 180 * latitude)
    assert port.position() == legacy._get_pos_astro()
    for azimuths, zeniths in ((4, 5), (12, 6)):
        legacy_with = legacy_sky.Sky(azimuths, zeniths)
        legacy_with.set_Rd(1., "soc")
        legacy_with.set_Rsun(legacy)
        legacy_with.set_Rsun2(legacy)
        port_with = lights.Sky(azimuths, zeniths).diffuse(1., "soc").place_sun(port).add_sun(port)
        assert port_with.lights() == parsed(GetLight.GetLight(legacy_with))


def test_merged_skies():
    skies = [GenSky.GenSky()(energy, "soc", 4, 5) for energy in (1., 2., 3.)]
    port = lights.merge_skies([lights.Sky(4, 5).diffuse(energy, "soc") for energy in (1., 2., 3.)])
    assert port.lights() == parsed(GetLight.GetLight(mergeNskies.mergeNskies()(skies)))


@pytest.mark.parametrize("global_radiation", [0., 50., 150., 300., 600., 900.])
def test_spitters(global_radiation):
    for day, hour, latitude in ((172, 12, 48.85), (10, 9, 43.6), (250, 16, 52.)):
        assert lights.diffuse_fraction(global_radiation, day, hour, latitude) == spitters_horaire.RdRsH(
            global_radiation, day, hour, latitude)
        assert lights.extraterrestrial(day, hour, latitude) == spitters_horaire.extra(global_radiation, day, hour,
                                                                                      latitude)
        assert lights.declination(day) == spitters_horaire.DecliSun(day)
        decline = spitters_horaire.DecliSun(day)
        assert lights.day_length(latitude, decline) == spitters_horaire.DayLength(latitude, decline)
        assert lights.angle_hours(1.3) == spitters_horaire.dH(1.3)


def test_hourly_global_radiation():
    """The legacy's RgH cannot run (its `^`): the port's irradiance (W m-2) integrates to about the daily radiation
    over the day."""
    with pytest.raises(TypeError):
        spitters_horaire.RgH(2e7, 12, 172, 48.85)
    hours = np.linspace(0, 24, 24 * 60 + 1)
    values = np.array([lights.hourly_global_radiation(2e7, hour, 172, 48.85) for hour in hours])
    daily = np.sum(np.clip(values, 0, None)[:-1]) * (hours[1] - hours[0]) * 3600
    assert daily == pytest.approx(2e7, rel=0.02)
