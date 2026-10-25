"""The port's file writers give the legacy's text (caribu.py's builders), its readers read it back."""
import numpy as np

from openalea.caribu import formats
from openalea.caribu.legacy import caribu

TRIANGLES = [[(0., 0., 0.), (1., 0., 0.1234567), (0., 1.5, 2.)], [(-1., -2., 3.), (4.25, 5., 6.), (7., 8., 9.)]]
LABELS = ["100001001000", "200001000000"]


def test_scene():
    assert formats.can_string(TRIANGLES, LABELS) == caribu.triangles_string(TRIANGLES, LABELS)


def test_scene_read_back(tmp_path):
    path = tmp_path / "scene.can"
    path.write_text(formats.can_string(TRIANGLES, LABELS))
    vertices, labels = formats.read_can_arrays(path)
    np.testing.assert_allclose(vertices, TRIANGLES, atol=5e-7)
    assert labels.tolist() == [int(label) for label in LABELS]


def test_lights_pattern_sensors():
    lights = [(1., (0., 0., -1.)), (0.25, (0.3, -0.2, -0.9))]
    assert formats.light_string(lights) == caribu.light_string(lights)
    assert formats.pattern_string((2., 1., -1., -3.)) == caribu.pattern_string((2., 1., -1., -3.))
    assert formats.sensor_string(TRIANGLES) == caribu.sensor_string(TRIANGLES)


def test_optics():
    species = {1: (0.1,), 2: (0.1, 0.05), 3: (0.12, 0.07, 0.03, 0.02)}
    text = formats.opt_string(species, 0.15)
    assert text == caribu.opt_string(species, 0.15)
    n, soil, read = formats.parse_opt(text)
    assert (n, soil) == (3, 0.15)
    assert read == {1: (0.1, -1, -1, -1, -1), 2: (-1, 0.1, 0.05, 0.1, 0.05), 3: (-1, 0.12, 0.07, 0.03, 0.02)}
