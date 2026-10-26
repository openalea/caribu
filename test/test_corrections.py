"""The corrections of the legacy's behaviours (C10): each restores a physical property the legacy misses, checked on
small scenes; without them the port stays the legacy's (the comparison tests)."""
import numpy as np
import pytest

from openalea.caribu import corrections, direct, periodic
from openalea.caribu.labels import encode_labels

LEAF = encode_labels(1, leaf=1)
OPTICS = (0.15, {1: (0.1, 0.1, 0.05, 0.1, 0.05)})
ABSORPTANCE = 1 - 0.1 - 0.05
ZENITH = [(1., (0., 0., -1.))]
PATTERN = (0., 0., 1., 1.)


def lit(vertices, lights=ZENITH, chosen=(), pattern=PATTERN, sensors=None, size=512):
    return direct.run(np.asarray(vertices, dtype=float), LEAF, lights, OPTICS, size=size, pattern=pattern,
                      sensors=sensors, corrections=corrections.resolve(chosen))


def square(x, y, side, z=0.5):
    """A horizontal square of two triangles, its lower corner at (x, y)."""
    a, b, c, d = (x, y, z), (x + side, y, z), (x + side, y + side, z), (x, y + side, z)
    return [[a, b, c], [a, c, d]]


def mean_irradiance(result):
    """The area-weighted Ei of a result's triangles (a square's two, their diagonal's pixels shared)."""
    return float(np.sum(result["Ei"] * result["area"]) / np.sum(result["area"]))


def test_resolve():
    assert corrections.resolve(None) == frozenset() and corrections.resolve(()) == frozenset()
    assert corrections.resolve("recommended") == corrections.RECOMMENDED
    assert corrections.resolve(["pixel_area"]) == {"pixel_area"}
    with pytest.raises(ValueError):
        corrections.resolve(["pixel_areas"])
    with pytest.raises(ValueError):
        corrections.resolve("recommended", precision="legacy")


@pytest.mark.parametrize("direction", [(0., 0., -1.), (0.5, 0.3, -0.8), (-0.6, 0.2, -0.5), (0.2, -0.9, -0.3)])
@pytest.mark.parametrize("x, y", [(0.8, 0.4), (0.4, 0.8), (0.8, 0.8), (-0.1, 0.4), (-0.1, -0.1), (0.3, 0.3)])
def test_border_copies(x, y, direction):
    """A lone leaf receives the horizontal irradiance, across the pattern's borders too: its parts beyond the pattern
    drawn where the pattern repeats them, each point of the periodic canopy counted once (in the pattern's cell; the
    tiling's translates only shading). The legacy loses the parts beyond xmax or ymax, or off its screen."""
    leaf = square(x, y, 0.3)
    lights = [(1., direction)]
    corrected = mean_irradiance(lit(leaf, lights, ["border_copies", "tiling", "pixel_area"], size=1536))
    assert corrected == pytest.approx(1., abs=3e-3)                # the raster's quantisation of its sides
    if (x + 0.3 > 1 or y + 0.3 > 1) and direction[2] == -1.:
        assert mean_irradiance(lit(leaf, lights, size=1536)) < 0.7


def test_pixel_area():
    """A leaf covering the pattern under a vertical beam receives the light on the pattern: the legacy's (T − 1)² / T²
    of it (the pixels' pitch extent / (T − 1), their area extent² / T²)."""
    leaf = square(0., 0., 1.)
    for size in (64, 512):
        legacy = lit(leaf, size=size)
        corrected = lit(leaf, size=size, chosen=["pixel_area"])
        for result, expected in ((legacy, ((size - 1) / size) ** 2), (corrected, 1.)):
            assert mean_irradiance(result) == pytest.approx(expected, rel=1e-6)


def test_upward_beams():
    """A beam going up (a sun below the horizon) or along the ground lights nothing (the legacy: inf, nan, −0)."""
    leaf = square(0.2, 0.2, 0.3)
    for direction in ((0., 0.5, 0.5), (1., 0., 0.)):
        result = lit(leaf, lights=[(1., direction)], chosen=["upward_beams"])
        np.testing.assert_array_equal(result["Ei"], 0.)
    both = lit(leaf, lights=[(1., (0., 0., -1.)), (1., (0., 1., 1.))], chosen=["upward_beams"])["Ei"]
    np.testing.assert_allclose(both, lit(leaf)["Ei"])


def test_sensor_copies():
    """A sensor across the borders measures the whole light (the legacy: its original only, its counts added after
    each copy)."""
    sensor = np.array(square(0.85, 0.85, 0.3, z=0.9)[:1], dtype=float)
    leaf = square(0.1, 0.1, 0.2)
    legacy = lit(leaf, sensors=sensor)["sensors"]["Ei"]
    corrected = lit(leaf, sensors=sensor, chosen=["border_copies", "sensor_copies", "pixel_area"])["sensors"]["Ei"]
    assert legacy[0] < 0.6
    np.testing.assert_allclose(corrected, 1., atol=0.01)


def test_periodise():
    """The periodised scene keeps the moved coordinates (float32, no 6-digit text) and its bottom at 0."""
    points = np.array([[[1.234567891, 0.5, -0.0123456], [1.4, 0.6, 0.3], [1.3, 0.7, 0.2]]], dtype=np.float32)
    legacy = periodic.periodise(points, PATTERN)
    corrected = periodic.periodise(points, PATTERN, {"periodise"})
    assert corrected[..., 2].min() == 0. and legacy[..., 2].min() != 0
    expected = points.astype(np.float64) + [-1., 0., 0.0123456]
    np.testing.assert_allclose(corrected, expected.astype(np.float32), atol=1e-7)
    assert np.abs(legacy - expected).max() > np.abs(corrected - expected).max()


def test_triangle_area():
    """A thin triangle's area, 0.5 |AB × AC| (the legacy's arc cosine of the edges' float32 cosine loses it)."""
    thin = np.array([[[0., 0., 0.], [1., 0., 0.], [0.5, 1e-4, 0.]]], dtype=np.float32)
    exact = 0.5 * 1e-4
    assert direct.triangle_surfaces(thin, {"triangle_area"})[0] == pytest.approx(exact, rel=1e-6)
    assert abs(direct.triangle_surfaces(thin)[0] / exact - 1) > 1e-3


def test_border_copies_conserve_the_light_entering_the_pattern():
    """Random tilted leaves over an opaque floor covering the pattern: every ray entering the pattern stops once, the
    intercepted light the pattern's area times the horizontal irradiance, whatever the beam (the legacy's total too,
    the floor catching what the border leaves lose: test_border_copies)."""
    random = np.random.default_rng(1)
    centres = np.c_[random.uniform(0, 1, (40, 2)), random.uniform(0.2, 0.9, 40)]
    leaves = []
    for centre in centres:
        edges = random.normal(0, 0.12, (2, 3))
        leaves.append([centre, centre + edges[0], centre + edges[1]])
    floor = [[(0, 0, 0.05), (1, 0, 0.05), (1, 1, 0.05)], [(0, 0, 0.05), (1, 1, 0.05), (0, 1, 0.05)]]
    vertices = np.array(leaves + floor, dtype=float)
    labels = np.r_[np.full(len(leaves), LEAF), np.full(2, encode_labels(1, leaf=0))]
    for direction in ((0., 0., -1.), (0.5, 0.3, -0.8), (-0.6, 0.2, -0.5)):
        result = direct.run(vertices, labels, [(1., direction)], OPTICS, size=1536, pattern=PATTERN,
                            corrections=corrections.resolve(("border_copies", "tiling", "pixel_area")))
        assert float(np.nansum(result["Ei"] * result["area"])) == pytest.approx(1., abs=3e-3)


def random_canopy(count=300, side=0.4, seed=3):
    """Small leaves scattered in a pattern of *side* (m), from 5 to 60 cm high."""
    random = np.random.default_rng(seed)
    centres = np.c_[random.uniform(0, side, (count, 2)), random.uniform(0.05, 0.6, count)]
    return np.array([[c, c + random.normal(0, 0.03, 3), c + random.normal(0, 0.03, 3)] for c in centres])


def test_voxel_floor_and_tiling_make_the_results_independent_of_the_unit():
    """The same canopy in metres and in centimetres lit alike: the legacy finds most near neighbours in cm only
    (its receiver's voxel int(trunc(G − O) / side)), and its tiling's steps, rounded to pixels then repeated,
    place the neighbouring patterns differently."""
    from openalea.caribu import lights, nested
    vertices = random_canopy()
    sky = lights.turtle_lights("16", "soc", 1.)
    results = {}
    for name, chosen in (("legacy", ()), ("corrected", "recommended")):
        for scale in (1., 100.):
            results[name, scale] = nested.run(
                vertices * scale, LEAF, sky, OPTICS, (0, 0, 0.4 * scale, 0.4 * scale), 5, 0.65 * scale, 0.1 * scale,
                precision="full", corrections=corrections.resolve(chosen), size=512)
    pairs = {key: len(result["counts"][1]) for key, result in results.items()}
    assert pairs["legacy", 1.] < pairs["legacy", 100.] / 5
    assert abs(pairs["corrected", 1.] - pairs["corrected", 100.]) <= 2
    difference = {name: np.abs(np.asarray(results[name, 1.]["Eabs"], float)
                               - np.asarray(results[name, 100.]["Eabs"], float)) for name in ("legacy", "corrected")}
    assert difference["corrected"].mean() < 2e-4 < difference["legacy"].mean()


def test_face_counts():
    """An emitter across the receiver's plane, seen by both its discs: the legacy nets the two counts (the
    receiver's upper face then gets ρ of the difference, its lower face nothing), each disc's count kept apart."""
    from openalea.caribu import geometry, nearfield
    receiver = [(-0.05, -0.05, 0.5), (0.05, -0.05, 0.5), (0., 0.05, 0.5)]
    emitter = [(0.1, -0.05, 0.42), (0.1, 0.05, 0.42), (0.1, 0., 0.6)]         # vertical, from below to above
    points = np.array([receiver, emitter], dtype=np.float32)
    translucent = np.array([True, True])
    first_face = np.array([0, 2])
    rows = {}
    for name, bits in (("legacy", 0), ("corrected", nearfield.FACE_COUNTS)):
        indptr, faces, values = nearfield.counts(points, np.array([LEAF, LEAF]), translucent, first_face, 4,
                                                 geometry.legacy_normals(points), geometry.legacy_centres(points),
                                                 1., options=bits)
        rows[name] = list(zip(faces[indptr[0]:indptr[1]].tolist(), values[indptr[0]:indptr[1]].tolist()))
    (face, net), = rows["legacy"]
    above, below = rows["corrected"]
    assert above[0] == below[0] == face and above[1] > 0 > below[1] and above[1] + below[1] == net


def test_projected_face():
    """An emitter seen in the sphere both as itself and as its periodic image, from its two sides: each pixel takes
    the face of the copy drawn there (the legacy: the face of the last copy activated, for both)."""
    from openalea.caribu import geometry, nearfield
    pattern = (0., 0., 0.5, 0.5)
    receiver = [(0.03, 0.2, 0.5), (0.07, 0.2, 0.5), (0.05, 0.3, 0.5)]
    emitter = [(0.3, 0.2, 0.45), (0.3, 0.3, 0.45), (0.3, 0.25, 0.55)]        # vertical, facing x
    points = np.array([receiver, emitter], dtype=np.float32)
    normals, centres = geometry.legacy_normals(points), geometry.legacy_centres(points)
    translucent = np.array([True, True])
    first_face = np.array([0, 2])
    cells = nearfield.grid(points, None, 0.3, pattern)
    seen = {}
    for name, bits in (("legacy", 0), ("corrected", nearfield.PROJECTED_FACE)):
        indptr, faces, values = nearfield.counts(points, np.array([LEAF, LEAF]), translucent, first_face, 4, normals,
                                                 centres, 0.3, grid=cells, options=bits)
        seen[name] = set(faces[indptr[0]:indptr[1]].tolist())
    assert len(seen["legacy"]) == 1 and seen["corrected"] == {2, 3}


def test_profile_interpolation_and_bounds():
    """sail_pur between two levels: the nearer level weighs more; above the top the top's fluxes, below the soil
    the soil's (the legacy: the weights inverted, the next column's values beyond the top)."""
    from openalea.caribu import sail
    profile = {"height": np.array([0., 1., 2.]), "downward": np.array([1., 2., 4.]),
               "upward": np.array([0.5, 0.25, 0.125]), "thickness": 1.}
    flat_up = np.array([[0., 0., 1.]], dtype=np.float32)

    def received(height, chosen=("profile_interpolation", "profile_bounds")):
        centre = np.array([[0., 0., height]], dtype=np.float32)
        return sail.far_field(centre, flat_up, np.array([False]), np.array([0]), 1, np.array([1.]),
                              np.array([0.]), profile, frozenset(chosen))[0]
    assert received(1.25) == pytest.approx(0.75 * 2 + 0.25 * 4)               # E− at 1.25: a horizontal upper face
    assert received(1.25, ()) == pytest.approx(0.25 * 2 + 0.75 * 4)
    assert received(2.5) == pytest.approx(4.) and received(-0.5) == pytest.approx(1.)
    assert received(2.5, ("profile_interpolation",)) != pytest.approx(4.)
