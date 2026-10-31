"""The users' entry point (openalea.caribu.simulation): SimplePlant's geometry, the summary against CaribuLight's
outputs on the stand, the hours, the given inputs and parameters."""
import numpy as np
import pandas as pd
import pytest

from openalea.caribu.simulation import MAIN_PARAMETERS, SimplePlant, parameters, simulate


@pytest.fixture(scope="module")
def noon():
    return simulate(hours=2, start_hour=12)


def test_simple_plant_geometry(noon):
    ds = noon.data_structure
    elements = noon.elements()
    n_plants = len(ds.entity_ids("Plant"))
    assert n_plants == 24 and (elements[elements["kind"] == "leaf"].groupby("plant").size() == 4).all()
    triangles = ds.triangles()
    leaf = np.flatnonzero(np.asarray(ds.labels()) == "LeafElement")[0]
    corners = triangles.vertices[triangles.owner == leaf].reshape(-1, 3)
    top = corners[:, 2].max() - np.asarray(ds.get("start_z"))[leaf]
    assert top == pytest.approx(0.2 * np.sin(np.radians(30.)), rel=1e-12)          # its length up its inclination
    leaf_area = elements.loc[elements["kind"] == "leaf", "area"]          # the lit area: the leaves', whole
    assert leaf_area.sum() == pytest.approx(n_plants * 4 * 0.2 * 0.015, rel=1e-4)


def test_the_summary_is_the_components(noon):
    ds = noon.data_structure
    last = noon.summary.iloc[-1]
    area = np.asarray(ds.get("lit_area"))
    assert list(noon.summary["hour"]) == [12., 13.]
    assert last["PAR_absorbed"] == pytest.approx(float(np.sum(ds.get("PARa") * area)) / noon.ground_area, rel=1e-12)
    assert last["absorbed_fraction"] == pytest.approx(float(np.sum(ds.get("Erel") * area)) / noon.ground_area)
    intercepted = 1. - last["soil_fraction"]
    assert 0. < last["absorbed_fraction"] < intercepted < 1.
    assert noon.plants()["absorbed"].sum() == pytest.approx(noon.elements()["absorbed"].sum() / 1e6, rel=1e-12)


def test_hours_and_given_PAR():
    seen = []
    result = simulate(hours=3, start_hour=10, PARi=lambda hour: seen.append(hour) or 100. * hour)
    assert list(result.summary["hour"]) == [10., 11., 12.]
    assert list(result.summary["PARi"]) == [1000., 1100., 1200.]
    by_series = simulate(hours=2, start_hour=10, PARi=pd.Series([0., 200.], index=[10., 12.]))
    assert list(by_series.summary["PARi"]) == [0., 100.]


def test_more_leaves_leave_less_light_to_the_soil(noon):
    dense = simulate({"n_leaves": 8}, hours=1, start_hour=12)
    assert dense.summary.iloc[-1]["soil_fraction"] < noon.summary.iloc[0]["soil_fraction"]
    assert dense.summary.iloc[-1]["absorbed_fraction"] > noon.summary.iloc[0]["absorbed_fraction"]


def test_optics_and_settings():
    result = simulate({"leaf_reflectance": 0.2, "mode": "nested", "sphere_diameter": 0.2}, hours=1, start_hour=12)
    ds = result.data_structure
    leaf = np.asarray(ds.labels()) == "LeafElement"
    np.testing.assert_allclose(np.asarray(ds.get("reflectance"))[leaf], 0.2)
    assert result.scene.environment[0].light.mode == "nested" and np.isnan(result.summary["soil_fraction"]).all()
    with pytest.raises(KeyError):
        simulate({"leaf_colour": "green"}, hours=1)


def test_parameters_table():
    table = parameters({"leaf_inclination": 60.})
    assert tuple(table["name"][:len(MAIN_PARAMETERS)]) == MAIN_PARAMETERS
    assert table.loc[table["name"] == "leaf_inclination", "value"].item() == 60.
    assert set(parameters(main_only=True)["model"]) == {"SimplePlant", "CaribuLight"}
    assert "leaf_width" in SimplePlant.declarations()["name"].values


def test_draw(noon):
    import matplotlib
    matplotlib.use("Agg")
    ax = noon.draw()
    assert ax.collections
