"""
The port on a scene given as legacy_run takes it, its outputs as canestrad's text: to compare them row by row.

    etri = direct_etri(vertices, labels, lights, materials)       the port's Etri.vec0 body
    outputs = direct(..., sensors=sensors)                         and its solem.dat
    outputs = radiosity(vertices, labels, lights, materials)       as legacy_run.radiosity
    outputs = layered(vertices, labels, lights, materials, pattern, layers, height)   as legacy_run.layered
"""
from .. import budgets, formats
from .. import radiosity as radiosity_module
from .. import nested, sail
from .. import direct as direct_module


def optics(materials, soil_reflectance=-1) -> tuple:
    """The port's optics (soil reflectance, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}) of
    legacy_run's *materials*, as canestrad reads them from their `.opt` text."""
    _, soil, species = formats.parse_opt(formats.opt_string(materials, soil_reflectance))
    return soil, species


def direct(vertices, labels, lights, materials, soil_reflectance=-1, size=1536, sensors=None, pattern=None) -> dict:
    """direct.run on the triangles, as legacy_run.direct's texts."""
    result = direct_module.run(vertices, labels, lights, optics(materials, soil_reflectance), size, sensors,
                               pattern=pattern)
    outputs = {"Etri.vec0": budgets.etri_rows(labels, result)}
    if sensors is not None:
        outputs["solem.dat"] = budgets.solem_rows(result["sensors"])
    return outputs


def radiosity(vertices, labels, lights, materials, soil_reflectance=-1, size=1536, sensors=None) -> dict:
    """radiosity.run on the triangles, as legacy_run.radiosity's texts (E0.dat and B.dat as canestrad prints
    them)."""
    result = radiosity_module.run(vertices, labels, lights, optics(materials, soil_reflectance), size, sensors)
    first_order = result["first_order"]
    if sensors is not None:
        first_order = list(first_order) + list(result["sensors"]["Ei0"])
    outputs = {"Etri.vec0": budgets.etri_rows(labels, result),
               "E0.dat": "".join("%.10f \n " % value for value in first_order),
               "B.dat": "".join("%.10f \n " % value for value in result["radiosity"])}
    if sensors is not None:
        outputs["solem.dat"] = budgets.solem_rows(result["sensors"])
    return outputs


def layered(vertices, labels, lights, materials, pattern, layers, height, soil_reflectance=-1, size=1536,
            sensors=None) -> dict:
    """nested.run with a null sphere (pure SAIL), as legacy_run.layered's texts."""
    result = nested.run(vertices, labels, lights, optics(materials, soil_reflectance), pattern, layers, height,
                        size=size, sensors=sensors)
    structure = result["structure"]
    outputs = {"leafarea": sail.leafarea_string(structure), "band.spec": sail.spec_string(structure),
               "cropchar": sail.cropchar_string(structure),
               "mlsail.env": sail.env_string(result["profile"], result["profile"]["thickness"]),
               "Etri.vec0": budgets.etri_rows(labels, result)}
    if sensors is not None:
        outputs["solem.dat"] = budgets.solem_rows(result["sensors"])
    return outputs


def direct_etri(vertices, labels, lights, materials, soil_reflectance=-1, size=1536) -> str:
    """direct's Etri.vec0 body."""
    return direct(vertices, labels, lights, materials, soil_reflectance, size)["Etri.vec0"]
