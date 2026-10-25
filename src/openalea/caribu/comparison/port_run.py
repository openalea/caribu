"""
The port on a scene given as legacy_run takes it, its outputs as canestrad's text: to compare them row by row.

    etri = direct_etri(vertices, labels, lights, materials)       the port's Etri.vec0 body
    outputs = direct(..., sensors=sensors)                         and its solem.dat
"""
from .. import budgets, formats
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


def direct_etri(vertices, labels, lights, materials, soil_reflectance=-1, size=1536) -> str:
    """direct's Etri.vec0 body."""
    return direct(vertices, labels, lights, materials, soil_reflectance, size)["Etri.vec0"]
