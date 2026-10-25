"""
The port on a scene given as legacy_run takes it, its outputs as canestrad's text: to compare them row by row.

    etri = direct_etri(vertices, labels, lights, materials)       the port's Etri.vec0 body
"""
from .. import budgets, direct, formats


def optics(materials, soil_reflectance=-1) -> tuple:
    """The port's optics (soil reflectance, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}) of
    legacy_run's *materials*, as canestrad reads them from their `.opt` text."""
    _, soil, species = formats.parse_opt(formats.opt_string(materials, soil_reflectance))
    return soil, species


def direct_etri(vertices, labels, lights, materials, soil_reflectance=-1, size=1536) -> str:
    """direct.run on the triangles: its Etri.vec0 body."""
    result = direct.run(vertices, labels, lights, optics(materials, soil_reflectance), size)
    return budgets.etri_rows(labels, result)
