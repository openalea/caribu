"""
From radiosities to the triangles' light budgets (canestrad's genres; doc/legacy.rst). A face of reflectance
ρ and radiosity B receives Ei = B / ρ (opaque) and absorbs Eabs = Ei − B. A leaf's two faces exchange light through
it: B_sup = ρ_sup Ei_sup + τ_inf Ei_inf, B_inf = ρ_inf Ei_inf + τ_sup Ei_sup, solved for (Ei_sup, Ei_inf); it absorbs
Eabs = Ei_sup + Ei_inf − (B_sup + B_inf). Energies are surfacic densities (per unit area of the triangle).

Rows follow the scene's triangles in their input order; a rejected triangle has area 0 and NaN energies; a black body
(ρ = 0, or a leaf whose system is singular) gets −1. The legacy's numerics are kept: its outputs are float32 (`reel`).
"""
import numpy as np

from .labels import is_translucent

F32 = np.float32


def etri_rows(labels, result):
    """canestrad's Etri.vec0 text body (no header) of a run's *result* (direct.run's): one row per input triangle,
    rejected ones `area 0` and NaN."""
    accepted = result["accepted"]
    translucent = is_translucent(labels)
    rows = []
    for number, label in enumerate(labels):
        if not accepted[number]:
            # canestrad writes `Nan` for a rejected triangle before an accepted one, `NaN` for trailing ones
            nan = "Nan" if np.any(accepted[number:]) else "NaN"
            rows.append("%d %.0f 0 %s NaN NaN NaN\n" % (number, label, nan))
            continue
        layout = "%d %.0f %f  %f  %f %f %f\n" if translucent[number] else "%d %.0f %f  %f %f %f %f\n"
        rows.append(layout % (number, label, *(result[name][number] for name in
                                              ("area", "Eabs", "Ei", "Ei_sup", "Ei_inf"))))
    return "".join(rows)


def solem_rows(sensors) -> str:
    """canestrad's solem.dat text of the sensors (numbered from 1): number, first-order and total irradiance
    (*sensors* "Ei0" and "Ei"; the same in direct light, without "Ei0"), area."""
    first_order = sensors.get("Ei0", sensors["Ei"])
    return "".join("%.0f\t %.10f\t %.10f \t%.6f\n" % (number + 1, first, irradiance, area)
                   for number, (first, irradiance, area) in enumerate(zip(first_order, sensors["Ei"],
                                                                          sensors["area"])))
