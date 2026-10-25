"""
From radiosities to the triangles' light budgets (canestrad's genres, B.9 of the description). A face of reflectance
ρ and radiosity B receives Ei = B / ρ (opaque) and absorbs Eabs = Ei − B. A leaf's two faces exchange light through
it: B_sup = ρ_sup Ei_sup + τ_inf Ei_inf, B_inf = ρ_inf Ei_inf + τ_sup Ei_sup, solved for (Ei_sup, Ei_inf); it absorbs
Eabs = Ei_sup + Ei_inf − (B_sup + B_inf). Energies are surfacic densities (per unit area of the triangle).

Rows follow the scene's triangles in their input order; a rejected triangle has area 0 and NaN energies; a black body
(ρ = 0, or a leaf whose system is singular) gets −1. The legacy's numerics are kept: its outputs are float32 (`reel`).
"""
import numpy as np
from numba import njit

from .labels import is_translucent

F32 = np.float32


@njit(cache=True, nogil=True)
def triangle_budgets(radiosity, translucent, reflectance, transmittance):
    """Per accepted triangle (n,) arrays (Eabs, Ei, Ei_sup, Ei_inf) as float32, from the faces' radiosities
    (n_faces,): Ei the total incident (sup + inf for a leaf), Ei_inf −1 for an opaque triangle."""
    count = len(translucent)
    absorbed, incident = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    incident_sup, incident_inf = np.zeros(count, dtype=F32), np.zeros(count, dtype=F32)
    face = 0
    total = F32(0.)  # canestrad keeps the previous triangle's total for a singular leaf with r0 != t1
    for t in range(count):
        if not translucent[t]:
            rho, b = reflectance[face], radiosity[face]
            if rho == 0:
                absorbed[t] = incident[t] = incident_sup[t] = F32(-1)
            else:
                incident_sup[t] = total = F32(b / rho)
                incident[t] = total
                absorbed[t] = F32(np.float64(incident_sup[t]) - b)
            incident_inf[t] = F32(-1)
            face += 1
            continue
        r0, t0 = F32(reflectance[face]), F32(transmittance[face])
        r1, t1 = F32(reflectance[face + 1]), F32(transmittance[face + 1])
        b_sup, b_inf = radiosity[face], radiosity[face + 1]
        determinant = F32(F32(r0 * r1) - F32(t0 * t1))
        if determinant == 0:
            absorbed[t] = incident_sup[t] = incident_inf[t] = F32(-1)
            if r0 == t1:
                absorbed[t] = F32(b_sup * np.float64(F32(F32(1) / r0 - F32(1))) - b_inf)
                total = F32(absorbed[t] / F32(F32(F32(1) - r0) - t0))
        else:
            sup = F32(F32(b_sup * np.float64(r1) - b_inf * np.float64(t1)) / determinant)
            inf = F32(F32(np.float64(r0) * b_inf - np.float64(t0) * b_sup) / determinant)
            incident_sup[t], incident_inf[t] = sup, inf
            absorbed[t] = F32(np.float64(F32(sup + inf)) - (b_sup + b_inf))
            total = F32(sup + inf)
        incident[t] = total
        face += 2
    return absorbed, incident, incident_sup, incident_inf


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
    """canestrad's solem.dat text of the sensors (numbered from 1): number, first-order and total irradiance (the
    same in direct light), area."""
    return "".join("%.0f\t %.10f\t %.10f \t%.6f\n" % (number + 1, irradiance, irradiance, area)
                   for number, (irradiance, area) in enumerate(zip(sensors["Ei"], sensors["area"])))
