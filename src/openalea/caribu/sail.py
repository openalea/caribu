"""
The far field: the canopy as a turbid medium of horizontal layers (s2v, A.7.2 of the description), the fluxes
through it by the multilayer SAIL model (mcsail: Verhoef 1984, 1985; A.7.3), and their light on the triangles
when no near field is kept (canestrad's sail_pur, Ross 1981; A.7.5).

    structure = layers(points, labels, 10, height, pattern, [optics])   per layer (top first) the leaf angle
                                                                         frequencies, LAI, ρ and τ of each band
    profile = profiles(structure["frequencies"][::-1], ...)             per level (from the soil up) E−, E+
    radiosity = far_field(points, labels, optics, profile)              each face's first scattering of them

The layers are numbered from the top in s2v, from the soil up in mcsail (its level 0 the soil); `profiles` takes
them from the soil up. The legacy's numerics are kept (double precision, its sums in its order; the C library's
functions through `math`), its files' printed digits applied by the callers when their precision is the legacy's.
"""
import math

import numpy as np
from numba import njit

from . import formats

F32 = np.float32
LEAF_ANGLE_CLASSES = 18     # s2v's zenith classes, 5° each
SUBDIVISION_LEVELS = 6      # s2v's LevelMax: a triangle across cells split at most 5 times
DEGREES = 57.29577951       # s2v's 180 / π


# --- Structure (s2v) ---

@njit(cache=True, error_model="numpy")
def _layer(z, tops, layer_count):
    """classe: the layer of height z (0 the top; *tops* each layer's top, cumulated from the bottom), −10 below 0."""
    if z < 0:
        return -10
    for j in range(1, layer_count):
        if z > tops[j]:
            return j - 1
    return layer_count - 1


@njit(cache=True, error_model="numpy")
def _zenith_class(p, class_count):
    """normal: the zenith of the triangle's normal (0 for a horizontal triangle), its class of 90 / class_count
    degrees. The normal's cross product written out as s2v's, its float32 vertices (Patch) mixed with doubles: the
    edges and the products of two vertices' coordinates in float32, the rest in double."""
    f64 = np.float64
    edge0, edge1, edge2 = f64(F32(p[1, 0] - p[2, 0])), f64(F32(p[1, 1] - p[2, 1])), f64(F32(p[1, 2] - p[2, 2]))
    u = f64(p[0, 1]) * edge2 - f64(p[0, 2]) * edge1 + f64(F32(p[1, 1] * p[2, 2])) - f64(F32(p[2, 1] * p[1, 2]))
    v = f64(-p[0, 0]) * edge2 + f64(p[0, 2]) * edge0 - f64(F32(p[1, 0] * p[2, 2])) + f64(F32(p[2, 0] * p[1, 2]))
    w = f64(p[0, 0]) * edge1 - f64(p[0, 1]) * edge0 + f64(F32(p[1, 0] * p[2, 1])) - f64(F32(p[2, 0] * p[1, 1]))
    square = u * u + v * v
    zenith = 0. if square == 0 else math.acos(abs(w / math.sqrt(square + w * w))) * DEGREES
    return min(class_count - 1, int(zenith / (90.0 / class_count)))


@njit(cache=True, error_model="numpy")
def _structure(points, species, leaf, band_optics, period_x, period_y, tops, layer_count, class_count):
    """The areas per (species, layer, zenith class) and per band their sums × (ρ, τ) per (species, layer), the
    triangles in order, each split depth-first (repart) while its vertices fall in different cells; *points*
    float32 (s2v's Patch), its sums of coordinates in float32, the rest in double."""
    species_count, band_count = band_optics.shape[1], band_optics.shape[0]
    areas = np.zeros((species_count, layer_count, class_count))
    optical = np.zeros((species_count, layer_count, band_count, 2))
    stack = np.zeros((4 * SUBDIVISION_LEVELS + 1, 3, 3), dtype=np.float32)
    levels = np.zeros(4 * SUBDIVISION_LEVELS + 1, dtype=np.int64)
    cells = np.zeros((3, 3), dtype=np.int64)
    piece = np.zeros((3, 3), dtype=np.float32)
    middle = np.zeros((3, 3), dtype=np.float32)
    for t in range(points.shape[0]):
        if species[t] < 0:  # the soil
            continue
        je = species[t]
        area_factor = 0.5 if leaf[t] else 0.25  # a stem counts for half its area
        zenith = _zenith_class(points[t], class_count)
        stack[0] = points[t]
        levels[0] = 0
        top = 1
        while top > 0:
            top -= 1
            for i in range(3):
                for k in range(3):
                    piece[i, k] = stack[top, i, k]
            level = levels[top] + 1
            for i in range(3):
                cells[i, 0] = int(np.float64(piece[i, 0]) / period_x)
                cells[i, 1] = int(np.float64(piece[i, 1]) / period_y)
                cells[i, 2] = _layer(np.float64(piece[i, 2]), tops, layer_count)
            across = 0
            for i in range(3):
                across += abs(cells[0, i] - cells[1, i])
            for i in range(3):
                across += abs(cells[0, i] - cells[2, i])
            if across % 256 == 0:  # a char: in a single cell
                if np.float64(F32(F32(piece[0, 2] + piece[1, 2]) + piece[2, 2])) / 3. < 0:
                    continue
                jz = cells[0, 2]
            elif level == SUBDIVISION_LEVELS:  # the centroid's cell
                centre_z = np.float64(F32(F32(piece[0, 2] + piece[1, 2]) + piece[2, 2])) / 3.
                if centre_z < 0:
                    continue
                jz = _layer(centre_z, tops, layer_count)
            else:  # into 4, pushed so that the first is processed first
                for j in range(3):
                    for i in range(3):
                        middle[j, i] = F32(np.float64(F32(piece[j, i] + piece[(j + 1) % 3, i])) / 2.)
                for child in range(3, -1, -1):
                    for i in range(3):
                        if child == 0:
                            stack[top, 0, i], stack[top, 1, i], stack[top, 2, i] = piece[0, i], middle[0, i], middle[2, i]
                        elif child == 1:
                            stack[top, 0, i], stack[top, 1, i], stack[top, 2, i] = middle[0, i], piece[1, i], middle[1, i]
                        elif child == 2:
                            stack[top, 0, i], stack[top, 1, i], stack[top, 2, i] = middle[1, i], middle[2, i], piece[2, i]
                        else:
                            stack[top, 0, i], stack[top, 1, i], stack[top, 2, i] = middle[0, i], middle[1, i], middle[2, i]
                    levels[top] = level
                    top += 1
                continue
            a0 = np.float64(F32(piece[1, 0] - piece[0, 0]))
            a1 = np.float64(F32(piece[1, 1] - piece[0, 1]))
            a2 = np.float64(F32(piece[1, 2] - piece[0, 2]))
            b0 = np.float64(F32(piece[2, 0] - piece[0, 0]))
            b1 = np.float64(F32(piece[2, 1] - piece[0, 1]))
            b2 = np.float64(F32(piece[2, 2] - piece[0, 2]))
            c0, c1, c2 = a1 * b2 - a2 * b1, -a0 * b2 + a2 * b0, a0 * b1 - a1 * b0
            area = area_factor * math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
            areas[je, jz, zenith] += area
            for band in range(band_count):
                if leaf[t]:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 1]
                    optical[je, jz, band, 1] += area * band_optics[band, je, 2]
                else:
                    optical[je, jz, band, 0] += area * band_optics[band, je, 0]
    return areas, optical


def s2v_optics(optics) -> tuple:
    """s2v's reading of a band's optics (soil, {species: (rho_stem, rho_sup, tau_sup, rho_inf, tau_inf)}): (soil
    reflectance, (species, 3): the stem's ρ, the leaf's mean ρ and mean τ of its faces)."""
    soil, species = optics
    table = np.array([[values[0], (values[1] + values[3]) / 2., (values[2] + values[4]) / 2.]
                      for _, values in sorted(species.items())]).reshape(-1, 3)
    return (0. if soil is None else soil), table


def layers(points, labels, layer_count, height, pattern, band_optics, class_count=LEAF_ANGLE_CLASSES) -> dict:
    """s2v: the canopy (*points* (n, 3, 3), as periodise writes them, read as float32) in *layer_count* layers of equal
    thickness up to *height*, horizontally homogeneous over the *pattern* (x1, y1, x2, y2). Per layer (top first):
    "frequencies" (layers, classes), the species-merged frequency of each zenith class of the normals;
    "leaf_area_index" (layers,), stems at half their area; per band of *band_optics* (each as direct's optics)
    "reflectance", "transmittance" (bands, layers), percent, the area-weighted means of the layer's stems' ρ and its
    leaves' mean ρ, τ; "soil_reflectance" (bands,); "thickness"."""
    labels = np.asarray(labels, dtype=np.int64)
    species = (labels // 1000) // 100000000 - 1
    leaf = (labels // 1000) % 1000 > 0
    read = [s2v_optics(optics) for optics in band_optics]
    species_count = len(read[0][1])
    tables = np.array([table for _, table in read]).reshape(len(read), species_count, 3)
    x1, y1, x2, y2 = (float(value) for value in formats.pattern_string(pattern).split())  # as s2v reads `.8`
    period_x, period_y = x2 - x1, y2 - y1
    thickness = float(str(height)) / float(int(layer_count))  # as s2v reads its arguments
    tops = np.zeros(layer_count)
    for i in range(layer_count - 1, -1, -1):
        tops[i] = thickness if i == layer_count - 1 else thickness + tops[i + 1]
    areas, optical = _structure(np.asarray(points, dtype=np.float64).astype(F32), species, leaf, tables, period_x, period_y, tops,
                                layer_count, class_count)
    per_class = areas.copy()  # xladi, its sum over the classes xlad, then normalised
    total = np.zeros((species_count, layer_count))
    for jz in range(layer_count):
        for je in range(species_count):
            for ji in range(class_count):
                total[je, jz] += per_class[je, jz, ji]
            if total[je, jz] > 0:
                per_class[je, jz] /= total[je, jz]
    volume_factor = (period_x * period_y) / (period_x * period_y)  # xymaille, 1 cell
    frequencies = np.zeros((layer_count, class_count))
    leaf_area_index = np.zeros(layer_count)
    for jz in range(layer_count):
        layer_total = 0.
        for je in range(species_count):
            layer_total += total[je, jz]
        for ji in range(class_count):
            weighted = 0.
            for je in range(species_count):
                weighted += per_class[je, jz, ji] * total[je, jz]
            if -1e-9 < layer_total < 1e-9:
                layer_total = -1e-9 if layer_total < 0 else 1e-9
            frequencies[jz, ji] = weighted / layer_total
        leaf_area_index[jz] = layer_total / (volume_factor * (period_x * period_y) * thickness) * thickness
    reflectance = np.zeros((len(read), layer_count))
    transmittance = np.zeros((len(read), layer_count))
    for band in range(len(read)):
        for jz in range(layer_count):
            rho = tau = weight = 0.
            for je in range(species_count):
                if total[je, jz] > 0:
                    weight += total[je, jz]
                    rho += optical[je, jz, band, 0]
                    tau += optical[je, jz, band, 1]
            if weight > 0:
                rho *= 100. / weight
                tau *= 100. / weight
            reflectance[band, jz], transmittance[band, jz] = rho, tau
    return {"frequencies": frequencies, "leaf_area_index": leaf_area_index, "reflectance": reflectance,
            "transmittance": transmittance, "soil_reflectance": np.array([soil for soil, _ in read]),
            "thickness": thickness}


def printed_layers(structure) -> dict:
    """*structure* as mcsail reads it from s2v's files: leafarea's values with 6 decimals, the spectra's with 3,
    the thickness (cropchar) with 6."""
    six, three = np.vectorize(lambda value: float("%f" % value)), np.vectorize(lambda value: float("%.3f" % value))
    return {"frequencies": six(structure["frequencies"]), "leaf_area_index": six(structure["leaf_area_index"]),
            "reflectance": three(structure["reflectance"]), "transmittance": three(structure["transmittance"]),
            "soil_reflectance": three(structure["soil_reflectance"]),
            "thickness": float("%f" % structure["thickness"])}


def leafarea_string(structure) -> str:
    """s2v's `leafarea` text."""
    return "".join("  %d  %d  " % (0, jz + 1) + "".join("%f " % value for value in frequencies)
                   + "0 0 %f\n" % lai for jz, (frequencies, lai) in
                   enumerate(zip(structure["frequencies"], structure["leaf_area_index"])))


def spec_string(structure, band=0) -> str:
    """s2v's `<band>.spec` text."""
    rows = zip(structure["reflectance"][band], structure["transmittance"][band])
    return ("%d\n%.3f\n" % (len(structure["leaf_area_index"]), structure["soil_reflectance"][band])
            + "".join("%.3f %.3f\n" % row for row in rows))


def cropchar_string(structure, class_count=LEAF_ANGLE_CLASSES) -> str:
    """s2v's `cropchar` text."""
    return "%d\n %d %f\n" % (class_count, len(structure["leaf_area_index"]), structure["thickness"])


# --- Profiles (mcsail) ---

def beta_frequencies(mu, nu) -> list:
    """distrib: the 13 classes of Verhoef (5, 15, … 75, 81, 83, … 89°) of a Beta leaf angle law (Goel & Strebel
    1984), the Gamma functions by Stirling's approximation; percent."""
    pie = math.pi * 2
    w = mu + nu
    gamma = [math.pow(pie / x, .5) * math.pow(x, x) * math.exp(1 / (12 * x) - 1 / (360 * x * x * x) - x)
             for x in (mu, nu, w)]
    ratio = gamma[2] / (gamma[0] * gamma[1])
    frequencies, total = [], 0.
    for i in range(13):
        angle, weight = (float(i * 10 + 5), 9.) if i < 8 else (float((i - 8) * 2 + 81), 45.)
        frequency = ratio / weight * math.pow(1 - angle / 90, mu - 1) * math.pow(angle / 90, nu - 1)
        total += frequency
        frequencies.append(frequency * 100)
    return [value + (1. - total) * value for value in frequencies]


def _divide(a, b):
    """a / b as the C's IEEE division (±inf, nan for a zero divisor)."""
    if b != 0:
        return a / b
    if a == 0 or a != a:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1., b)


def _coefficients(frequencies, lai, rho, tau, source_zenith, view_zenith, azimuth, beta=None):
    """msailad for one layer: (att, sig, ks, sb, sf, ko, uf, ub, w); *frequencies* in percent, ρ and τ in
    percent, angles in degrees (the source's and the view's zenith, their azimuth difference)."""
    pi, rd = math.atan(1.) * 4., math.atan(1.) * 4. / 180.
    rtp = (rho + tau) * .005
    if rtp == 0:  # s2v writes ρ = τ = 0 for an empty layer
        rtp = rtm = .1
    else:
        rtm = (rho - tau) * .005
    tants, tanto = math.tan(rd * source_zenith), math.tan(rd * view_zenith)
    cspsi, psir = math.cos(rd * azimuth), rd * azimuth
    sbf = sks = sko = sw1 = sw2 = 0.
    classes = len(frequencies) if beta is None else 13
    for ia in range(classes):
        if beta is None:
            ttl, frequency = 90. / float(classes) * (float(ia) + .5), frequencies[ia]
        else:
            ttl, frequency = float(ia * 10 + 5 if ia < 8 else (ia - 8) * 2 + 81), beta[ia]
        cstl = math.cos(rd * ttl)
        cs2tl = cstl * cstl
        sn2tl = 1. - cs2tl
        sntl = math.sqrt(sn2tl)
        sbf = sbf + cs2tl * frequency
        bts = math.acos(_divide(-cstl, sntl * tants)) if ttl + source_zenith > 90 else pi
        bto = math.acos(_divide(-cstl, sntl * tanto)) if ttl + view_zenith > 90 else pi
        btran1, btran2 = abs(bts - bto), pi * 2 - bts - bto
        if psir <= btran1:
            bt1, bt2, bt3 = psir, btran1, btran2
        elif psir <= btran2:
            bt1, bt2, bt3 = btran1, psir, btran2
        else:
            bt1, bt2, bt3 = btran1, btran2, psir
        sks += ((bts - pi * .5) * cstl + math.sin(bts) * tants * sntl) * frequency
        sko += ((bto - pi * .5) * cstl + math.sin(bto) * tanto * sntl) * frequency
        tsin = sn2tl * .5 * tants * tanto
        t1 = cs2tl + tsin * cspsi
        sw1 += (cs2tl + tsin * cspsi) * frequency
        if bt2 != 0.:
            sw2 += (-bt2 * t1 + math.sin(bt2) * (_divide(cs2tl, math.cos(bts) * math.cos(bto))
                                                 + math.cos(bt1) * math.cos(bt3) * tsin)) * frequency
    sbfs = sbf * rtm * lai * .01
    att, sig = lai * (1 - rtp) + sbfs, lai * rtp + sbfs
    ks, ko = sks * .02 * lai / pi, sko * .02 * lai / pi
    w = lai * .01 * (rho * .01 * sw1 + 2 * rtp * sw2 / pi)
    return att, sig, ks, ks * rtp + sbfs, ks * rtp - sbfs, ko, ko * rtp + sbfs, ko * rtp - sbfs, w


def _layer_matrices(lai, coefficients):
    """mlayer for one layer: (tss, too, rdd, tdd, rsd, tsd, rdo, tdo, rso)."""
    if lai == 0:
        return 1., 1., 0., 1., 0., 0., 0., 0., 0.
    att, sig, ks, sb, sf, ko, uf, ub, w = coefficients
    som = att * att - sig * sig
    m = math.sqrt(0.0 if som < 0.0 else som)
    h1 = _divide(att + m, sig)
    h2 = _divide(1., h1)
    e1 = math.exp(m)
    e2 = 1. / e1
    dns, dno = ks * ks - m * m, ko * ko - m * m
    cs = _divide(sb * (ks - att) - sf * sig, dns)
    ds = _divide(-sf * (ks + att) - sb * sig, dns)
    co = _divide(uf * (ko - att) - ub * sig, dno)
    do = _divide(-ub * (ko + att) - uf * sig, dno)
    tss, too = math.exp(-ks), math.exp(-ko)
    if m != 0.:
        dnd = h1 * e1 - h2 * e2
        rdd, tdd = _divide(e1 - e2, dnd), _divide(h1 - h2, dnd)
    else:
        rdd = sig / (1 + sig)
        tdd = 1 - rdd
    rsd = cs * (1 - tss * tdd) - ds * rdd
    tsd = ds * (tss - tdd) - cs * tss * rdd
    rdo = co * (1 - too * tdd) - do * rdd
    tdo = do * (too - tdd) - co * too * rdd
    ho = _divide(sf * co + sb * do + w, ko + ks)
    rso = ho * (1 - tss * too) - co * tsd * too - do * rsd
    return tss, too, rdd, tdd, rsd, tsd, rdo, tdo, rso


def _adding(matrices, soil, direct, diffuse):
    """mprofil: the profile of a stack (*matrices* from the soil up, level 0 the soil) under a direct and a
    diffuse incident flux: per level (transdir, transdif, trans, refdif, refdir, absc)."""
    levels = len(matrices)  # N: the soil and the layers
    rmdd, rmsd, rmdo, rmso = [0.] * (levels + 1), [0.] * (levels + 1), [0.] * (levels + 1), [0.] * (levels + 1)
    tmss, tmsd, tmdd = [0.] * (levels + 1), [0.] * (levels + 1), [0.] * (levels + 1)
    rmdd[0] = rmsd[0] = rmdo[0] = rmso[0] = soil
    for ic in range(1, levels):  # level N's coefficients (an unfilled layer in the C) are never read
        tss, too, rdd, tdd, rsd, tsd, rdo, tdo, rso = matrices[ic]
        den = 1. - rmdd[ic - 1] * rdd
        rmdd[ic] = rdd + tdd * rmdd[ic - 1] * tdd / den
        rmsd[ic] = rsd + (tss * rmsd[ic - 1] + tsd * rmdd[ic - 1]) * tdd / den
        rmdo[ic] = rdo + tdd * (rmdd[ic - 1] * tdo + rmdo[ic - 1] * too) / den
        part = (tss * rmsd[ic - 1] + tsd * rmdd[ic - 1]) * tdo
        part = part + (tsd + tss * rmsd[ic - 1] * rdd) * rmdo[ic - 1] * too
        rmso[ic] = rso + tss * rmso[ic - 1] * too + part / den
        tmss[ic] = tss
        tmsd[ic] = (tss * rdd * rmsd[ic - 1] + tsd) / den
        tmdd[ic] = tdd / den
    tmtss, tmtsd, tmtds, tmtdd = [0.] * (levels + 1), [0.] * (levels + 1), [0.] * (levels + 1), [0.] * (levels + 1)
    tmtss[levels], tmtds[levels], tmtsd[levels], tmtdd[levels] = 1., 0., 0., 1.
    for ic in range(levels - 1, 0, -1):
        tmtss[ic] = tmss[ic] * tmtss[ic + 1]
        tmtsd[ic] = tmsd[ic] * tmtss[ic + 1] + tmdd[ic] * tmtsd[ic + 1]
        tmtds[ic] = tmss[ic] * tmtds[ic + 1]
        tmtdd[ic] = tmdd[ic] * tmtdd[ic + 1] + tmsd[ic] * tmtds[ic + 1]
    profile = np.zeros((levels + 1, 6))
    for ic in range(1, levels + 1):
        profile[ic, 0] = tmtss[ic] * direct
        profile[ic, 1] = tmtsd[ic] * direct + tmtdd[ic] * diffuse
        profile[ic, 2] = profile[ic, 0] + profile[ic, 1]
    for ic in range(levels):
        profile[ic, 3] = rmsd[ic] * profile[ic + 1, 0] + rmdd[ic] * profile[ic + 1, 1]
        profile[ic, 4] = rmso[ic] * profile[ic + 1, 0] + rmdo[ic] * profile[ic + 1, 1]
    for ic in range(1, levels):
        absorbed = profile[ic + 1, 0] * (1. - rmsd[ic])
        absorbed += profile[ic + 1, 1] * (1. - rmdd[ic])
        absorbed += - profile[ic, 0] * (1. - rmsd[ic - 1])
        absorbed += - profile[ic, 1] * (1. - rmdd[ic - 1])
        profile[ic, 5] = absorbed
    return profile


PROFILE_COLUMNS = ("direct", "downward", "total", "upward", "view", "absorbed")


def profiles(frequencies, leaf_area_index, reflectance, transmittance, soil_reflectance, thickness, lights,
             view_zenith=0., view_azimuth=0., diffuse=None, beta=None) -> dict:
    """mcsail: the fluxes through the layers (given from the soil up: *frequencies* (layers, classes) fractions,
    *leaf_area_index*, *reflectance* and *transmittance* percent), summed over the *lights* [(E, (vx, vy, vz))]
    weighted by their energy (each a direct beam of zenith acos(|vz| / |v|)). Per level 0 … layers (the soil, then
    each layer's top): "height", "direct" (uncollided), "downward" (E−, diffuse), "total", "upward" (E+, at the
    level below), "view" (the radiance towards the view direction: *view_zenith*, *view_azimuth* in degrees),
    "absorbed" (per layer). *diffuse*: the legacy's incident mode, (sun height, diffuse fraction) replacing the
    lights; *beta*: per layer (μ, ν) of Verhoef's 13 classes of a Beta law, replacing the frequencies."""
    layer_count = len(leaf_area_index)
    levels = layer_count + 1
    lai = [0.] + [float(value) for value in leaf_area_index]
    percent = [None] + [[float(value) * 100 for value in row] for row in frequencies]
    sources = ([(1., 90 - diffuse[0], diffuse[1])] if diffuse is not None else
               [(float(energy), math.acos(abs(vector[2]) / math.sqrt(vector[0] * vector[0] + vector[1] * vector[1]
                                                                     + vector[2] * vector[2])) / (math.atan(1.) * 4. / 180.),
                 0.) for energy, vector in lights])
    total = np.zeros((levels + 1, 6))
    for energy, source_zenith, diffuse_fraction in sources:
        matrices = [_layer_matrices(0., None)]
        for ic in range(1, levels):
            layer_beta = None if beta is None else beta_frequencies(*beta[ic - 1])
            coefficients = _coefficients(percent[ic], lai[ic], float(reflectance[ic - 1]),
                                         float(transmittance[ic - 1]), source_zenith, view_zenith, view_azimuth,
                                         layer_beta)
            matrices.append(_layer_matrices(lai[ic], coefficients))
        profile = _adding(matrices, float(soil_reflectance), 1 - diffuse_fraction, diffuse_fraction)
        total += energy * profile
    out = {"height": np.array([(i - 1) * float(thickness) for i in range(1, levels + 1)])}
    for index, name in enumerate(PROFILE_COLUMNS):
        values = total[:, index]
        out[name] = values[:levels] if name in ("upward", "view") else values[1:]
    out["absorbed"] = total[1:levels, 5]
    return out


def env_string(profile, thickness) -> str:
    """mcsail's `mlsail.env` text: layers, thickness; per level from the soil up its height, E−, E+."""
    return ("%d  %f\n" % (len(profile["height"]) - 1, thickness)
            + "".join("%f  %f  %f\n" % row for row in zip(profile["height"], profile["downward"], profile["upward"])))


def printed_profile(profile, thickness) -> dict:
    """*profile* as canestrad reads `mlsail.env` (6 decimals), with its thickness."""
    six = np.vectorize(lambda value: float("%f" % value))
    return {"height": six(profile["height"]), "downward": six(profile["downward"]),
            "upward": six(profile["upward"]), "thickness": float("%f" % thickness)}


# --- Pure SAIL (sail_pur) ---

def far_field(centres, normals, translucent, first_face, n_faces, reflectance, transmittance, profile):
    """sail_pur: each face's radiosity from the profile's E−, E+ at its triangle's centre height (the levels'
    weights inverted, as canestrad: (1 − f) on the level above), an inclined face's irradiance by Ross's formula
    ((1 ± cos θ) / 2 of each flux), reflected and transmitted (*centres*, *normals* float32, as canestrad).
    canestrad's table (Tabdyn) is column-major: a level beyond the top reads the next column's first levels (E−
    beyond the top is E+ at the soil, E+ beyond it the soil level's height); beyond the table, the port takes the
    top level."""
    heights, downward, upward = profile["height"], profile["downward"], profile["upward"]
    rows = len(heights)
    table = np.concatenate([downward, upward, heights])  # Tenv(i, c) = table[i + rows c]

    def level_value(level, column):
        index = level + rows * column
        return table[index] if index < len(table) else table[rows - 1 + rows * column]

    radiosity = np.zeros(n_faces)
    for t in range(len(centres)):
        nz = float(normals[t, 2])
        cosine = abs(nz)
        layer = (float(centres[t, 2]) - heights[0]) / profile["thickness"]
        level = int(layer) % 256  # an unsigned char
        layer -= level
        if layer == 0.0:
            fluxes = (level_value(level, 0), level_value(level, 1))
        else:
            fluxes = (layer * level_value(level, 0) + (1 - layer) * level_value(level + 1, 0),
                      layer * level_value(level, 1) + (1 - layer) * level_value(level + 1, 1))
        up = 0 if nz > 0 else 1
        down = (up + 1) % 2
        upper = .5 * (1 + cosine) * fluxes[up] + .5 * (1 - cosine) * fluxes[down]
        face = first_face[t]
        radiosity[face] = reflectance[face] * upper
        if translucent[t]:
            radiosity[face + 1] = transmittance[face] * upper
            lower = .5 * (1 - cosine) * fluxes[up] + .5 * (1 + cosine) * fluxes[down]
            radiosity[face] += transmittance[face + 1] * lower
            radiosity[face + 1] += reflectance[face + 1] * lower
    return radiosity
