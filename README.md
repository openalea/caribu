# openalea.caribu

[![Documentation status](https://readthedocs.org/projects/caribu/badge)](https://caribu.readthedocs.io)
[![CI Status](https://github.com/openalea/caribu/actions/workflows/openalea_ci.yml/badge.svg)](https://github.com/openalea/caribu/actions/workflows/openalea_ci.yml?query=branch%3Amaster)
[![Anaconda version](https://anaconda.org/openalea3/openalea.caribu/badges/version.svg)](https://anaconda.org/openalea3/openalea.caribu)
[![Latest release date](https://anaconda.org/openalea3/openalea.caribu/badges/latest_release_date.svg)](https://anaconda.org/openalea3/openalea.caribu)
[![Platforms](https://anaconda.org/openalea3/openalea.caribu/badges/platforms.svg)](https://anaconda.org/openalea3/openalea.caribu)
[![License](https://anaconda.org/openalea3/openalea.caribu/badges/license.svg)](https://anaconda.org/openalea3/openalea.caribu)
[![Downloads](https://anaconda.org/openalea3/openalea.caribu/badges/downloads.svg)](https://anaconda.org/openalea3/openalea.caribu)

---

## What is Caribu?

Caribu is a modelling suite for lighting 3D virtual scenes, especially designed for the illumination of virtual plant canopies such as virtual crop fields.  
It uses a special algorithm, the nested radiosity (Chelle et al., 1998), that allows for a precise estimation of light absorption at the level of small canopy elements (typically 1 cm²). It takes into account multiple scattering, allows for infinitisation of the scene (by virtual replication) and performs in a reasonable time (typically a few minutes).

The idea is to mix:
- a projection model (Z-buffer) that solves the first order illumination,
- a model that solves the radiosity equations for the light exchanges between a canopy element and its close neighbourhood,
- and a model that solves turbid medium equations for the exchanges between a canopy element and the rest of the canopy.

**Reference:**  
Michael Chelle, Bruno Andrieu, K. Bouatouch. Nested radiosity for plant canopies. *The Visual Computer*, 1998, 14, pp.109-125. [10.1007/s003710050127](https://doi.org/10.1007/s003710050127). [hal-02697207](https://hal.inrae.fr/hal-04945340v1)

## Content

The suite is composed of two main sub-models : MCSail, that computes turbid
medium equations on a layered canopy (derived from the SAIL model (Verhoef, 1984) and Canestra, that computes radiosity
and projection. The suite also includes two utilities : periodise, that
makes a scene suitable for infinite replication and S2v, that transforms a
3D scene in a 1D multi-layer system.

It operates on a special scene object called Caribuscene, composed of a list
of triangles with optical properties representing the plants, a set of
direction and intensities (called light sources) representing the sky and
a pattern delimiting the scene, used for infinitisation.

This model is completed with a set of utilities for visualisation
(using PlantGL), import of caribuscene from files or MTG, and tools for
building light sources from meteorological data

Ref: Wout Verhoef (1984), Light scattering by leaf layers with application to canopy reflectance modeling: the SAIL model. Remote Sensing of Environment, 16, 125-141


## The Python port

This branch is Caribu in Python (numpy, numba): the projection, radiosity, nested radiosity, s2v, mcsail and
periodise of libcaribu's C++, without executables or files. The previous package is frozen in
`openalea.caribu.legacy`; it drives libcaribu and is the port's reference in the tests.

```python
from openalea.caribu import api, lights

sky = lights.turtle_lights("46", "soc", energy=1.)             # or lights.Sky(4, 5).diffuse(...), sun_light(...)
out = api.raycasting(triangles, materials, sky, domain=None)   # direct light
out = api.radiosity(triangles, materials, sky)                 # every order of scattering
out = api.mixed_radiosity(triangles, materials, sky, domain, soil_reflectance=0.15, diameter=0.5, layers=10,
                          height=1.)                            # nested radiosity in an infinite canopy
raw, aggregated = api.CaribuScene(scene, light=sky, pattern=domain, opt={"par": material}).run(direct=False,
                                                                                                infinite=True)
```

- **Exactness first.** With `precision="legacy"` (the default), every output equals the legacy's to its printed
  digits: canestrad's Etri.vec0, E0.dat, B.dat, solem.dat, its matrix files, s2v's and mcsail's files, the API's
  dicts, on the filter and the 100-plant Adel scenes (`test/comparison`). The legacy's own tests run on the port.
- **Corrections.** `corrections="recommended"` (with `precision="full"`; `openalea.caribu.corrections` lists them)
  corrects the legacy's behaviours, each tested by the physics it restores (`test/test_corrections.py`): a
  triangle across the pattern's borders lit whole, the same canopy in metres and in centimetres lit alike (the near
  field's voxels, the tiling of the neighbouring patterns), the layers' exact areas, each face's irradiance
  accumulated (leaves with ρ = τ budgeted), one azimuth convention for the sky and the sun, and more.
- **The metafspm component.** `openalea.caribu.component.CaribuLight` lights the elements of one or several plant
  populations from their triangles (`ds.triangles()`), as a metafspm environment model (`CaribuLightModel`):
  Erel and PARa per element, the canopy kept between lightings; with the corrections by default.
- **Speed** (100-plant Adel scene, 34,647 triangles, 46 sources, a laptop CPU; legacy → port): direct 1.9 s →
  0.3 s; infinite 3.1 → 0.5; pure SAIL 3.3 → 0.6; nested radiosity (D = 0.5) 11.3 → 1.4; full radiosity of the
  whole scene 307 → 62. `openalea.caribu.canopy.Canopy` keeps what depends only on the geometry: a second band or
  a new hour costs 0.03 s (direct) or 0.12 s (nested).
- **GPU:** `device="cuda"` computes the near field (radiosity's form factors) with numba.cuda, from the CPU
  kernels' own functions; tested under numba's CUDA simulator.
- **Light sources:** `lights` holds the turtle skies, the sector sky (soc, uoc), the sun's position, Spitters'
  diffuse fraction and Kaplanis' hourly radiation.
- **Examples:** `example/light_modes.py` (the models compared), `example/day_course.py` (a day hour by hour),
  `example/canopy_light.py` (CaribuLight on a stand grown by Adel's port); the tutorial
  `example/tutorials/tutorial_caribu.ipynb`.

Tests: `pytest test` (the comparisons with the legacy need libcaribu, in the environment below).

## Installation

### Users

```bash
mamba env create -n caribu -c openalea3 -c conda-forge openalea.caribu 
``` 

### Developers

```bash
git clone 'https://github.com/openalea/caribu.git'
cd caribu
mamba env create -n caribu_dev -f ./conda/environment.yml
```

## License

**Caribu** is released under the open source **CeCILL-C license**.  
See the [LICENSE](LICENSE) file.
