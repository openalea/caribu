# openalea.caribu

[![Documentation status](https://readthedocs.org/projects/caribu/badge/?version=latest)](https://caribu.readthedocs.io/en/latest/?badge=latest)
[![CI status](https://github.com/openalea/caribu/actions/workflows/openalea-ci.yml/badge.svg)](https://github.com/openalea/caribu/actions/workflows/openalea-ci.yml)  
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


## Installation

### Users

```bash
mamba create -n caribu -c openalea3 -c conda-forge openalea.caribu 
``` 

### Developers

#### Build cpp extension

```bash
git clone 'https://github.com/openalea/caribu.git'
cd caribu
# unix (conda required as mamba does not instantiate env vars)
conda env create -n caribu_cpp -f ./conda/unix_build_env.yml
# windows (conda required as mamba does not instantiate env vars)
conda env create -n caribu_cpp -f ./conda/windows_build_env.yml
conda activate caribu_cpp
pip install -e .[test] -vv
```

#### Python/doc development

```bash
git clone 'https://github.com/openalea/caribu.git'
cd caribu
#(conda required as mamba does not instantiate env vars)
conda env create -n caribu_py -f ./conda/python_dev_env.yml
conda activate caribu_py
pip install -e .[test,doc] -vv
```
Note : on unix, build and python dev environnment can be combined, but not on windows

## License

**Caribu** is released under the open source **CeCILL-C license**.  
See the [LICENSE](LICENSE) file.
