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

### Developers (unix)

```bash
git clone 'https://github.com/openalea/caribu.git'
cd caribu
# this will create a fresh astk_dev env for you
mamba env create -f ./conda/environment.yml
# As an alternative, if you want to work in an existing environment:
mamba install --only-deps -c openalea3 -c conda-forge openalea.caribu
mamba install oawidgets 
pip install -e .[doc,test] -vv
```

### On windows, building c extension requires a special dedicated environment
mamba env create -n build_win -f ./conda/build_env_win.yml
mamba activate build_win
pip install . --config-setting=build-dir=build
mamba deactivate
mamba env create -n caribu_dev -f ./conda/pydev.yml
mamba activate caribu_dev
pip install -e .[test,doc] --config-setting=build-dir=build
## Installation

See [Installation Guide](./install/install.rst)

---



## License

**Caribu** is released under the open source **CeCILL-C license**.  
See the [LICENSE](LICENSE) file.
