=============
alinea.caribu
=============

.. {# pkglts, doc


.. image:: https://readthedocs.org/projects/caribu/badge/?version=latest
    :alt: Documentation status
    :target: https://caribu.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/openalea-incubator/caribu/actions/workflows/conda-package-build.yml/badge.svg
    :alt: CI status
    :target: https://github.com/openalea-incubator/caribu/actions/workflows/conda-package-build.yml


.. image:: https://anaconda.org/openalea3/alinea.caribu/badges/version.svg
    :target: https://anaconda.org/openalea3/alinea.caribu

.. image:: https://anaconda.org/openalea3/alinea.caribu/badges/latest_release_date.svg
    :target: https://anaconda.org/openalea3/alinea.caribu

.. image:: https://anaconda.org/openalea3/alinea.caribu/badges/platforms.svg
    :target: https://anaconda.org/openalea3/alinea.caribu

.. image:: https://anaconda.org/openalea3/alinea.caribu/badges/license.svg
    :target: https://anaconda.org/openalea3/alinea.caribu

.. image:: https://anaconda.org/openalea3/alinea.caribu/badges/downloads.svg
    :target: https://anaconda.org/openalea3/alinea.caribu

.. #}

What is Caribu ?
----------------

Caribu is a modelling suite for lighting 3D virtual scenes, especially designed
for the illumination of virtual plant canopies such as virtual crop fields.
It uses a special algorithm, the nested radiosity (Chelle et al., 1998), that
allows for a precise estimation of light absorption at the level of small
canopy elements (typically 1 cm2). It takes into account multiple scattering,
allows for infinitisation of the scene (by virtual replication) and performs
in a reasonable time (typically a few minutes).

The idea is to mix a projection model (Z-buffer) that solves the first order illumination,
a model that solves the radiosity equations for the light exchanges between a
canopy element and its close neighbourhood, and a model that solves turbid
medium equations for the exchanges between a canopy element and the rest of
the canopy.

Ref: Michaël Chelle, Bruno Andrieu, K. Bouatouch. Nested radiosity for plant canopies. The Visual Computer, 1998, 14, pp.109-125. `⟨10.1007/s003710050127⟩ <https://doi.org/10.1007/s003710050127>`_. `⟨hal-02697207⟩ <https://hal.inrae.fr/hal-04945340v1>`_

Content
'''''''

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

Installation
------------

.. toctree::
    :maxdepth: 2

    ./install/install.rst

Documentation
-------------

Notebook
''''''''

.. toctree::
    :maxdepth: 2

    ./notebook/notebook.rst


References
''''''''''

.. toctree::
    :maxdepth: 4

    ./_dvlpt/modules.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Authors
-------

.. include:: ../AUTHORS.rst

Contributing
------------

.. toctree::

    ./contributing.rst

License
-------

**Caribu** is released under the open source CeCILL-C license. See LICENSE
file.

