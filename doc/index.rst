======================================================
Welcome to openalea.caribu's documentation!
======================================================
.. sidebar:: Summary

    :Version: |version|
    :Release: |release|
    :Date: |today|
    :Author: See `Authors`_ section
    :License: See `License`_ section

Caribu computes the light each leaf of a plant canopy receives and absorbs, from the sky and the sun, with or without
the light the leaves scatter onto each other, in a canopy repeated on every side like a field.

User guide
==========

For newcomers to Caribu, to light models or to Python modelling: light a canopy, change its parameters, explore
them, use experimental data, light plants grown by another model. Each tutorial is a notebook in
``example/tutorials/user/``, shown here with its results.

.. toctree::
    :maxdepth: 1

    readme
    tutorials/user/01_first_run
    tutorials/user/02_change_a_parameter
    tutorials/user/03_combine_parameters
    tutorials/user/04_sensitivity_analysis
    tutorials/user/05_experimental_data
    tutorials/user/06_with_other_models
    parameters
    usage

Developer guide
===============

For those who read, check, change or couple the model: how it is written as a metafspm component, how it was ported
from the C++ executables and validated, its equations and its performance. Best read after the first user tutorial.

.. toctree::
    :maxdepth: 1

    tutorials/developer/00_overview
    tutorials/developer/01_caribu_in_metafspm
    tutorials/developer/02_extend_the_component
    components
    equations
    legacy
    performance
    api
    authors


.. seealso::

   More documentation on OpenAlea can be found on the
   `OpenAlea <https://openalea.readthedocs.io>`__ website.

Authors
=======

.. include:: ../AUTHORS.rst

License
=======

caribu is released under a Cecill-C License.

.. note:: `Cecill-C <http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html>`_
    license is a LGPL compatible license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
