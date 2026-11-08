.. :changelog:

History
-------

Unreleased (the metafspm port)
------------------------------

* Breaking: the projection threads are named ``threads`` everywhere, no longer ``workers`` (no alias):
  ``CaribuLight``'s setting (in a metafspm Scene, its ``Compute(threads=)`` unless the scenario gives it; metafspm's
  process pool and ``Compute.workers`` are removed, devplan_generic_kernels QG-3), ``Canopy(threads=)``,
  ``direct.first_order``/``run``, ``radiosity.run``, ``nested``, ``nearfield``, and the legacy entry points
  ``raycasting``, ``radiosity``, ``mixed_radiosity`` (and their ``x_`` forms) and ``CaribuScene.run(threads=)``.

creation (TODAY)
----------------

* First release on PyPI.


version 1.4
------------

* caribu is compatible with North, positive clockwise azimuth convention for light sources
* Correct positioning of sources has been tested in command line interface