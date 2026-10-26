=============================
The parameters in plain words
=============================

The main parameters of the user tutorials' stand (``SimplePlant``) and of the light model (``CaribuLight``), with
their default values. ``openalea.caribu.simulation.parameters()`` lists every parameter and setting with its value
in the model's unit and its description, and :doc:`components` gives the complete reference.

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - parameter
     - what it controls
     - default
   * - ``n_leaves``
     - number of leaves per plant
     - 4
   * - ``leaf_length``, ``leaf_width``
     - size of a leaf
     - 20 cm × 1.5 cm
   * - ``leaf_inclination``
     - angle of the leaves above the horizontal (0: flat, 90: upright)
     - 30°
   * - ``plant_height``
     - height of the stem, where the top leaf starts
     - 40 cm
   * - ``leaf_reflectance``
     - share of the light hitting a leaf that it reflects
     - 0.10
   * - ``leaf_transmittance``
     - share of the light hitting a leaf that it lets through
     - 0.05
   * - ``mode``
     - the light model: ``direct`` (no scattering), ``nested`` (with scattering, in an infinite canopy),
       ``radiosity`` (with scattering, the scene alone)
     - ``direct``
   * - ``sun``
     - the sun, besides the diffuse sky
     - no
   * - ``latitude``
     - the site's latitude
     - 48.85° (Paris)
   * - ``soil_reflectance``
     - share of the light reaching the soil that it reflects (with scattering only)
     - 0.15

A leaf absorbs what it neither reflects nor transmits: 85 % with the defaults. The PAR is counted in µmol of photons
per m² per second (µmol m⁻² s⁻¹); the hours are in universal time. The stand (its rows and spacing) and the time (its
day and hours) are arguments of ``simulate``.
