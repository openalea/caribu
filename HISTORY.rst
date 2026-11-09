.. :changelog:

History
-------

Unreleased (the metafspm port)
------------------------------

* ``PlantOptics`` reads the elements' ``lit_area`` from ``CaribuLight`` (an input): the weight of its reflectance's
  and transmittance's means in a scene's summaries is on the plant's DataStructure whatever else its plant model reads
  (CNW-Grass's gas exchange stand warned "averaged without a declared weight"). ``example/canopy_light.py``: the
  Scene's time step the duration ``"1 h"``.
* metafspm fix-B (a packed run's frames folder removed once its archive is checked): ``simulate(outputs=)`` packs
  the frames into ``frames.fspm.zip`` when the run finishes (``SceneRecorder(pack_at_finish=True)``: the archive
  checked, then ``frames/`` removed); ``example/exchange/export_frames.py`` packs both of its exports and validates
  the archives, ``import_frames.py`` reads ``outputs/frames.fspm.zip``; user tutorial 1 lists ``frames.fspm.zip``
  (re-run).
* Breaking: the projection threads are named ``threads`` everywhere, no longer ``workers`` (no alias):
  ``CaribuLight``'s setting (in a metafspm Scene, its ``Compute(threads=)`` unless the scenario gives it; metafspm's
  process pool and ``Compute.workers`` are removed, devplan_generic_kernels QG-3), ``Canopy(threads=)``,
  ``direct.first_order``/``run``, ``radiosity.run``, ``nested``, ``nearfield``, and the legacy entry points
  ``raycasting``, ``radiosity``, ``mixed_radiosity`` (and their ``x_`` forms) and ``CaribuScene.run(threads=)``.
* metafspm P1 (the MPG's properties as columns): ``simulate``'s stand-in plant writes its elements' values with
  ``g.set_values`` and reads its position with ``g.values_at``; the ``api``'s star writes an MPG's columns in bulk
  (``g.set_values``), a plain openalea MTG keeping openalea's API.
* Breaking (metafspm O-12, part b): ``CaribuLightModel``'s schedule is split, the lighting (``CaribuLight``'s rate
  stage: its optics and Erel) every 4 h, covering its interval (metafspm's ``Cadence("4 h", at="start")``), and the
  absorbed PAR (its state stage, ``"absorbed_light"``) every step. The scenario's ``"cadence"`` key
  (``{"light": "1 h"}``) sets the lighting's cadence and replaces ``"interval"``; ``CaribuLightModel.interval`` and
  ``between`` are removed (no alias). ``simulate`` keeps ``{"cadence": {"light": "1 h"}}``.
* Behaviour change (metafspm O-12, part b): Caribu's own default lighting (every 4 h) now covers its interval, giving
  the PAR-weighted mean Erel over the sun's course in that interval, instead of holding the lit step's sun until the
  next lighting.
* metafspm O-12 (part b): ``SimplePlantModel`` and the examples' plant models declare their parts (``parts = {...}``,
  ``LitShoot`` nesting Adel's ``ScenarioShoot``) instead of building and running their components by hand.
* metafspm O-12 (part b): the light clips the planted plants (the mask ``"planted"``, was ``"emerged"``); the
  examples' and tests' stands come from metafspm's ``stand`` with ``Planting(depth=)`` (Adel's ``sown_stand`` gone).
* metafspm O-12 (part a): a Scene's start is a date (``Scene(start="2026-06-21 11:00")``, was ``(day_of_year,
  hour)``): ``openalea.caribu.simulation.calendar_date(day_of_year, hour)`` gives it and ``hours_since(scene,
  midnight)`` a summary's hour; ``simulate``'s start, the examples' and the tests' are dates; a component lit without
  a Scene start takes ``WITHOUT_START`` (``"2026-01-01 00:00"``, was day 1, 0 h); ``CaribuLightModel.interval`` is
  ``"4 h"`` (was ``4 * 3600.``) and the frames are written every ``"step"`` (was 1); the developer tutorials
  re-executed.
* metafspm O-10: ``PlantOptics``' ``PAR_reflectance`` and ``PAR_transmittance`` are summarised weighted by
  ``lit_area``; user tutorial 5's synthetic observations made again with the sun at mid-step (the fit finds 2.21 cm
  again); the tutorials re-executed.
* In a metafspm Scene, ``CaribuLight``'s ``device`` setting (the near field's) defaults to the Scene's
  ``Compute(device=...)``, with metafspm's GPU backend; a scenario's ``device`` overrides it (README,
  doc/performance.rst).
* The simulation step is each Scene's (metafspm's ``DataStructure.simulation_time_step``): the tests'
  ``Choregrapher().reset()`` and ``set_simulation_time_step`` calls are removed.
* Behaviour change: the sun over the step. ``CaribuLight.step_sources`` splits the lit step into panels of at most an
  hour, each panel's sky and sun at its middle, weighted by its share of the step's PAR, consistent with PARi read as
  the step's mean: an hourly step's sun is at its middle (it was at the step's end), a daily step gets the sun's
  course. PARa differs accordingly.
* Breaking: the soil is the Scene's soil surface (``ds.geometry.clipped("air")``); the ``soil_level`` setting is
  removed.
* metafspm O-5: the numba kernels are in one module, ``openalea.caribu.kernels`` (projections, budgets, periodic
  copies, direct light, near field and its option bits and work rows, nested radiosity, SAIL's layers, the solver;
  the GPU clones the near field's from it); its ``warm_up`` (a tiny scene: direct light, radiosity, mixed radiosity,
  legacy and full precision) is registered for metafspm's warm_up (entry point ``openalea.metafspm.warm_up``) and
  run before the conda recipe's tests.
* metafspm O-10 (forcings over the step, weighted summaries): PARi is read as its mean over the step (the summary's
  PARi the hour's mean); the light's intensive variables (reflectance, transmittance, Erel, Erel_incident, PARa,
  PAR_incident) are declared with their weight, the lit area, by which metafspm's per-plant summaries weigh them.
* ``CaribuLight``: a setting for the threads projecting the sources (``"auto"`` half the processors, or a number; in
  a Scene, its ``Compute``'s unless the scenario gives it; introduced as ``workers``, now ``threads``, see above) and
  ``pixel_size`` (m), replacing ``screen_size`` (breaking): the screen's side is the scene's horizontal extent over
  it, so that a larger pattern keeps its resolution (None: 1536 pixels, the reference); both checked after a
  scenario (``check_settings``). ``simulate(compute=)`` forwards metafspm's ``Compute`` to the Scene.
* The metafspm port: ``CaribuLight``, the metafspm component, and ``CaribuLightModel``, its environment model (the
  elements' triangles from a population or a union of populations, their optics from ``PlantOptics``, a plant
  model's component, or their label; the diffuse sky and the sun from the Scene's calendar; direct light, nested
  radiosity or radiosity; Erel and PARa per element; PARi a required forcing, the diffuse fraction an optional one);
  the corrections of the direct light, the radiosity, the turbid medium, the budgets and the sky (C10 K1 to K5,
  ``corrections.resolve``: ``"recommended"`` or names, none at the legacy's precision); ``openalea.caribu.simulation``
  (``simulate``, ``parameters``), user tutorials 01-06 and developer tutorials 00-02; the exchange examples
  (``example/exchange``: ``export_frames.py``, a filtered export for logging, ``import_frames.py``) through encoding3d
  (formerly fspm-TF).

creation (TODAY)
----------------

* First release on PyPI.


version 1.4
------------

* caribu is compatible with North, positive clockwise azimuth convention for light sources
* Correct positioning of sources has been tested in command line interface