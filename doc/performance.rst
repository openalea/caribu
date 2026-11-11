=====================================
Precision, corrections and speed
=====================================

Precision
=========

``precision="legacy"`` (the default of the legacy API: ``raycasting``, ``radiosity``, ``mixed_radiosity``,
``CaribuScene``) reproduces the legacy's numbers digit for digit: canestrad prints its results with 6 decimals and
caribu reads them back, s2v and mcsail exchange files of 3 to 6 decimals; the port rounds its values the same way.
It is the regression reference (``test/comparison``).

``precision="full"`` keeps the computed values. It changes the results by the legacy's rounding only, except in
metres: canestrad prints the triangles' areas with 6 decimals, and a wheat element's triangles (around 6e-6 m²)
lose most of their digits, the area-weighted means of an element (Erel) by up to 0.0125.

It also lets a ``Canopy`` keep the nested radiosity's near field and the far field's weights between calls (another
band, other optics, a new hour on the same geometry): canestrad recomputes them at each call, and so does the legacy
precision.

Corrections
===========

``corrections="recommended"`` (with ``precision="full"``; the default of the metafspm component) corrects the
legacy's behaviours (``openalea.caribu.corrections``; their equations in :doc:`equations`). They do not make the
models faster, and one makes nested radiosity slower: the receiver's voxel ``floor((G − O) / side)``
(``voxel_floor``) finds the near neighbours canestrad missed. On the 100-plant Adel scene (in m) the near field holds
2.67 million emitter–receiver pairs instead of 1.50 million: the legacy left 44 % of them to the far field.

Speed
=====

The 100-plant Adel scene (34,644 triangles, in m), the 46-direction turtle sky, a laptop (2 performance cores, 8
efficient and 2 low-power ones), 26 October 2026:

.. list-table::
   :header-rows: 1

   * - model
     - libcaribu
     - port
     - port, corrected (``full``)
   * - direct light, infinite canopy
     - 3.2 s
     - 0.6 s
     - 0.6 s
   * - pure SAIL
     - 3.3 s
     - 0.7 s
     - 0.7 s
   * - nested radiosity, D = 0.5 m
     - 11.8 s
     - 1.5 s
     - 3.1 s (the near pairs found)
   * - radiosity of the whole scene
     - 307 s
     - 62 s
     -  
   * - the same, again on the same geometry
     - —           1.8 s
     -  
     - 0.2–0.3 s (``Canopy``)

Where the time goes: in radiosity on 5,000 triangles (1.1 s), the near field's form factors take 85 % and the
linear solve 1.3 % (5 steps); in the nested radiosity of the Adel scene (3.1 s), the near field 63 %, the direct
light and the turbid medium most of the rest, the solve 3 %.

The radiosity method
====================

The radiosity system is solved by meschach's MGCR, transcribed (``radiosity_method="mgcr"``, canestrad's), or by scipy's
GMRES (``"gmres"``, ``precision="full"``): the same radiosities in the same few steps. The solve is a few percent of
the run: neither a faster solver nor a GPU would change the time.

Threads, screen and memory
==========================

The sources are projected in parallel threads (numba releases the GIL), each holding a screen of float32 depths of
side ``size`` pixels: the component's ``threads`` setting bounds them (``"auto"``, by default: half the processors;
in a metafspm Scene, its ``Compute(threads=)`` unless the scenario gives it). The screen's side is 1536 pixels by
default whatever the scene's extent; ``pixel_size`` (m) sets it to the scene's horizontal extent (the pattern's) over
the pixel size instead, so that a larger pattern keeps its resolution: a 0.3 m pattern at ``pixel_size=0.3 / 1536``
is the default's screen, a 3 m pattern at that pixel size a screen ten times as wide (and its buffers a hundred
times as large: bound the threads accordingly).

The compiled kernels
====================

Every numba kernel (the projections' z-buffer, the budgets, the periodic copies, the near field, SAIL's layers, the
solver) is in ``openalea.caribu.kernels``, ``@njit(cache=True)``: compiled at its first call, once per machine.
``kernels.warm_up()`` compiles them on a tiny scene (the direct light, radiosity and mixed radiosity, legacy and
full precision); metafspm's ``python -m openalea.metafspm warm_up`` runs it with every port's, at install and before
the tests.

GPU
===

``device="cuda"`` computes the near field on a GPU (numba.cuda): one thread per receiver, with the CPU kernels'
own functions compiled as device functions, the same arithmetic in the same order. The near field is most of the
time of radiosity and nested radiosity: it is where a GPU helps. The GPU path is experimental: it should work on
suitable hardware but has not been tested on a GPU yet. The kernel is tested under numba's CUDA simulator, which runs
it in Python. On a GPU, the results should agree with the CPU's within round-off, not bit for bit: numba does not
forbid CUDA's compiler (NVVM) to fuse a multiply and an add into one operation, which it does by default.

In a metafspm scene, the ``device`` setting that a scenario does not give takes the scene's
``Compute(device=...)``: ``Scene(..., compute=Compute(device="cuda"))`` puts Caribu's near field on the GPU together
with the plants' reduced kinds and linear solves (metafspm's "Running on a GPU", which needs CuPy); a scenario's
``device="cpu"`` keeps Caribu on the CPU.

The direct light stays on the CPU: canestrad's z-buffer resolves the depth ties in its drawing order, which a
parallel z-buffer cannot reproduce bit for bit; it is already fast and its projections are kept by a ``Canopy``.
With the corrections, exactness to canestrad no longer binds, and a GPU z-buffer becomes possible.
