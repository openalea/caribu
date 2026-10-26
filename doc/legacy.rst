=================================
The legacy: algorithms and quirks
=================================

The port reproduces libcaribu's executables (``periodise``, ``s2v``, ``mcsail``, ``canestrad``) and the Python layer
driving them (``openalea.caribu.legacy``), digit for digit with ``precision="legacy"`` and no corrections. This page
describes their algorithms as the port implements them, then registers every quirk of the legacy with what the port
does about it. The equations are in :doc:`equations`, the corrections' cost in :doc:`performance`.


The legacy's chain
==================

caribu's Python wrote text files and ran the executables in a temporary directory:

================================================= =========================================== ==========================
step                                              legacy                                      port
================================================= =========================================== ==========================
scene, optics, lights, pattern, sensors           ``.can``, ``.opt``, ``.light``, ``.8`` files  arrays (``formats`` reads
                                                                                              and writes the files)
the scene moved into the pattern                  ``periodise``                               ``periodic.periodise``
the canopy as layers of a turbid medium           ``s2v`` (``leafarea``, ``<band>.spec``)       ``sail.layers``
the layers' fluxes (multilayer SAIL)              ``mcsail`` (``mlsail.env``)                 ``sail.profiles``
direct light, radiosity, budgets                  ``canestrad`` (``Etri.vec0``, ``solem.dat``)  ``direct``, ``nearfield``,
                                                                                              ``solver``, ``budgets``
the API: per triangle and primitive               ``caribu.py``, ``CaribuScene``                ``api``
================================================= =========================================== ==========================

``canestrad``'s modes: ``-1`` direct light only (``direct.run``); ``-8`` the pattern repeated (``direct.run`` with a
pattern); ``-d D`` radiosity within a sphere of diameter ``D`` (``-d -1``: the whole scene, ``radiosity.run``), with
``-e mlsail.env`` the far field beyond it (``nested.run``); ``-f``, ``-w`` the form factors and the far field's
weights written by a first band and read by the next (``nested.run(weights=...)``, a ``Canopy``).


Direct light
============

**The screen** (``projection.legacy_screen``). For a beam ``u`` (normalised in float32), an orthographic screen
covers the scene's bounding box (x, y the pattern's in an infinite canopy) projected along the beam, its plane lifted
one length unit above the scene; its image axes ``v`` (horizontal, perpendicular to the beam) and ``w``; a vertical
beam (``|u_z + 1| < 1e-6``) has its own screen, the bounds themselves. Points and vectors are float32 (canestrad's
``reel``), its locals double, in its order of operations.

**The z-buffer** (``rasterise``, ``_draw``, ``_colour_triangle``: ``Canopy::projplan``, ``colorie_triangle``). Each
triangle, in scene order, is projected and split at its middle vertex into flat-based triangles; a pixel row ``j``
and column ``i`` are drawn when the pixel lies within the triangle (scanline at ``(j + 0.5)/k``, columns
``[x_left k + 0.5, x_right k − 0.5]``), with the scale ``k = (T − 1)/extent``. The nearest triangle along the beam is
kept (strict ``<``: the first drawn wins a tie); the depth buffer starts at ``99999999999.9``. A triangle seen edge-on
is skipped.

**Sunlit areas** (``projected_areas``). Each pixel adds its area :math:`A_\text{pix}` (canestrad's
:math:`d_u d_v / T^2 / \cos\theta`, the correction ``pixel_area``: :math:`(T - 1)^2`) to the face the beam meets
(upper when :math:`\mathbf{u}\cdot\mathbf{n} < 0`, float32), and removes it from a leaf's other face (the light it
transmits). The sources' areas are summed in their order (``direct.first_order``).

**Sensors** (``sensor_areas``: ``colorie_capteur``). Virtual triangles drawn after the scene (and after the tiling):
a pixel counts when the sensor would be in front; they neither shade nor scatter, reflect all (ρ = 1), are lit only
from above. canestrad adds each copy's count cumulatively (the first copies counted again) and never sets their
border flag; their first-order ``Ei0`` in nested radiosity includes the far field (``B0 += Cfar`` in place).


The infinite canopy
===================

**periodise** (``periodic.periodise``). Each triangle is moved by whole pattern lengths until its centre (float32
sums, divided in double) lies in ``(xmin, xmax] × (ymin, ymax]``; the scene is raised by ``−zmin`` (with a 1 % margin,
leaving the lowest point near ``−4e-8``) when it goes below 0; written with ``%.6g`` and read back as float32.

**canestrad -8** (``border_flags``). A triangle whose centre lies outside the pattern is rejected; one with vertices
beyond a side is flagged (1 in x, 2 in y, counted with signs: a triangle beyond both sides of an axis counts 0).
projplan draws a flagged triangle with copies shifted by ``+dx`` (flag 1) or ``+dy`` (flag 2) only; flag 3 is not in
its switch and takes the previous triangle's number of copies (``border_copies``).

**infinitise** (``tiling_steps``, ``translations``, ``tile``). The pattern's corners at the scene's top give, in
pixels, the image of the lattice's steps, rounded (mean of two opposite edges, C's ``round``). The image is translated
by these steps, then repeated along them; a translated pixel in front (its depth increased by
``tan θ d_v / T`` per pixel) makes the pixel a neighbouring pattern's shade. Not done for a vertical beam.

The corrections ``border_copies``, ``tiling`` and ``sensor_copies`` replace this by copies on each side crossed, the
lattice's exact image vectors, and a count in the pattern's cell (:doc:`equations`).


Radiosity: the near field
=========================

**The receiver's discs** (``nearfield``: ``init_NFF``, ``proj_ortho``, ``NFF``). A receiver's frame is its centre,
``u = (P0 − P1)/|P0 − P1|``, ``v = w × u``, ``w`` its normal. Its hemisphere is the disc of ``NB × NB`` pixels (NB = 52,
even), pixel centres ``(2i + 1)/NB − 1``, the hemisphere's height ``√(1 − x² − y²)`` folded by symmetry; a leaf has
an upper and a lower disc. Each emitter is moved into the frame, cut by the receiver's plane into up to three
triangles (vertices at ``|z| ≤ 1e-6`` on the plane, lifted by ``1e-6``, zero x, y set to ``1e-7``: absolute, in the
scene's unit), each projected as the spherical triangle of its edges' great circles; a pixel inside all three, the
nearest (the most negative ``d`` of the emitter's plane ``b·X = −1``, strict ``<``) keeps the emitter. An opaque
emitter seen from behind is skipped. The pixel test takes each edge plane's x coefficient as 1 (``init_MPE``
normalises it on its first non-zero coefficient).

**The rows** (``_receiver_rows``). Per receiver, each emitter face's pixels on the upper disc minus those on the lower
one (one signed counter); the face is the one its centre faces, read when the discs are scanned (the last copy
activated). ``dF = 4 / (π NB²)``, not renormalised: an empty hemisphere sums to 1.002.

**The candidates** (``grid``, ``_neighbourhood``, ``_candidates``: ``calc_FF_Bfar``). Voxels of side ``2 · ratio ·
R`` (R = D/2, ratio 1) over the scene, padded; a BSP of boxes split at voxel faces (z first) while they hold more than
50 diffusers (``grille_par``), a straddling diffuser in both. The receiver's voxel ``int(trunc(G − O) / side)``
(correction ``voxel_floor``) and its neighbours towards the nearer faces, up to 8 boxes, with the pattern's
translations ±period in an infinite canopy. An emitter is a candidate when its translated centre lies within R of
the receiver's (``-B``: within R of the sphere, its hits beyond R dropped); a face is projected once untranslated and
once translated at most (the anti-doublon flags). The full radiosity is one box holding all, R the scene's diagonal.

**The far field** (``_environment``, ``_faces``: ``env``). A pixel with no emitter sees the far field where its
direction leaves the sphere, at ``z = z_receiver + R d_z`` (float32 Point arithmetic): ``E−`` looking up, ``E+``
looking down, from canestrad's column-major table of the profile, interpolated between levels with inverted weights
(``f`` on the lower level), 0 outside ``(z_soil, z_top]``. Its levels' weights (bfc · dF) are kept for ``-w``.


Radiosity: the solver
=====================

``hd_mv_mlt`` (``solver.product``): per receiver, ``B_upper − Σ c dF (c > 0 ? ρ_sup : τ_inf) B_j``, a leaf's
``B_lower − Σ c dF (c > 0 ? τ_sup : ρ_inf) B_j``. ``hd_mgcr`` (``solver.mgcr``): meschach's restarted MGCR (Leyk's
modified generalised conjugate residuals), restart 100, from ``x = 0``, tolerance ``1e-6`` relative to the first
residual on its estimated residual, at most 1000 steps, transcribed in its order of operations.


The turbid medium
=================

**s2v** (``sail.layers``). The periodised scene (``%.6g``, read as float32) in ``n`` layers of ``H/n`` (boundaries
cumulated from the bottom: a vertex on a boundary in the lower layer, above ``H`` in the top layer, below 0 dropped).
Each triangle is subdivided 4 ways at its edges' midpoints (depth first, 6 levels at most) while its vertices lie in
different cells (``int(x/period)``, ``int(y/period)``, the layer); a piece in one cell, or at the last level by its
centre, adds ``0.5 |cross|`` (a leaf) or ``0.25 |cross|`` (a stem) to its layer, in its zenith class (18 classes of
5°, ``acos(|n_z|/|n|) · 57.29577951``, from s2v's float32 normal). Per layer: the classes' frequencies, the leaf area
index, the area-weighted ρ and τ (percent, a leaf's the mean of its faces, a stem's its ρ). Written with 6 decimals
(``leafarea``) and 3 (``<band>.spec``, percent).

**mcsail** (``sail.profiles``). Verhoef's SAIL per layer (``msailad``: the coefficients; ``mlayer``: the layer's
reflectances and transmittances; ``rtp == 0`` taken as 0.1), the layers added from the soil up (``mprofil``), each
source a collimated beam of its horizontal energy (``es = 1``, ``ed = 0``), summed without normalisation. Its
``mlsail.env``: per level ``z``, ``E−(z_i) = transdif[i]``, ``E+(z_i) = refdif[i − 1]``, with 6 decimals.

**sail_pur** (``sail.far_field``). With D = 0, each face receives the profile at its centre's height (inverted
weights; beyond the top, canestrad's table read past its column: E− there is E+ at the soil, E+ the soil level's
height), by Ross's formula, and B = B0 + ρ E + τ E across a leaf.


Budgets and outputs
===================

``genres`` (``budgets.triangle_budgets``): an opaque face ``Ei = B/ρ``, ``Eabs = Ei − B``; a leaf's two faces by
inverting ``B_sup = ρ_sup E_sup + τ_inf E_inf``, ``B_inf = ρ_inf E_inf + τ_sup E_sup`` (float32), ``Eabs = E_sup +
E_inf − B_sup − B_inf``; singular (``ρ_sup ρ_inf = τ_sup τ_inf``): −1, except ``ρ_sup = τ_inf``. ``Etri.vec0``: one row
per input triangle, ``%f``, a rejected one ``area 0`` and ``NaN`` (``Nan`` before an accepted row). caribu.py reads them
back and overwrites ``Ei`` by ``Eabs / (1 − ρ − τ)``; ``CaribuScene`` converts units and aggregates per primitive, the
area-weighted mean of the printed values.


The quirk registry
==================

Each behaviour of the legacy that is odd, with what the port does: **reproduced** (with ``precision="legacy"`` and
no corrections; the port's numbers are then the legacy's), **corrected** by the named correction
(``openalea.caribu.corrections``), **fixed** (an API bug, not physics, fixed in the port whatever the options),
**kept** (deliberately kept in every mode), **dropped** (files, processes, dead code: nothing to port).

Parsing, geometry, rejection
----------------------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-G01  float32 geometry and z-buffers, arithmetic partly in double; outputs ``%f``          reproduced (float32 points
                                                                                            in every mode; precision
                                                                                            "full": no printing)
Q-G02  ``.can`` parser: fixed buffers, ≥ 10-vertex polygons misread, unknown letters         dropped (arrays; ``formats``
                                                                                            parses tokens)
Q-G03  degeneracy from the first three vertices: equal vertices, ``|u·v| > 0.9999999999``,  reproduced
       null cross product (no area threshold)                                               (``geometry.rejected``)
Q-G04  area ``|P0P1||P0P2| sin(acos(u·v))/2``: thin triangles lose it                        corrected: ``triangle_area``
Q-G05  a species beyond the optics, or a leaf of species 0: out-of-bounds read              fixed (NaN optics)
Q-G06  rejected triangles' rows; with ``-n`` missing rows misalign                           fixed (one row per input
                                                                                            triangle)
Q-G07  ``Nan`` before an accepted row, ``NaN`` after                                        reproduced (``etri_rows``)
Q-G08  side-effect files (logs, ``trinf.can``, matrices, debug outputs)                      dropped
Q-G09  a polygon's centre its vertices' mean; ``dGS`` its largest vertex distance           reproduced
Q-G10  ``Macos``, ``Masin``, ``Msqrt`` guards, abort beyond 1e-5                             reproduced (clamped, no
                                                                                            abort)
====== ==================================================================================== ===========================

Direct light, sensors
---------------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-D01  pitch ``extent/(T − 1)``, pixel area ``d_u d_v / T²``: a ``(T − 1)²/T²`` bias           corrected: ``pixel_area``
Q-D02  anisotropic screen, float32 depth, truncation of slightly negative bounds            reproduced
Q-D03  a beam with ``d_z ≥ 0``: ``inf``, ``nan``, −0.0 results                              reproduced (IEEE division);
                                                                                            corrected: ``upward_beams``
Q-D04  a leaf blocks the beam; its transmission is the other face's radiosity               kept (the model)
Q-D05  opaque triangles lit from both sides; seen from behind, not emitters                 reproduced
Q-D06  sensors outside the scene's bounds clipped, their copies counted cumulatively,       corrected: ``sensor_copies``
       their border flag never set
Q-D07  a triangle seen edge-on skipped                                                      reproduced
Q-D08  the last source's energy passed on (see Q-R12)                                       see Q-R12
Q-D09  the depth buffer's start ``99999999999.9``                                           reproduced
====== ==================================================================================== ===========================

The infinite canopy
-------------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-P01  border copies only +dx, +dy (a triangle across xmax or ymax loses its part beyond:   corrected: ``border_copies``
       Ei 0.67 for 1); flag 3 takes the previous triangle's copies
Q-P02  the tiling's steps rounded to pixels, then repeated (the k-th neighbour up to k      corrected: ``tiling``
       pixels off); its depth ``tan θ d_v / T`` per pixel; translates only shade
Q-P03  periodise by the centre; ``%.6g``; the bottom near ``−4e-8``                          corrected: ``periodise``
Q-P04  a centre outside ``(min, max]`` rejected                                             reproduced
Q-P05  the pattern's bounds replace x, y; z from the triangles                              reproduced
====== ==================================================================================== ===========================

Radiosity, form factors, solver
-------------------------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-R01  the receiver's voxel ``int(trunc(G − O)/side)``: in a metre scene with D < 1 m most   corrected: ``voxel_floor``
       near neighbours searched around the wrong voxel (Adel in m: 1.50 M pairs for 2.67 M)
Q-R02  candidates by their centre's distance (``-B``: to the sphere); one face per pair;     reproduced
       one untranslated and one translated image
Q-R03  voxels and BSP as a spatial index (``grille_par``: 50, ratio 1)                       reproduced, without files
Q-R04  NB even; ``dF = 4/(π NB²)`` not renormalised (an empty hemisphere 1.002)              reproduced
Q-R05  ``proj_ortho``'s absolute ``1e-6``, ``1e-7``: results depend on the unit (slightly)    reproduced in every mode
Q-R06  the pixel test's x coefficient taken as 1 (a plane normalised on y or z)              reproduced
Q-R07  a pixel on an edge: strict tests; the first projected emitter wins ties              reproduced
Q-R08  one signed counter per emitter face: seen from both discs, the counts cancel        corrected: ``face_counts``
Q-R09  each pixel's face the last activated one (a copy facing the other way)               corrected: ``projected_face``
Q-R10  the profile interpolated with inverted weights (env, sail_pur, the weights)          corrected:
                                                                                            ``profile_interpolation``
Q-R11  the far field 0 outside ``(z_soil, z_top]``; sail_pur reads past its table           corrected: ``profile_bounds``
Q-R12  ``-w`` bands scaled by the last source's energy, with an index slip: bands 2…n       reproduced with precision
       wrong unless that energy is 1                                                        "legacy"; corrected:
                                                                                            ``band_scaling``
Q-R13  MGCR: restart 100, tolerance on the estimated residual, silent non-convergence       kept (MGCR, the default);
                                                                                            GMRES as an option
Q-R14  ``-d D``: radius D/2 (a negative D: the scene's diagonal)                            reproduced
Q-R15  ``-s``, canestrad's own soil mesh (never used by caribu)                             dropped (the soil as
                                                                                            triangles)
Q-R16  the matrix files (``diag_``, ``nz_``, ``Bfar_``)                                     dropped (in memory, a
                                                                                            ``Canopy``)
Q-R17  multi-simulation scaffolding, shared memory                                          dropped
Q-R18  no light source: ``Esource`` uninitialised                                           fixed (no source, no light)
Q-R19  non-HD solver branches                                                               dropped
— (*)  a sensor beyond the scene's box or the grid crashes canestrad                        fixed (its voxel clamped)
— (*)  the sensors' first order in nested radiosity includes the far field                  reproduced (``Ei0``; the
                                                                                            first order apart)
====== ==================================================================================== ===========================

The turbid medium
-----------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-S01  stems at half their area (``0.25 |cross|``); a leaf's optics its faces' mean          kept (documented)
Q-S02  the layers' areas by 4-way subdivision (6 levels); x, y cells without the origin     corrected: ``exact_layers``
Q-S03  the zenith class from ``57.29577951``: a class edge in the lower class               reproduced
Q-S04  s2v's input contract (argument counts, file errors, ``p`` in comments)               dropped (arrays)
Q-S05  ``.spec`` with 3 decimals, in percent                                                 reproduced with precision
                                                                                            "legacy"; "full": values
Q-S06  mcsail: sources as collimated beams, summed unnormalised; ``rtp == 0`` → 0.1;        reproduced (the BRF view and
       a soil of −1 used as is; a ``VISEE`` file switches to the BRF                         Verhoef's Beta law as
                                                                                            options); the soil:
                                                                                            corrected ``no_soil``
Q-S07  ``mlsail.env`` with 6 decimals; ``E−(z_i)``, ``E+(z_{i−1})``; E− at the top 0          reproduced with precision
                                                                                            "legacy"; "full": values
Q-S08  a source's horizontal energy as SAIL's unit flux (consistent)                        kept
Q-S09  mcsail's debug files                                                                 dropped
Q-S10  periodise reads the optics it never uses                                             dropped
— (*)  canestrad's profile table is column-major: beyond the top, the next column            reproduced (sail_pur);
                                                                                            corrected:
                                                                                            ``profile_bounds``
====== ==================================================================================== ===========================

Budgets, outputs
----------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-O01  ``Ei`` by inverting B: singular leaves (ρ = τ) and black faces give −1                corrected: ``face_irradiance``
Q-O02  float32 budgets printed ``%f``: in metres, a wheat triangle's area (≈ 6e-6 m²) loses  reproduced with precision
       most of its digits, an element's mean (Wheat-BRIDGES' Erel) moves by up to 0.0125     "legacy"; "full": values
Q-O03  caribu.py overwrites ``Ei`` by ``Eabs / α``                                          reproduced in the legacy
                                                                                            API; with corrections
                                                                                            ``Ei = Ei_sup + Ei_inf``
Q-O04  units converted for the −1 sentinels too                                             reproduced
Q-O05  per primitive, the area-weighted mean over the triangles of positive area            reproduced
Q-O06  ``x_raycasting``: the last band computed, the others from its ``Ei``                  reproduced; with
                                                                                            corrections each face's
                                                                                            Eabs; sensors converted
                                                                                            once (fixed)
Q-O07  an opaque triangle's ``Ei_inf`` −1                                                   reproduced
====== ==================================================================================== ===========================

The Python layers
-----------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-Y01  CaribuScene's caches and temporary files                                             dropped
Q-Y02  several bands with a soil mesh: TypeError; the soil mesh counted as stems by s2v      fixed (the TypeError);
                                                                                            reproduced (s2v)
Q-Y03  ``auto_screen`` multiplies by the unit                                               reproduced
Q-Y04  species numbered in ``set`` order; black bodies refused                              reproduced
Q-Y05  the light's direction a tuple; dict scenes of lists of tuples                        relaxed (sequences, arrays)
Q-Y06  caribu_shell's files and logs                                                        dropped
Q-Y07  libcaribu's second driver                                                            dropped
Q-Y08  ``label.py``'s Python 3 bugs                                                          dropped (``labels``)
Q-Y09  the display's colours                                                                dropped (metafspm's
                                                                                            ``display``)
====== ==================================================================================== ===========================

Sky, light, STAR
----------------

====== ==================================================================================== ===========================
ID     legacy                                                                               port
====== ==================================================================================== ===========================
Q-L01  two azimuth conventions (light.py from X+, light_sources from North; the sector sky  reproduced; corrected:
       from X+, the sun from North)                                                         ``sky``
Q-L02  ``turtle()`` calls ``emission_inv`` with its arguments swapped                       reproduced; corrected:
                                                                                            ``sky`` (E_h / sin e)
Q-L03  ``light_source`` returns a 1-tuple of a list                                          fixed (a list)
Q-L04  turtle weights rounded, not renormalised                                             reproduced
Q-L05  the sun: 3.14 for π, a zenith named ``elev``, no night, the wrong sector;             reproduced; corrected:
       ``mergeNskies`` drops the first sky; ``RgH`` cannot run (its ``^``)                    ``sky``; ``RgH`` ported
                                                                                            (W m⁻²)
Q-L06  STAR's horizontal weights                                                            reproduced (``api.star``,
                                                                                            ``rain_star``,
                                                                                            ``light_star``)
Q-L07  the Visualea nodes                                                                   dropped (in the legacy)
====== ==================================================================================== ===========================

(*) Found while porting, not in the registry's first version.


Constants
=========

============================================ ============================================================
the screen                                   1536 pixels; lifted one length unit above the scene
vertical beam; tiling                        ``|u_z + 1| < 1e-6``; ``u_z > −1 + 1e-6``
the depth buffer's start                     ``99999999999.9`` (float32: 99999997952)
labels                                       species ``label // 1e11``; opaque when ``(label // 1000) % 1000 == 0``
collinear vertices                           ``|u·v| > 0.9999999999``
the discs                                    NB = 52, ``dF = 4/(π NB²)``
``proj_ortho``                               plane ``1e-6``, lift ``+1e-6``, zero ``1e-7``
voxels                                       side ``2 · ratio · R``, ratio 1, 50 diffusers per box
MGCR                                         restart 100, tolerance ``1e-6`` relative, 1000 steps, ``x0 = 0``
s2v                                          18 zenith classes of 5°, 6 subdivision levels, ``57.29577951``
mcsail                                       ``rtp == 0`` → 0.1; ``tan(90°)`` finite (1.6e16)
printed outputs                              ``Etri.vec0`` ``%f``; ``solem.dat`` ``%.10f``; ``mlsail.env`` ``%f``;
                                             ``.spec`` ``%.3f``; periodise ``%.6g``; caribu's ``.can`` ``%.6f``
============================================ ============================================================
