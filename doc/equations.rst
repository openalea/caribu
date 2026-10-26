=========
Equations
=========

The physics of the port, model by model, with the functions that compute it. The code reproduces the legacy's
numerics first (float32 points, its order of operations), which hides the equations: they are written here. The
corrections (``openalea.caribu.corrections``) are given where they change an equation.

Notation: a scene of triangles :math:`t` of area :math:`S_t`; a translucent triangle (a leaf) has two faces, upper
and lower, an opaque one (a stem, the soil) one face. Each face :math:`f` has a reflectance :math:`\rho_f` and,
a leaf's, a transmittance :math:`\tau_f`. A light source :math:`s` is :math:`(E_s, \mathbf{u}_s)`: :math:`E_s` the
irradiance it gives a horizontal plane, :math:`\mathbf{u}_s` its beam's direction (downwards: :math:`u_z < 0`).
Energies are per unit area of triangle. :math:`B_f` is a face's radiosity (the light it emits: reflected and
transmitted), :math:`E_f` its irradiance (the light reaching it).


Light sources (``lights``)
==========================

**Sector sky** (``Sky.diffuse``): sectors of azimuth width :math:`\Delta\varphi` and zenith width
:math:`\Delta\theta` around :math:`(\theta, \varphi)`, with :math:`x = \cos(\theta - \Delta\theta/2)`,
:math:`y = \cos(\theta + \Delta\theta/2)`. A sector's share of the sky's horizontal irradiance is

.. math::

   \text{uoc:}\quad \frac{(x^2 - y^2)\,\Delta\varphi}{2\pi}, \qquad
   \text{soc:}\quad \left(\frac{3}{14}(x^2 - y^2) + \frac{6}{21}(x^3 - y^3)\right)\frac{\Delta\varphi}{\pi},

the uniform overcast sky and the standard overcast sky (Moon & Spencer, radiance
:math:`\propto (1 + 2\cos\theta)/3`). The turtle skies (``turtle_lights``) are 16 or 46 fixed directions with
tabulated weights of the same skies.

**The sun** (``Sun.place``): on day :math:`J` at solar hour :math:`t` and latitude :math:`\phi`, with
:math:`\alpha = 2\pi(J - 1)/365` and the hour angle :math:`h = 2\pi(t - 12)/24`,

.. math::

   \delta = 0.006918 - 0.399912\cos\alpha + 0.070257\sin\alpha, \qquad
   \cos\theta_s = \sin\phi\sin\delta + \cos\phi\cos\delta\cos h,

and, with the correction ``sky``, its azimuth from North, clockwise,
:math:`A = \pi + \operatorname{atan2}(\sin h,\ \sin\phi\cos h - \cos\phi\tan\delta)`. No sun when
:math:`\theta_s \ge 90°` (the legacy: a sun below the horizon shines upwards).

**Diffuse fraction** (``diffuse_fraction``, Spitters et al. 1986): with the extraterrestrial irradiance
:math:`S_0\cos\theta_s`, :math:`S_0 = 1370\,(1 + 0.033\cos(2\pi(J - 4)/366))` W m⁻², the clearness
:math:`r = R_g / (S_0\cos\theta_s)` and :math:`R = 0.847 - 1.61\cos\theta_s + 1.04\cos^2\theta_s`,
:math:`K = (1.47 - R)/1.66`:

.. math::

   \frac{R_d}{R_g} = \begin{cases} 1 & r \le 0.22 \\ 1 - 6.4\,(r - 0.22)^2 & r \le 0.35 \\
   1.47 - 1.66\,r & r \le K \\ R & \text{otherwise.} \end{cases}

**Hourly global radiation** (``hourly_global_radiation``, Kaplanis 2006), from the day's :math:`R_{g,d}`, with
:math:`a = \sin\phi\sin\delta`, :math:`b = \cos\phi\cos\delta`:

.. math::

   R_g(t) = \psi\,\left(a - b\cos\frac{2\pi t}{24}\right), \qquad
   \psi = \frac{\pi R_{g,d} / 86400}{a \arccos(-a/b) + b\sqrt{1 - (a/b)^2}}.


Direct light (``projection``, ``direct``)
=========================================

Each source's beam projects the scene orthographically on a screen of :math:`T \times T` pixels
(``projection.legacy_screen``, ``rasterise``); a pixel belongs to the nearest triangle along the beam. A pixel's
area, counted on the horizontal (the sources' energies are horizontal irradiances), is

.. math::

   A_\text{pix} = \frac{d_u\, d_v}{(T - 1)^2 \cos\theta_s}

with the correction ``pixel_area`` (the pixels' pitch is :math:`d/(T-1)`; the legacy divides by :math:`T^2`),
:math:`d_u, d_v` the screen's extents and :math:`\theta_s` the beam's zenith angle. A face's sunlit area is
:math:`A_{f,s} = n_{f,s}\, A_\text{pix}`, its pixels' count (``projected_areas``; the face met by the beam,
:math:`\mathbf{u}_s \cdot \mathbf{n} < 0` for the upper face). The first order (``direct.accumulate``):

.. math::

   E^0_f = \sum_s E_s \frac{A_{f,s}}{S_t}, \qquad
   B^0_f = \sum_s E_s\,\rho_f \frac{A_{f,s}}{S_t} + \sum_s E_s\,\tau_{\bar f} \frac{A_{\bar f,s}}{S_t},

:math:`\bar f` the other face of a leaf (the light it transmits).

**Infinite canopy** (``periodic``): a pattern :math:`[x_0, x_1] \times [y_0, y_1]` repeats the scene. Each
triangle is moved into it by whole periods (``periodise``); the image is tiled by the neighbouring patterns' images
(``tile``), whose pixels in front shade the pattern's. With the corrections ``border_copies`` and ``tiling``, a
triangle across a border is drawn with a copy on each side it crosses, the neighbours' images are placed by the
lattice's exact vectors with their depth offset :math:`\mathbf{u}\cdot\mathbf{L}`, and a pixel counts only when its
visible point lies in the pattern's half-open cell: every point of the periodic canopy is counted once, and the
light entering the pattern, :math:`E_s (x_1 - x_0)(y_1 - y_0)`, is conserved.


Radiosity (``nearfield``, ``solver``, ``radiosity``)
=====================================================

Each receiver sees its neighbours on the disc of its hemisphere (the Nusselt analogue: a direction's orthographic
projection on the receiver's plane, ``nearfield._project``): :math:`N \times N` pixels of side :math:`2/N` on the
unit disc, each of form factor

.. math::

   dF = \frac{(2/N)^2}{\pi} = \frac{4}{\pi N^2}

(``form_factor``). The nearest emitter face is kept per pixel; :math:`c^{+}_{ij}` and :math:`c^{-}_{ij}` count the
pixels of emitter face :math:`j` on receiver :math:`i`'s upper and lower discs, :math:`F^\pm_{ij} = c^\pm_{ij}\,dF`
(the legacy stores :math:`c^+ - c^-`; the correction ``face_counts`` keeps both). The radiosities solve
(``solver.product``)

.. math::

   \text{opaque:}\quad & B_i = B^0_i + \rho_i \textstyle\sum_j F^+_{ij} B_j \\
   \text{leaf, upper:}\quad & B_i^\text{sup} = B^{0,\text{sup}}_i + \rho^\text{sup}_i \textstyle\sum_j F^+_{ij} B_j
       + \tau^\text{inf}_i \textstyle\sum_j F^-_{ij} B_j \\
   \text{leaf, lower:}\quad & B_i^\text{inf} = B^{0,\text{inf}}_i + \tau^\text{sup}_i \textstyle\sum_j F^+_{ij} B_j
       + \rho^\text{inf}_i \textstyle\sum_j F^-_{ij} B_j

a linear system :math:`(I - \mathcal{R} F)\,B = B^0`, solved by meschach's MGCR as canestrad (``solver.mgcr``), or
by scipy's GMRES (``linear_solver="gmres"``). In the whole scene (``radiosity.run``), every triangle is in every
receiver's sphere.


Nested radiosity (``nested``)
=============================

In an infinite canopy, a receiver exchanges radiosity with the emitters within a sphere of diameter :math:`D`
(their periodic images included); beyond, the canopy is a turbid medium of horizontal layers, the far field. A
pixel of the disc where no emitter lies sees the far field where its direction :math:`\mathbf{d}` leaves the
sphere, at height :math:`z = z_i + \tfrac{D}{2} d_z` (``nearfield._environment``): the profile's downward flux
:math:`E^-(z)` when looking up, its upward :math:`E^+(z)` when looking down, interpolated between the levels
:math:`k, k+1` as :math:`(1 - f)\,E_k + f\,E_{k+1}` (the correction ``profile_interpolation``; the legacy inverts
the weights) and clamped to the profile (``profile_bounds``). The far field's light on face :math:`i`,

.. math::

   B^\text{far}_i = \rho_i \sum_{\text{empty pixels}} dF\,E^\mp(z) \quad (\text{and } \tau \text{ across a leaf}),

is added to the first order: :math:`(I - \mathcal{R} F)\,B = B^0 + B^\text{far}`. With :math:`D = 0` (pure SAIL,
``sail.far_field``), no near field: each face receives the profile at its centre's height, by Ross's formula for a
face of normal zenith :math:`\theta_n`,

.. math::

   E^\text{sup} = \tfrac{1}{2}(1 + |\cos\theta_n|)\,E^- + \tfrac{1}{2}(1 - |\cos\theta_n|)\,E^+

(for an upward face; the lower face the other way round), and :math:`B = B^0 + \rho E^\text{sup} + \tau E^\text{inf}`.


The turbid medium (``sail``)
============================

**Layers** (``sail.layers``, s2v): the canopy below the height :math:`H` in :math:`n` layers of thickness
:math:`H/n`; per layer :math:`k`, its leaf area index

.. math::

   L_k = \frac{1}{(x_1 - x_0)(y_1 - y_0)} \sum_t w_t\, S_t\, \phi_{t,k},

:math:`\phi_{t,k}` the share of triangle :math:`t` in the layer (the correction ``exact_layers``: the exact share of a
planar triangle below a height, piecewise quadratic; s2v subdivides it), :math:`w_t = 1` for a leaf, :math:`1/2` for
a stem (its silhouette, kept from s2v); the frequencies of its normals' zenith angles in 18 classes of 5°; the
area-weighted mean :math:`\rho`, :math:`\tau` of the layer.

**SAIL** (``sail.profiles``, mcsail; Verhoef 1984): in each layer, the direct flux :math:`E_s` and the diffuse
fluxes :math:`E^-` (down) and :math:`E^+` (up) follow, :math:`x` the cumulated leaf area index from the top,

.. math::

   \frac{dE_s}{dx} = -k\,E_s, \qquad
   \frac{dE^-}{dx} = -a\,E^- + \sigma E^+ + s'\,E_s, \qquad
   \frac{dE^+}{dx} = a\,E^+ - \sigma E^- - s\,E_s,

with Verhoef's coefficients from the layer's leaf angle distribution, :math:`\rho`, :math:`\tau` and the source's
zenith angle (``sail._coefficients``: ``ks``, ``att``, ``sig``, ``sb``, ``sf``): :math:`k` the extinction of the
direct flux, :math:`a` the attenuation and :math:`\sigma` the backscatter of the diffuse fluxes, :math:`s`,
:math:`s'` the direct flux scattered backward and forward; each layer's solution gives its reflectances and
transmittances (``sail._layer_matrices``). The layers are combined by the adding method (``sail._adding``) over the soil of reflectance
:math:`\rho_\text{soil}`, which gives the profile :math:`E^-_k, E^+_k` at each level.


Light budgets (``budgets``)
===========================

The legacy recovers each face's irradiance from its radiosity (``triangle_budgets``): an opaque face
:math:`E = B/\rho`; a leaf's two faces from

.. math::

   B^\text{sup} = \rho^\text{sup} E^\text{sup} + \tau^\text{inf} E^\text{inf}, \qquad
   B^\text{inf} = \rho^\text{inf} E^\text{inf} + \tau^\text{sup} E^\text{sup},

singular when :math:`\rho^\text{sup}\rho^\text{inf} = \tau^\text{sup}\tau^\text{inf}`. With the correction
``face_irradiance`` (``face_budgets``), each face's irradiance is accumulated with the light: the first order's
:math:`E^0_f`, the near field's :math:`\sum_j F^\pm_{ij} B_j` and the far field's. Either way,

.. math::

   E_i = E^\text{sup} + E^\text{inf}, \qquad
   E_{abs} = (1 - \rho^\text{sup} - \tau^\text{sup})\,E^\text{sup} + (1 - \rho^\text{inf} - \tau^\text{inf})\,E^\text{inf}

(an opaque triangle: :math:`(1 - \rho)\,E`).


The metafspm component (``component``)
======================================

``CaribuLight`` lights the canopy under a sky of unit horizontal energy, so that each element :math:`e` (its
triangles :math:`t \in e` above the soil) gets its light relative to the incident PAR,

.. math::

   E_{rel,e} = \frac{\sum_{t \in e} E_{abs,t}\, S_t}{\sum_{t \in e} S_t}, \qquad
   \text{PARa}_e = E_{rel,e}\,\text{PAR}_i

(``_relative_light``, ``_absorbed_PAR``); with the sun on, the sky shares the unit energy between the diffuse sky,
:math:`R_d/R_g`, and the sun, :math:`1 - R_d/R_g`.
