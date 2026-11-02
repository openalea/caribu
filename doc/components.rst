==========
Components
==========

The metafspm components and the environment model, rendered from their declarations (their variables by kind,
their schedule): the light (``CaribuLight``), its environment model on the plants' union (``CaribuLightModel``) and
the plants' optics (``PlantOptics``, a component of a plant model, read by the light). :doc:`equations` gives the
physics, :doc:`performance` the settings' effect on speed.

.. metafspm-component:: openalea.caribu.component.CaribuLight

.. metafspm-component:: openalea.caribu.component.CaribuLightModel

.. metafspm-component:: openalea.caribu.optics.PlantOptics
