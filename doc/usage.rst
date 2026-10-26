=====================
The classic interface
=====================

Before the metafspm component, Caribu was used through ``CaribuScene``, which takes a scene (a PlantGL scene, an
MTG, a ``.can`` file or triangles), light sources and optical properties, and returns each triangle's light. This
interface is kept (``openalea.caribu.api``, on the ported models); the original package's notebooks below show it.
They are shown without their results.

.. nbgallery::
    notebook/caribu_crops
    notebook/Raycasting_methods_with_simple_object_python
