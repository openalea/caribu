"""
The optics of a plant model's elements in the PAR, for a light model (CaribuLight): an optical component of the plant
model, its parameters the plant's (each plant's scenario may change them)::

    class Wheat(CompositeModel):
        parts = {"shoot": ShootArchitecture, "optics": PlantOptics}

Each element's reflectance and transmittance by its kind, its MTG label among leaf_labels (two faces, translucent) or
stem_labels (opaque): Wheat-BRIDGES' leaves ρ 0.10, τ 0.05, stems ρ 0.10; another element (a root) has none, so the
light leaves it out. They reach CaribuLight (its inputs PAR_reflectance, PAR_transmittance) through the scene's
links, identity ones by default. The elements' lit areas come back from CaribuLight (the input lit_area): the weight of their optics' means
in the scene's summaries, whatever else the plant model reads from the light.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import (FunctionalComponent, input_variable, parameter, setting,
                                                   state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import state

ELEMENT = dict(scale=scales.SubOrgan)


@dataclass
class PlantOptics(FunctionalComponent):
    """A plant model's elements' optics in the PAR, by kind (leaf, stem), per plant."""
    PAR_reflectance: float = state_variable(unit="adim", description="The element's reflectance in the PAR (NaN: none, "
                                            "not lit)", kind="intensive", weight="lit_area", default=np.nan, **ELEMENT)
    PAR_transmittance: float = state_variable(unit="adim", description="The element's transmittance in the PAR (NaN: "
                                              "opaque)", kind="intensive", weight="lit_area", default=np.nan, **ELEMENT)
    lit_area: float = input_variable(unit="m2", description="The element's lit area (CaribuLight's): the weight of the "
                                     "optics' means in the scene's summaries", by="CaribuLight", default=0.,
                                     kind="extensive", **ELEMENT)
    leaf_reflectance: float = parameter(unit="adim", description="A leaf element's reflectance in the PAR",
                                        by="Wheat-BRIDGES", default=0.10, min_value=0., max_value=1.)
    leaf_transmittance: float = parameter(unit="adim", description="A leaf element's transmittance in the PAR",
                                          by="Wheat-BRIDGES", default=0.05, min_value=0., max_value=1.)
    stem_reflectance: float = parameter(unit="adim", description="A stem element's reflectance in the PAR (opaque)",
                                        by="Wheat-BRIDGES", default=0.10, min_value=0., max_value=1.)
    leaf_labels: tuple = setting(description="The MTG labels of the leaf elements (two faces, translucent)",
                                 default=("LeafElement", "LeafElement1"))
    stem_labels: tuple = setting(description="The MTG labels of the stem elements (opaque)", default=("StemElement",))

    @state(include_inactive=True)
    def _optics(self, leaf_reflectance, leaf_transmittance, stem_reflectance, leaf_labels, stem_labels) -> dict:
        """Each element's optics by its kind; NaN for another kind."""
        labels = np.asarray(self.data_structure.labels())
        leaf, stem = np.isin(labels, list(leaf_labels)), np.isin(labels, list(stem_labels))
        return {"PAR_reflectance": np.where(leaf, leaf_reflectance, np.where(stem, stem_reflectance, np.nan)),
                "PAR_transmittance": np.where(leaf, leaf_transmittance, np.nan)}
