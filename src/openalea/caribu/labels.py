"""
Caribu's labels: a 12-digit number per triangle, `species · 1e11 + plant · 1e6 + leaf · 1e3 + element` (species: the
optical species, numbered from 1 in the optical properties' order, 0 for the soil; plant on 5 digits, leaf and element
on 3). A triangle is translucent (a leaf, two faces) when its leaf field is not 0, else opaque (a stem, the soil: one
face). Only the species and the opacity matter to the physics; plant and element are carried along.
"""
import numpy as np

SPECIES_DIVISOR = 10 ** 11


def encode_labels(species, leaf=0, plant=1, element=0) -> np.ndarray:
    """The labels (int64) of triangles of optical *species* (1-based; 0: soil), translucent when *leaf* is not 0;
    *plant* (< 100000), *leaf* and *element* (< 1000) per triangle or for all."""
    species, leaf, plant, element = (np.asarray(value, dtype=np.int64) for value in (species, leaf, plant, element))
    if np.any(plant > 99999) or np.any(leaf > 999) or np.any(element > 999):
        raise ValueError("labels: plant < 100000, leaf and element < 1000")
    return species * SPECIES_DIVISOR + plant * 10 ** 6 + leaf * 10 ** 3 + element


def species_of(labels) -> np.ndarray:
    """Each label's optical species (0: soil)."""
    return np.asarray(labels, dtype=np.int64) // SPECIES_DIVISOR


def is_translucent(labels) -> np.ndarray:
    """Whether each label is a translucent leaf (its leaf field is not 0), as canestrad decides it."""
    return (np.asarray(labels, dtype=np.int64) // 1000) % 1000 != 0
