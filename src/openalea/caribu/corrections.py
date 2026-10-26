"""
The corrections of the legacy's behaviours (devplan, C10): each a name, applied when given in a model's
*corrections*; none by default in the legacy API, all of them ("recommended") in the metafspm component. With
precision "legacy", the legacy's printed digits, none applies: it stays the bit-for-bit reference.
"""
NAMES = {
    "border_copies": "a triangle across the pattern's borders copied on each side it crosses, corners included",
    "sensor_copies": "sensors copied across the borders as triangles, each copy's pixels counted once",
    "pixel_area": "a pixel's area the pitch's, du dv / (T - 1)^2",
    "upward_beams": "a source whose beam goes up or along the ground lights nothing",
    "periodise": "the periodised scene in float32 without its text rounding, lifted to exactly z >= 0",
    "triangle_area": "a triangle's area 0.5 |AB x AC|",
    "voxel_floor": "the receiver's voxel floor((G - O) / side)",
    "face_counts": "an emitter's two faces counted apart",
    "projected_face": "each pixel's emitter face the one of the copy projected there",
    "profile_interpolation": "the far field's levels interpolated with weights (1 - f, f)",
    "profile_bounds": "the far field above the top level the top's, below the soil the soil's",
    "band_scaling": "a reused far field never scaled by the last source's energy",
    "exact_layers": "the layers' leaf areas by clipping the triangles at the layers' planes",
    "no_soil": "a soil reflectance of -1 (none given) taken as 0",
    "face_irradiance": "each face's irradiance accumulated with its radiosity, Eabs from it",
    "sky": "one azimuth convention, the turtle's emission, the sun's position and night",
}
RECOMMENDED = frozenset(NAMES)


def resolve(corrections, precision="full") -> frozenset:
    """The set of correction names of *corrections* ("recommended", None or () for none, or names), checked."""
    if corrections is None:
        return frozenset()
    if isinstance(corrections, str):
        corrections = RECOMMENDED if corrections == "recommended" else (corrections,)
    resolved = frozenset(corrections)
    unknown = sorted(resolved - RECOMMENDED)
    if unknown:
        raise ValueError(f"unknown corrections {unknown}: {sorted(NAMES)} or 'recommended'")
    if resolved and precision == "legacy":
        raise ValueError("corrections apply with precision 'full': precision 'legacy' is the legacy's numbers")
    return resolved
