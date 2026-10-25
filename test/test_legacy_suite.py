"""The legacy's own tests run on the port: each legacy test module loaded with its algorithms (raycasting,
x_raycasting, CaribuScene) rebound to the port's, its tests run when every algorithm they call is ported."""
import importlib.util
import sys
from pathlib import Path

import pytest

from openalea.caribu import api

LEGACY_TESTS = Path(__file__).parent / "legacy"
PORTED = {"raycasting": api.raycasting, "x_raycasting": api.x_raycasting, "CaribuScene": api.CaribuScene,
          "green_leaf_PAR": api.GREEN_LEAF_PAR}
NOT_YET = {"radiosity", "mixed_radiosity", "x_radiosity", "x_mixed_radiosity", "light_sources"}
# legacy tests of what stays in the legacy (its display), and those running scattering through CaribuScene.run
LEFT = {("test_caribuscene", "test_display")}
SCATTERING = {("test_caribuscene", "test_run_monochrome"), ("test_caribuscene", "test_run_polychrome")}
MODULES = ["test_null_triangle", "test_simple_triangle", "test_triangle_faces", "test_material", "test_domain",
           "test_caribu", "test_multi_band", "test_caribuscene", "test_caribu_sensors"]


def _load(name):
    sys.path.insert(0, str(LEGACY_TESTS))
    try:
        spec = importlib.util.spec_from_file_location("legacy_suite_" + name, LEGACY_TESTS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(LEGACY_TESTS))
    for symbol, port in PORTED.items():
        if hasattr(module, symbol):
            setattr(module, symbol, port)
    return module


def _cases():
    for name in MODULES:
        module = _load(name)
        for test_name in sorted(vars(module)):
            if not test_name.startswith("test"):
                continue
            calls = set(getattr(module, test_name).__code__.co_names)
            reason = None
            if calls & NOT_YET:
                reason = "uses " + ", ".join(sorted(calls & NOT_YET)) + ", not yet ported"
            elif (name, test_name) in LEFT:
                reason = "stays in the legacy"
            elif (name, test_name) in SCATTERING:
                reason = "runs scattering, not yet ported"
            yield pytest.param(module, test_name, id=f"{name}::{test_name}",
                               marks=[pytest.mark.skip(reason=reason)] if reason else [])


@pytest.mark.parametrize("module, test_name", list(_cases()))
def test_legacy_case(module, test_name):
    getattr(module, test_name)()
