"""
test_smoke.py - Stage 03 Core Functionality Smoke Tests
"""
import pytest
from ftqc.surface_patch import SurfacePatch
from ftqc.ftqc_tsim_backend import TsimStateBackend
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def test_surface_patch_d3():
    sp = SurfacePatch(distance=3)
    assert len(sp.data) == 9

def test_tsim_backend_d3():
    backend = TsimStateBackend(n=2, distance=3)
    assert backend.name == "tsim"
    assert backend.d == 3

def test_hybrid_runtime_stage03():
    runtime = HybridFTQCRuntime(n_logical=2)
    assert runtime is not None
