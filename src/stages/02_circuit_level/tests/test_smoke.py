"""
test_smoke.py - Stage 02 Core Functionality Smoke Tests
"""
import pytest
from ftqc.surface_patch import SurfacePatch
from ftqc.circuit_level_merge import CircuitLevelMerge
from ftqc.ls_cnot_ft import LSCnotFT
from ftqc.ls_handoff_shells import build_ls_s_shell
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def test_surface_patch_d3():
    sp = SurfacePatch(distance=3)
    assert len(sp.data) == 9

def test_circuit_level_merge():
    clm = CircuitLevelMerge(kind="ZZ")
    assert clm.kind == "ZZ"

def test_ls_cnot_ft():
    cnot = LSCnotFT()
    assert cnot is not None

def test_build_ls_s_shell():
    shell = build_ls_s_shell(p=0.0, merge_rounds=2)
    assert shell is not None

def test_hybrid_runtime_stage02():
    runtime = HybridFTQCRuntime(n_logical=2)
    assert runtime is not None
