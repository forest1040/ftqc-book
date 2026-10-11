"""
test_smoke.py - Stage 01 Core Functionality Smoke Tests
"""
import pytest
from ftqc.surface_patch import SurfacePatch
from ftqc.clifford_ft_full import CliffordFTCompiler
from ftqc.t_gate import build_15to1, distill
from ftqc.qpd_backend import QPDBackend
from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

def test_surface_patch_d3():
    sp = SurfacePatch(distance=3)
    assert len(sp.data) == 9
    assert len(sp.stabilizers) == 8

def test_clifford_compiler():
    compiler = CliffordFTCompiler(n_logical=2)
    assert compiler is not None

def test_t_gate_distill():
    HX, HZ = build_15to1()
    acc, err = distill(HX, HZ, p=0.001, shots=50)
    assert 0.0 <= acc <= 1.0

def test_qpd_backend():
    qpd = QPDBackend()
    assert qpd is not None

def test_hybrid_runtime():
    runtime = HybridFTQCRuntime(n_logical=2)
    assert runtime is not None
