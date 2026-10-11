"""
test_boundaries.py - Stage 03 Boundary Isolation & Distance Guard Tests
Ensures later-stage and forbidden modules cannot be imported, and d != 3 is strictly rejected.
"""
import pytest
from ftqc.ftqc_tsim_backend import validate_distance_d3, TsimStateBackend

def test_forbidden_later_modules():
    with pytest.raises(ImportError):
        import ftqc.core.surface_code_d  # Stage 05 module

    with pytest.raises(ImportError):
        import ftqc.runtime.symplectic_frame  # Stage 04 module

def test_distance_d3_guard():
    # d=3 is valid
    validate_distance_d3(3)
    backend = TsimStateBackend(n=2, distance=3)
    assert backend.d == 3

    # d != 3 must raise ValueError
    with pytest.raises(ValueError, match="Stage 03 only supports distance d=3 fixed"):
        validate_distance_d3(5)

    with pytest.raises(ValueError, match="Stage 03 only supports distance d=3 fixed"):
        TsimStateBackend(n=2, distance=5)
