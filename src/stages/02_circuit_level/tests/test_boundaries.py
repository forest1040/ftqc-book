"""
test_boundaries.py - Stage 02 Boundary Isolation Tests
Ensures later-stage and forbidden modules cannot be imported in Stage 02.
"""
import pytest

def test_forbidden_later_modules():
    with pytest.raises(ImportError):
        import ftqc.core.surface_code_d  # Stage 05 module

    with pytest.raises(ImportError):
        import ftqc.runtime.symplectic_frame  # Stage 04 module
