"""
test_examples.py - Stage 01 Example Execution Tests
"""
import importlib
import pathlib
import sys
import pytest

STAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

EXAMPLES = [
    "examples.ch06_surface_patch",
    "examples.ch07_mpp_syndrome",
    "examples.ch08_lattice_surgery",
    "examples.ch09_clifford_ft",
    "examples.ch10_t_gate",
    "examples.ch11_qpd",
    "examples.ch12_minimal_runtime",
    "examples.ch13_mpp_integration",
]

@pytest.mark.parametrize("mod_name", EXAMPLES)
def test_example_execution(mod_name):
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, "main")
    mod.main()
