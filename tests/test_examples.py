"""
test_examples.py - Stage 03 Example Execution Tests
"""
import importlib
import pathlib
import sys
import pytest

STAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

EXAMPLES = [
    "examples.ch20_d3_architecture",
    "examples.ch21_core7_integration",
    "examples.ch22_event_scheduling",
    "examples.ch23_backend_switch",
    "examples.ch24_visualization_reports",
    "examples.ch25_d3_benchmark",
]

@pytest.mark.parametrize("mod_name", EXAMPLES)
def test_example_execution(mod_name):
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, "main")
    mod.main()
