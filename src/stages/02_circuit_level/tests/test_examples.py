"""
test_examples.py - Stage 02 Example Execution Tests
"""
import importlib
import pathlib
import sys
import pytest

STAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

EXAMPLES = [
    "examples.ch14_ancilla_noise",
    "examples.ch15_hook_error",
    "examples.ch16_circuit_merge",
    "examples.ch17_ls_cnot",
    "examples.ch18_shell_api",
    "examples.ch19_circuit_integration",
]

@pytest.mark.parametrize("mod_name", EXAMPLES)
def test_example_execution(mod_name):
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, "main")
    mod.main()
