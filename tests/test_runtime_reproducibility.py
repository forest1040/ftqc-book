"""Runtime seed, factory restoration, and report generation contracts."""
import pytest

from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime, make_backend
from ftqc.viz.ftqc_run_report import build_run_report_html
from ftqc.viz.ftqc_run_visualizer import event_circuits


def test_runtime_repeats_for_same_seed_and_reports_all_shots():
    runtime = HybridFTQCRuntime(1, p_phys=0.0, qec_mode="off", seed=31)
    gates = [("H", 0, None), ("T", 0, None), ("H", 0, None)]
    first = runtime.run(gates, shots=600, seed=91)
    second = runtime.run(gates, shots=600, seed=91)
    assert first.measurement_counts == second.measurement_counts
    assert sum(first.measurement_counts.values()) == 600


def test_report_builder_returns_complete_html():
    html = build_run_report_html(
        [("H", 0, None), ("T", 0, None)], n_logical=1, shots=32
    )
    assert "<!doctype html>" in html.lower()
    assert "H" in html and "T" in html


def test_report_rejects_unimplemented_t_qec_modes():
    with pytest.raises(ValueError, match="t_qec_mode"):
        build_run_report_html(
            [("T", 0, None)], n_logical=1, t_qec_mode="lattice_surgery", shots=32
        )


def test_bare_s_report_uses_the_runtime_memory_handoff():
    runtime = HybridFTQCRuntime(1, p_phys=0.0, qec_mode="off")
    event = event_circuits(runtime, [("S", 0, None)])[0]
    # Runtime handles bare S with a zero-gate generic memory shell.
    assert event["kind"] == "special:S (memory shell)"
    assert event["ls_builder"] is None


def test_tsim_wrapper_restores_runtime_and_backend_factory():
    pytest = __import__("pytest")
    pytest.importorskip("tsim")
    import ftqc.hybrid_ftqc_runtime as runtime_module
    from ftqc.ftqc_tsim_backend import run_tsim

    runtime = HybridFTQCRuntime(1, qec_mode="off", backend="numpy")
    original_factory = runtime_module.make_backend
    original_state = (runtime.backend_name, runtime.nonclifford_mode)
    result = run_tsim(runtime, [("H", 0, None), ("T", 0, None)], shots=32, seed=5)
    assert result.backend == "tsim"
    assert runtime_module.make_backend is original_factory
    assert (runtime.backend_name, runtime.nonclifford_mode) == original_state
    assert make_backend(1, "numpy").name == "numpy"
