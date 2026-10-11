"""Scientific invariants for the Stage 03 lattice-surgery gadgets."""
import numpy as np

from ftqc.ls_cnot_ft import LSCnotFT
from ftqc.ls_handoff_shells import (
    ShellCircuit,
    build_ls_cnot_shell,
    build_ls_s_shell,
    build_ls_t_injection_shell,
)


def _gauge_count(circuit, shots=128):
    det, _ = circuit.compile_detector_sampler().sample(
        shots, separate_observables=True
    )
    return int(det.any(axis=0).sum())


def test_ls_cnot_is_deterministic_and_distance_three():
    clean, _, _ = LSCnotFT().build(p=0.0, noise=False)
    noisy, _, _ = LSCnotFT().build(p=0.005, noise=True)
    assert _gauge_count(clean) == 0
    assert len(noisy.shortest_graphlike_error()) == 3


def test_ls_cnot_computational_truth_table():
    gadget = LSCnotFT()
    for control in (0, 1):
        for target in (0, 1):
            circuit, data_rec, joint_rec = gadget.build(
                control_one=bool(control), target_one=bool(target), noise=False
            )
            samples = circuit.compile_sampler().sample(32).astype(np.uint8)
            measured_control = samples[:, [data_rec[q] for q in gadget.ZL_C]].sum(axis=1) % 2
            measured_target = (
                samples[:, [data_rec[q] for q in gadget.ZL_T]].sum(axis=1)
                + samples[:, [data_rec[q] for q in gadget.ZL_A]].sum(axis=1)
                + samples[:, list(joint_rec)].sum(axis=1)
            ) % 2
            assert np.all(measured_control == control)
            assert np.all(measured_target == (control ^ target))


def test_handoff_shells_are_gauge_free_and_distance_three():
    builders = (build_ls_s_shell, build_ls_cnot_shell, build_ls_t_injection_shell)
    for build in builders:
        clean = build(p=0.0, noise=False)
        noisy = build(p=0.005, noise=True)
        assert isinstance(clean, ShellCircuit)
        assert _gauge_count(clean.circuit) == 0
        assert len(noisy.circuit.shortest_graphlike_error()) == 3
