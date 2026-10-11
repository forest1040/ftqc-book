"""Exact code and teleportation checks for the Stage 03 magic-state model."""
from ftqc import t_gate


def test_15_to_1_code_has_distance_three_and_35_weight_three_logicals():
    hx, hz = t_gate.build_15to1()
    css, logical_qubits, counts = t_gate.code_properties(hx, hz)
    assert css
    assert logical_qubits == 1
    assert counts[1] == 0
    assert counts[2] == 0
    assert counts[3] == 35


def test_ideal_t_teleportation_matches_t_gate():
    assert t_gate.verify_t_teleportation() is True
