"""Ideal non-Clifford sampling checks against the analytic H-T-H result."""
import math

import pytest

pytest.importorskip("tsim")

from ftqc.ftqc_tsim_backend import tsim_logical_distribution


def test_ideal_hth_distribution():
    p_zero = (2 + math.sqrt(2)) / 4
    distribution = tsim_logical_distribution(
        [("H", 0, None), ("T", 0, None), ("H", 0, None)],
        1,
        shots=6000,
        seed=17,
    )
    assert abs(distribution.get((0,), 0.0) - p_zero) < 0.04


def test_ideal_bell_correlations_survive_t_and_cnot():
    distribution = tsim_logical_distribution(
        [("H", 0, None), ("T", 0, None), ("CNOT", 0, 1)],
        2,
        shots=6000,
        seed=23,
    )
    assert distribution.get((0, 0), 0) + distribution.get((1, 1), 0) > 0.97
    assert abs(distribution.get((0, 0), 0) - 0.5) < 0.04
