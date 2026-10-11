"""Chapter 25: verify the d=3 backend and an ideal non-Clifford circuit."""
import math

from ftqc.ftqc_tsim_backend import (
    TsimStateBackend,
    tsim_logical_distribution,
    validate_distance_d3,
)

def main():
    print("=== Chapter 25 Example: d=3 Integrated Benchmark ===")
    validate_distance_d3(3)
    backend = TsimStateBackend(n=2, distance=3)
    assert backend.name == "tsim" and backend.d == 3

    # H-T-H |0> has P(0)=(2+sqrt(2))/4. This exercises tsim sampling,
    # rather than stopping after backend construction.
    distribution = tsim_logical_distribution(
        [("H", 0, None), ("T", 0, None), ("H", 0, None)],
        n_logical=1,
        shots=6000,
        seed=25,
    )
    p_zero = distribution.get((0,), 0.0)
    expected = (2 + math.sqrt(2)) / 4
    assert abs(p_zero - expected) < 0.04, (p_zero, expected)
    print(f"d={backend.d}; H-T-H P(0)={p_zero:.3f} (expected {expected:.3f})")

if __name__ == '__main__':
    main()
