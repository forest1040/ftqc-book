"""
QPD backend for the hybrid FTQC runtime.

This module turns each T/Tdag channel into a signed Clifford-channel ensemble.
Each expanded branch contains only Clifford gates (I/S/SDG/Z plus the original
Clifford gates), so the physical-QEC part can be routed through Stim shells and
the S/SDG term of the T-QPD decomposition can be routed through the LS-S shell
inside hybrid_ftqc_runtime.HybridFTQCRuntime.

Two execution styles are provided:

* QPDBackend.run(...) delegates to HybridFTQCRuntime with
  nonclifford_mode="qpd".  This is the noisy/trajectory backend that samples
  Stim QEC chunks when stim/pymatching are installed.
* QPDBackend.run_exact_noiseless(...) enumerates all 3**T Clifford branches and
  exactly reconstructs the final Z-basis distribution.  This is intended as a
  small-T validation path and has no Stim dependency.

Important boundary: a QPD backend does not put a T gate into one Stim circuit.
It runs a signed ensemble of Clifford-only branches and reconstructs observables
by classical post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

from ftqc.hybrid_ftqc_runtime import (
    Gate,
    HybridFTQCRuntime,
    NumpyStateBackend,
    T_QPD_TERMS,
    TDG_QPD_TERMS,
    infer_n_logical,
    normalize_gate_name,
    parse_qulacs_or_gate_list,
)


@dataclass(frozen=True)
class QPDExpandedBranch:
    """One Clifford-only branch of the QPD expansion."""

    coefficient: float
    gates: Tuple[Gate, ...]
    term_path: Tuple[str, ...]

    @property
    def sign(self) -> int:
        return 1 if self.coefficient >= 0 else -1

    @property
    def abs_coefficient(self) -> float:
        return abs(self.coefficient)


@dataclass
class QPDExactResult:
    """Exact branch-enumerated QPD reconstruction result."""

    n_logical: int
    t_count: int
    num_branches: int
    coefficient_l1: float
    coefficient_sum: float
    distribution: Dict[Tuple[int, ...], float]
    branches: List[QPDExpandedBranch] = field(default_factory=list)

    @property
    def branch_count(self) -> int:
        return self.num_branches

    @property
    def gamma(self) -> float:
        """Total QPD 1-norm for the full circuit."""
        return self.coefficient_l1

    @property
    def variance_overhead(self) -> float:
        """Monte Carlo variance overhead scale gamma**2."""
        return self.coefficient_l1 ** 2

    def qpd_distribution(self, normalize: bool = False) -> Dict[Tuple[int, ...], float]:
        return self.normalized_distribution() if normalize else self.distribution

    def normalized_distribution(self) -> Dict[Tuple[int, ...], float]:
        if abs(self.coefficient_sum) < 1e-15:
            return {k: float("nan") for k in sorted(self.distribution)}
        return {k: v / self.coefficient_sum for k, v in sorted(self.distribution.items())}

    def expectation_z(self, mask: Sequence[int], normalize: bool = False) -> float:
        dist = self.normalized_distribution() if normalize else self.distribution
        if not mask:
            return 1.0
        if all(x in (0, 1, False, True) for x in mask) and len(mask) == self.n_logical:
            qubits = [i for i, bit in enumerate(mask) if int(bit)]
        else:
            qubits = [int(q) for q in mask]
        exp = 0.0
        for outcome, prob in dist.items():
            parity = 0
            for q in qubits:
                parity ^= int(outcome[q])
            exp += float(prob) * (1.0 if parity == 0 else -1.0)
        return float(exp)

    def tvd_against(self, reference: Dict[Tuple[int, ...], float], normalize: bool = False) -> float:
        dist = self.normalized_distribution() if normalize else self.distribution
        keys = set(dist) | set(reference)
        return 0.5 * sum(abs(dist.get(k, 0.0) - reference.get(k, 0.0)) for k in keys)


def qpd_terms_for_gate(name: str) -> Tuple[Tuple[str, float], ...]:
    """Return the exact Clifford-channel QPD terms for T or Tdag."""
    name = normalize_gate_name(name)
    if name == "T":
        return T_QPD_TERMS
    if name == "T_DAG":
        return TDG_QPD_TERMS
    raise ValueError(f"No QPD decomposition registered for {name!r}")


def count_t_gates(gates: Sequence[Gate]) -> int:
    return sum(1 for name, _, _ in gates if normalize_gate_name(name) in ("T", "T_DAG"))


def expand_qpd_branches(qc_or_gates: Any, max_branches: Optional[int] = None) -> List[QPDExpandedBranch]:
    """Enumerate all Clifford-only QPD branches.

    T expands to I/S/Z with coefficients
        1/2, 1/sqrt(2), -(sqrt(2)-1)/2.
    Tdag expands to I/SDG/Z with the same coefficients.
    """
    gates = parse_qulacs_or_gate_list(qc_or_gates)
    t_count = count_t_gates(gates)
    n_branches = 3 ** t_count
    if max_branches is not None and n_branches > int(max_branches):
        raise ValueError(
            f"QPD exact expansion would create {n_branches} branches. "
            f"Increase max_branches or use QPDBackend.run(...) for Monte Carlo sampling."
        )

    branches: List[QPDExpandedBranch] = [QPDExpandedBranch(1.0, tuple(), tuple())]
    for name, a, b in gates:
        name = normalize_gate_name(name)
        if name in ("T", "T_DAG"):
            next_branches: List[QPDExpandedBranch] = []
            for branch in branches:
                for term_name, coeff in qpd_terms_for_gate(name):
                    if term_name == "I":
                        new_gates = branch.gates
                    else:
                        new_gates = branch.gates + ((term_name, a, None),)
                    next_branches.append(
                        QPDExpandedBranch(
                            coefficient=float(branch.coefficient) * float(coeff),
                            gates=new_gates,
                            term_path=branch.term_path + (term_name,),
                        )
                    )
            branches = next_branches
        else:
            branches = [
                QPDExpandedBranch(
                    coefficient=branch.coefficient,
                    gates=branch.gates + ((name, a, b),),
                    term_path=branch.term_path,
                )
                for branch in branches
            ]
    return branches


def _z_distribution_from_statevector(state: NumpyStateBackend) -> Dict[Tuple[int, ...], float]:
    probs = state.probabilities_z()
    n = state.n
    out: Dict[Tuple[int, ...], float] = {}
    for idx, p in enumerate(probs):
        if abs(float(p)) < 1e-15:
            continue
        key = tuple((idx >> (n - 1 - q)) & 1 for q in range(n))
        out[key] = out.get(key, 0.0) + float(p)
    return out


def branch_distribution(branch: QPDExpandedBranch, n_logical: int) -> Dict[Tuple[int, ...], float]:
    """Exact Z-basis distribution of one Clifford branch, using NumPy statevector."""
    state = NumpyStateBackend(n_logical)
    for name, a, b in branch.gates:
        if normalize_gate_name(name) == "I":
            continue
        state.apply_gate(name, a, b)
    return _z_distribution_from_statevector(state)


def exact_qpd_distribution(
    qc_or_gates: Any,
    n_logical: Optional[int] = None,
    max_branches: int = 100_000,
    keep_branches: bool = False,
) -> QPDExactResult:
    """Exactly reconstruct final Z-basis probabilities by branch enumeration.

    This routine has no QEC noise and no Stim dependency.  It is useful for
    testing the QPD decomposition before running noisy FT trajectories.
    """
    gates = parse_qulacs_or_gate_list(qc_or_gates)
    n = infer_n_logical(gates, n_logical)
    branches = expand_qpd_branches(gates, max_branches=max_branches)
    dist: Dict[Tuple[int, ...], float] = {}
    coeff_sum = 0.0
    coeff_l1 = 0.0
    for branch in branches:
        coeff_sum += float(branch.coefficient)
        coeff_l1 += abs(float(branch.coefficient))
        bdist = branch_distribution(branch, n)
        for key, p in bdist.items():
            dist[key] = dist.get(key, 0.0) + float(branch.coefficient) * float(p)

    # Clean tiny numerical noise without hiding meaningful negative estimates.
    for key in list(dist):
        if abs(dist[key]) < 1e-14:
            dist[key] = 0.0

    return QPDExactResult(
        n_logical=n,
        t_count=count_t_gates(gates),
        num_branches=len(branches),
        coefficient_l1=float(coeff_l1),
        coefficient_sum=float(coeff_sum),
        distribution={k: dist[k] for k in sorted(dist)},
        branches=branches if keep_branches else [],
    )


class QPDBackend:
    """Convenience backend wrapping QPD exact expansion and noisy runtime sampling."""

    exact_distribution = staticmethod(exact_qpd_distribution)
    expand_branches = staticmethod(expand_qpd_branches)

    def __init__(
        self,
        n_logical: Optional[int] = None,
        p_phys: float = 0.005,
        qec_mode: str = "optional",
        route_qpd_s_via_ls: bool = True,
        backend: str = "stim_auto",
        seed: int = 1,
        **runtime_kwargs: Any,
    ):
        self.n_logical = None if n_logical is None else int(n_logical)
        self.p_phys = float(p_phys)
        self.qec_mode = qec_mode
        self.route_qpd_s_via_ls = bool(route_qpd_s_via_ls)
        self.backend = backend
        self.seed = int(seed)
        self.runtime_kwargs = dict(runtime_kwargs)

    def expand(self, qc_or_gates: Any, max_branches: Optional[int] = None) -> List[QPDExpandedBranch]:
        return expand_qpd_branches(qc_or_gates, max_branches=max_branches)

    def run_exact_noiseless(
        self,
        qc_or_gates: Any,
        max_branches: int = 100_000,
        keep_branches: bool = False,
    ) -> QPDExactResult:
        gates = parse_qulacs_or_gate_list(qc_or_gates)
        n = infer_n_logical(gates, self.n_logical)
        return exact_qpd_distribution(gates, n_logical=n, max_branches=max_branches, keep_branches=keep_branches)

    def run(
        self,
        qc_or_gates: Any,
        shots: int = 1000,
        seed: Optional[int] = None,
        sample_final_z: bool = True,
    ):
        """Run the noisy trajectory QPD backend through HybridFTQCRuntime.

        This path is where Stim physical-QEC chunks and the LS-S shell are used
        when the relevant dependencies are available.  The returned object is a
        HybridRunResult with qpd_distribution()/qpd_expectation_z() helpers.
        """
        gates = parse_qulacs_or_gate_list(qc_or_gates)
        n = infer_n_logical(gates, self.n_logical)
        kwargs = dict(self.runtime_kwargs)
        kwargs.update(
            p_phys=self.p_phys,
            backend=self.backend,
            qec_mode=self.qec_mode,
            nonclifford_mode="qpd",
            route_qpd_s_via_ls=self.route_qpd_s_via_ls,
            seed=self.seed,
        )
        rt = HybridFTQCRuntime(n, **kwargs)
        return rt.run(gates, shots=shots, seed=self.seed if seed is None else seed, sample_final_z=sample_final_z)


def run_qpd_backend(qc_or_gates: Any, n_logical: Optional[int] = None, shots: int = 1000, **kwargs: Any):
    """One-line helper for noisy QPD backend execution."""
    backend = QPDBackend(n_logical=n_logical, **kwargs)
    return backend.run(qc_or_gates, shots=shots)


def demo() -> None:
    gates = [("H", 0, None), ("T", 0, None), ("H", 0, None)]
    qb = QPDBackend(n_logical=1, qec_mode="optional", route_qpd_s_via_ls=True, seed=5)

    exact = qb.run_exact_noiseless(gates, keep_branches=True)
    print("# QPD exact branch enumeration")
    print("t_count:", exact.t_count)
    print("num_branches:", exact.num_branches)
    print("coefficient_l1:", f"{exact.coefficient_l1:.6f}")
    print("coefficient_sum:", f"{exact.coefficient_sum:.6f}")
    print("distribution:", {''.join(map(str, k)): round(v, 6) for k, v in exact.distribution.items()})
    print("branches:")
    for branch in exact.branches:
        print(f"  coeff={branch.coefficient:+.6f}, terms={branch.term_path}, gates={branch.gates}")

    result = qb.run(gates, shots=1000, seed=7)
    print("\n# QPD noisy/runtime sampling")
    for line in result.summary_lines():
        print(line)
    for report in result.qpd_reports:
        print(report)


if __name__ == "__main__":
    demo()
