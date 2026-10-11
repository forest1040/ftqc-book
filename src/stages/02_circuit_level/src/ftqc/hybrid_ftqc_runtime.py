"""
Hybrid FTQC runtime: Stim physical-QEC chunks + Qulacs/state-vector T handoff.

This module adds the missing integration layer between the existing components:

* Clifford chunks ({X,Y,Z,H,CNOT}) are compiled into physical d=3 surface-code
  Stim circuits using clifford_ft_full.CliffordFTCompiler, sampled with detector
  sampling, and decoded with PyMatching/BP+OSD fallback.
* T gates can be handled in two modes. The default statevector mode applies
  T/Tdag at the logical state-vector layer and samples a magic-state fault model.
  The qpd mode replaces T/Tdag by signed Clifford-channel branches.  The S/SDG
  branch is routed through a detector-bearing LS-S shell, while I/Z branches
  stay on a generic Stim memory shell.  With qpd_backend='stim', the final
  branch circuit is sampled by Stim as a Clifford-only logical circuit instead
  of by Qulacs/NumPy.
* A Stim QEC shell is sampled at every handoff boundary, including immediately
  before/after each T event, so that non-Clifford events are not merely counted
  in an error budget. They are part of a trajectory-level simulation loop.

Important modelling boundary
----------------------------
Stim is a stabilizer simulator and cannot contain the non-stabilizer magic state
|A> itself. Therefore the non-Clifford state update is deliberately performed by
Qulacs/NumPy while Stim supplies the physical syndrome sampling and decoded
logical residuals around the event. This is a hybrid handoff model, not a single
all-Stim physical injection circuit.

The current Stim compiler exports Z-basis logical observable flips. Those are
applied to the logical state as decoded residual X_L faults. Optional mirrored
Z_L faults can be enabled for stress testing, but the physically grounded output
of the existing compiler is the X_L residual channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

Gate = Tuple[str, int, Optional[int]]

CLIFFORD_CHUNK_GATES = {"X", "Y", "Z", "H", "CNOT", "CX"}
ONE_QUBIT_GATES = {"X", "Y", "Z", "H", "S", "SDG", "S_DAG", "T", "TDG", "T_DAG"}

# Exact Clifford-channel quasi-probability decompositions.
# T rho T^dag = 1/2 I rho I + 1/sqrt(2) S rho S^dag
#                 - (sqrt(2)-1)/2 Z rho Z.
# T^dag is obtained by replacing S with S^dag.  Each branch is Clifford and
# therefore can be routed through Stim; finite-shot estimates use signed weights.
T_QPD_TERMS = (
    ("I", 0.5),
    ("S", 1.0 / math.sqrt(2.0)),
    ("Z", -(math.sqrt(2.0) - 1.0) / 2.0),
)
TDG_QPD_TERMS = (
    ("I", 0.5),
    ("SDG", 1.0 / math.sqrt(2.0)),
    ("Z", -(math.sqrt(2.0) - 1.0) / 2.0),
)


# ---------------------------------------------------------------------------
# Gate parsing
# ---------------------------------------------------------------------------

def normalize_gate_name(name: str) -> str:
    name = name.upper()
    aliases = {
        "CX": "CNOT",
        "S_DAG": "SDG",
        "S†": "SDG",
        "TDG": "T_DAG",
        "T†": "T_DAG",
    }
    return aliases.get(name, name)


def parse_qulacs_or_gate_list(qc_or_gates: Any) -> List[Gate]:
    """Return a normalized list of (name, a, b) logical gates.

    Accepts either:
      * a Qulacs QuantumCircuit-like object, or
      * an iterable of tuples such as ("H", 0, None), ("CNOT", 0, 1).
    """
    if isinstance(qc_or_gates, (list, tuple)):
        out: List[Gate] = []
        for item in qc_or_gates:
            if len(item) == 2:
                name, a = item
                b = None
            elif len(item) == 3:
                name, a, b = item
            else:
                raise ValueError(f"Gate tuple must have length 2 or 3, got {item!r}")
            out.append((normalize_gate_name(str(name)), int(a), None if b is None else int(b)))
        return out

    if not hasattr(qc_or_gates, "get_gate_count"):
        raise TypeError("Expected a Qulacs QuantumCircuit-like object or a list of gate tuples.")

    gates: List[Gate] = []
    for k in range(qc_or_gates.get_gate_count()):
        gate = qc_or_gates.get_gate(k)
        name = normalize_gate_name(gate.get_name())
        controls = gate.get_control_index_list()
        targets = gate.get_target_index_list()
        if name == "CNOT":
            if len(controls) != 1 or len(targets) != 1:
                raise ValueError(f"Unsupported CNOT shape at gate {k}: controls={controls}, targets={targets}")
            gates.append(("CNOT", int(controls[0]), int(targets[0])))
        elif name in ONE_QUBIT_GATES:
            if len(targets) != 1:
                raise ValueError(f"Unsupported one-qubit gate shape at gate {k}: {name}, targets={targets}")
            gates.append((name, int(targets[0]), None))
        else:
            raise ValueError(f"Unsupported gate {name!r}. Supported: Clifford+T subset.")
    return gates


def parse_universal(qc_or_gates: Any) -> List[Gate]:
    """Compatibility alias used by the demo/driver files."""
    return parse_qulacs_or_gate_list(qc_or_gates)


def infer_n_logical(gates: Sequence[Gate], n_logical: Optional[int] = None) -> int:
    if n_logical is not None:
        return int(n_logical)
    if not gates:
        return 1
    m = 0
    for _, a, b in gates:
        m = max(m, a if b is None else max(a, b))
    return m + 1


# ---------------------------------------------------------------------------
# State-vector backends
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X1 = np.array([[0, 1], [1, 0]], dtype=complex)
Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S1 = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG1 = np.array([[1, 0], [0, -1j]], dtype=complex)
T1 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG1 = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

_SINGLE = {
    "I": I2,
    "X": X1,
    "Y": Y1,
    "Z": Z1,
    "H": H1,
    "S": S1,
    "SDG": SDG1,
    "T": T1,
    "T_DAG": TDG1,
}


class NumpyStateBackend:
    """Small logical state-vector backend used as fallback and for tests."""

    name = "numpy"

    def __init__(self, n: int):
        self.n = int(n)
        self.state = np.zeros(2 ** self.n, dtype=complex)
        self.state[0] = 1.0

    def copy(self) -> "NumpyStateBackend":
        other = NumpyStateBackend(self.n)
        other.state = self.state.copy()
        return other

    def apply_1q(self, U: np.ndarray, q: int) -> None:
        st = self.state.reshape([2] * self.n)
        st = np.tensordot(U, st, axes=([1], [q]))
        self.state = np.moveaxis(st, 0, q).reshape(-1)

    def apply_cnot(self, c: int, t: int) -> None:
        st = self.state.reshape([2] * self.n)
        sl = [slice(None)] * self.n
        sl[c] = 1
        sub = st[tuple(sl)]
        sub = np.flip(sub, axis=(t if t < c else t - 1))
        st[tuple(sl)] = sub
        self.state = st.reshape(-1)

    def apply_gate(self, name: str, a: int, b: Optional[int] = None) -> None:
        name = normalize_gate_name(name)
        if name == "CNOT":
            if b is None:
                raise ValueError("CNOT requires a target qubit")
            self.apply_cnot(a, b)
        else:
            if name not in _SINGLE:
                raise ValueError(f"Unsupported state-vector gate {name}")
            self.apply_1q(_SINGLE[name], a)

    def vector(self) -> np.ndarray:
        return self.state.copy()

    def probabilities_z(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def sample_z(self, rng: np.random.Generator) -> Tuple[int, ...]:
        idx = int(rng.choice(len(self.state), p=self.probabilities_z()))
        return tuple((idx >> (self.n - 1 - q)) & 1 for q in range(self.n))


class QulacsStateBackend:
    """Qulacs logical state-vector backend.

    The dense matrix path is used for S^dag/T^dag so that the code does not
    depend on a particular convenience method name in Qulacs.
    """

    name = "qulacs"

    def __init__(self, n: int):
        self.n = int(n)
        from qulacs import QuantumState  # type: ignore
        self._QuantumState = QuantumState
        self.state = QuantumState(self.n)
        self.state.set_zero_state()

    def copy(self) -> "QulacsStateBackend":
        other = QulacsStateBackend(self.n)
        other.state.load(self.state.get_vector())
        return other

    def apply_1q(self, U: np.ndarray, q: int) -> None:
        from qulacs import QuantumCircuit  # type: ignore
        from qulacs.gate import DenseMatrix  # type: ignore
        circ = QuantumCircuit(self.n)
        circ.add_gate(DenseMatrix(q, U.tolist()))
        circ.update_quantum_state(self.state)

    def apply_cnot(self, c: int, t: int) -> None:
        from qulacs import QuantumCircuit  # type: ignore
        circ = QuantumCircuit(self.n)
        circ.add_CNOT_gate(c, t)
        circ.update_quantum_state(self.state)

    def apply_gate(self, name: str, a: int, b: Optional[int] = None) -> None:
        name = normalize_gate_name(name)
        if name == "CNOT":
            if b is None:
                raise ValueError("CNOT requires a target qubit")
            self.apply_cnot(a, b)
        else:
            if name not in _SINGLE:
                raise ValueError(f"Unsupported state-vector gate {name}")
            self.apply_1q(_SINGLE[name], a)

    def vector(self) -> np.ndarray:
        return np.array(self.state.get_vector(), dtype=complex)

    def probabilities_z(self) -> np.ndarray:
        v = self.vector()
        return np.abs(v) ** 2

    def sample_z(self, rng: np.random.Generator) -> Tuple[int, ...]:
        p = self.probabilities_z()
        idx = int(rng.choice(len(p), p=p))
        return tuple((idx >> (self.n - 1 - q)) & 1 for q in range(self.n))



class StimCliffordBackend:
    """Stim TableauSimulator backend for Clifford-only logical branches.

    This backend is intended for nonclifford_mode='qpd', where every T/Tdag has
    already been replaced by one of the Clifford terms I/S/SDG/Z.  It refuses a
    true T/Tdag gate, because that would no longer be a Clifford branch.
    """

    name = "stim"

    def __init__(self, n: int):
        self.n = int(n)
        import stim  # type: ignore
        self._stim = stim
        self.sim = stim.TableauSimulator()

    def copy(self) -> "StimCliffordBackend":
        other = StimCliffordBackend(self.n)
        other.sim = self.sim.copy()
        return other

    def _do(self, stim_name: str, targets: Sequence[int]) -> None:
        circ = self._stim.Circuit()
        circ.append(stim_name, [int(t) for t in targets])
        # Stim has exposed this as do(circuit) for TableauSimulator for a long
        # time; the fallback direct methods cover older builds.
        if hasattr(self.sim, "do"):
            self.sim.do(circ)
            return
        if hasattr(self.sim, "do_circuit"):
            self.sim.do_circuit(circ)
            return
        raise RuntimeError("Stim TableauSimulator does not expose do/do_circuit")

    def apply_gate(self, name: str, a: int, b: Optional[int] = None) -> None:
        name = normalize_gate_name(name)
        if name in ("T", "T_DAG"):
            raise ValueError("StimCliffordBackend cannot apply true T/Tdag; use QPD branches first")
        if name == "I":
            return
        if name == "CNOT":
            if b is None:
                raise ValueError("CNOT requires a target qubit")
            self._do("CX", [a, b])
        elif name == "SDG":
            self._do("S_DAG", [a])
        elif name in ("X", "Y", "Z", "H", "S"):
            self._do(name, [a])
        else:
            raise ValueError(f"Unsupported Stim Clifford gate {name}")

    def vector(self) -> np.ndarray:
        # Used only for optional single-shot diagnostics.  It is exponential in n,
        # so large-QEC work should rely on sample_z instead.
        return np.array(self.sim.state_vector(), dtype=complex)

    def probabilities_z(self) -> np.ndarray:
        v = self.vector()
        return np.abs(v) ** 2

    def sample_z(self, rng: np.random.Generator) -> Tuple[int, ...]:
        # Use a copied simulator so the final sampling does not mutate the stored
        # branch state if a caller later inspects vector().
        sim = self.sim.copy()
        return tuple(int(sim.measure(q)) for q in range(self.n))

def make_backend(n: int, backend: str = "auto"):
    backend = backend.lower().replace("-", "_")
    if backend == "numpy":
        return NumpyStateBackend(n)
    if backend == "stim":
        return StimCliffordBackend(n)
    if backend in ("stim_auto", "auto_stim"):
        try:
            return StimCliffordBackend(n)
        except Exception:
            return NumpyStateBackend(n)
    if backend in ("auto", "qulacs"):
        try:
            return QulacsStateBackend(n)
        except Exception:
            if backend == "qulacs":
                raise
            return NumpyStateBackend(n)
    raise ValueError("backend must be 'auto', 'qulacs', 'numpy', 'stim', or 'stim_auto'")


# ---------------------------------------------------------------------------
# Events and result objects
# ---------------------------------------------------------------------------

@dataclass
class CliffordChunk:
    gates: List[Gate]
    label: str


@dataclass
class SpecialGate:
    name: str
    target: int
    label: str


@dataclass
class LSCnotEvent:
    """A logical CNOT routed through the circuit-level LS CNOT shell."""
    control: int
    target: int
    label: str


PlanEvent = Any


@dataclass
class QECReport:
    label: str
    kind: str
    n_gates: int
    decoder: str
    samples_shape: Tuple[int, int]
    x_residual_rate: List[float]
    z_residual_rate: Optional[List[float]] = None
    note: str = ""
    active_shots: Optional[int] = None


@dataclass
class QPDBranchReport:
    label: str
    target: int
    gate: str
    gamma: float
    terms: List[Tuple[str, float]]
    term_counts: Dict[str, int]
    signed_weight_mean: float
    signed_weight_abs_mean: float
    s_branch_shell_label: Optional[str] = None
    note: str = "T QPD branch ensemble; S/SDG branch is routed through LS-S shell."


@dataclass
class MagicReport:
    label: str
    target: int
    p_magic: float
    sampled_faults: int
    shots: int
    correction_model: str = "Z|A> fault -> S^dag after ideal T"


@dataclass
class HybridRunResult:
    n_logical: int
    shots: int
    backend: str
    measurement_counts: Dict[Tuple[int, ...], int]
    qec_reports: List[QECReport] = field(default_factory=list)
    magic_reports: List[MagicReport] = field(default_factory=list)
    qpd_reports: List[QPDBranchReport] = field(default_factory=list)
    qpd_weighted_counts: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    qpd_weight_sum: float = 0.0
    qpd_weight_abs_sum: float = 0.0
    qpd_execution_backend: Optional[str] = None
    final_state_if_single_shot: Optional[np.ndarray] = None
    plan_summary: List[str] = field(default_factory=list)

    def logical_distribution(self) -> Dict[Tuple[int, ...], float]:
        return {k: v / self.shots for k, v in sorted(self.measurement_counts.items())}

    def qpd_reconstructed_distribution(self, normalize: bool = False) -> Dict[Tuple[int, ...], float]:
        """Signed QPD estimator for output probabilities.

        With normalize=False, returns the unbiased Monte Carlo estimator
        E[w * 1[outcome=x]], i.e. weighted_counts/shots.  normalize=True divides
        by the estimated total trace weight, which can make a finite-shot display
        easier to read but introduces a small bias.
        """
        if not self.qpd_weighted_counts:
            return self.logical_distribution()
        denom = self.qpd_weight_sum if normalize else float(self.shots)
        if abs(denom) < 1e-15:
            return {k: float("nan") for k in sorted(self.qpd_weighted_counts)}
        return {k: v / denom for k, v in sorted(self.qpd_weighted_counts.items())}

    def qpd_distribution(self, normalize: bool = False) -> Dict[Tuple[int, ...], float]:
        """Compatibility alias for qpd_reconstructed_distribution()."""
        return self.qpd_reconstructed_distribution(normalize=normalize)

    @property
    def qpd_weight_abs_mean(self) -> float:
        """Mean absolute signed trajectory weight, inferred from QPD reports."""
        if not self.qpd_reports:
            return 1.0
        prod = 1.0
        for report in self.qpd_reports:
            prod *= float(report.signed_weight_abs_mean)
        return prod

    def qpd_expectation_z(self, mask: Sequence[int], normalize: bool = False) -> float:
        """Estimate a final Z-parity observable from the QPD distribution.

        mask may be either a length-n bit mask or a list of qubit indices.
        The observable value is +1 for even parity and -1 for odd parity.
        """
        dist = self.qpd_reconstructed_distribution(normalize=normalize)
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

    def summary_lines(self) -> List[str]:
        dist = self.logical_distribution()
        dist_s = ", ".join(f"{''.join(map(str, k))}:{v:.3f}" for k, v in dist.items())
        lines = [
            f"backend={self.backend}, shots={self.shots}",
            f"plan_events={len(self.plan_summary)}, qec_reports={len(self.qec_reports)}, magic_reports={len(self.magic_reports)}, qpd_reports={len(self.qpd_reports)}",
            "sampled_branch_distribution=" + (dist_s if dist_s else "<not sampled>"),
        ]
        if self.qpd_weighted_counts:
            qdist = self.qpd_reconstructed_distribution(normalize=False)
            qdist_s = ", ".join(f"{''.join(map(str, k))}:{v:.3f}" for k, v in qdist.items())
            lines.append("qpd_reconstructed_distribution=" + (qdist_s if qdist_s else "<empty>"))
            lines.append(f"qpd_weight_sum_per_shot={self.qpd_weight_sum / self.shots:.6f}")
            lines.append(f"qpd_weight_abs_mean={self.qpd_weight_abs_mean:.6f}")
        return lines


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class HybridFTQCRuntime:
    """Trajectory-level hybrid FTQC simulator.

    Parameters
    ----------
    n_logical:
        Number of logical qubits.
    p_phys:
        Physical error probability supplied to the Stim Clifford/QEC compiler.
    backend:
        'auto' uses Qulacs if installed, otherwise NumPy. Use 'qulacs' to force
        Qulacs and fail if unavailable. Use 'qpd' for the QPD backend with
        Stim logical Clifford branch sampling when available, or 'stim_qpd' to
        require Stim for the QPD branch sampler.
    tail_rounds:
        Extra syndrome rounds after each Clifford chunk.
    magic_distillation_shots:
        Monte Carlo shots used once to estimate the 15-to-1 output error rate.
        Use 0 for the deterministic leading-order estimate 35*p_phys**3.
    mirror_z_residuals:
        If True, sample an independent Z_L residual channel with the same
        per-qubit rates as the decoded X_L channel. This is an approximation;
        the existing Stim compiler physically exposes Z-observable flips, i.e.
        residual X_L faults.
    qec_mode:
        "required" raises if Stim/PyMatching are unavailable; "optional" records
        skipped zero-residual QEC shells; "off" disables QEC sampling.
    """

    def __init__(
        self,
        n_logical: int,
        p_phys: float = 0.005,
        backend: str = "auto",
        tail_rounds: int = 1,
        magic_distillation_shots: int = 0,
        mirror_z_residuals: bool = False,
        seed: int = 1,
        qec_mode: str = "optional",
        nonclifford_mode: str = "statevector",
        route_qpd_s_via_ls: bool = True,
        use_lattice_surgery_for_qpd_s: Optional[bool] = None,
        ls_merge_rounds: int = 3,
        use_lattice_surgery_for_s: bool = False,
        use_lattice_surgery_for_t_correction: bool = False,
        cnot_mode: str = "transversal",
    ):
        self.n = int(n_logical)
        self.p_phys = float(p_phys)
        backend_key = str(backend).lower().replace("-", "_")
        self.backend_requested = backend_key
        self.backend_is_qpd_alias = backend_key in ("qpd", "auto_qpd", "qpd_auto", "stim_qpd", "qpd_stim")
        if backend_key in ("qpd", "auto_qpd", "qpd_auto"):
            # Convenience API: backend="qpd" selects QPD mode and uses Stim for
            # the final Clifford branch sampler when Stim is installed, falling
            # back to NumPy only for dependency-light smoke tests.
            backend = "stim_auto"
            nonclifford_mode = "qpd"
        elif backend_key in ("stim_qpd", "qpd_stim"):
            # Strict QPD backend: every sampled T-QPD branch is a Clifford-only
            # logical circuit sampled by Stim.  If Stim is not installed, this
            # raises instead of silently using NumPy.
            backend = "stim"
            nonclifford_mode = "qpd"
        self.backend_name = backend
        self.tail_rounds = int(tail_rounds)
        self.magic_distillation_shots = int(magic_distillation_shots)
        self.mirror_z_residuals = bool(mirror_z_residuals)
        self.seed = int(seed)
        if qec_mode not in ("required", "optional", "off"):
            raise ValueError("qec_mode must be required, optional, or off")
        self.qec_mode = qec_mode
        nonclifford_mode = nonclifford_mode.lower()
        aliases = {"magic": "statevector", "magic_teleport": "statevector", "teleport": "statevector"}
        nonclifford_mode = aliases.get(nonclifford_mode, nonclifford_mode)
        if nonclifford_mode not in ("statevector", "qpd"):
            raise ValueError("nonclifford_mode must be 'statevector' or 'qpd'")
        self.nonclifford_mode = nonclifford_mode
        if use_lattice_surgery_for_qpd_s is not None:
            route_qpd_s_via_ls = bool(use_lattice_surgery_for_qpd_s)
        self.route_qpd_s_via_ls = bool(route_qpd_s_via_ls)
        self.ls_merge_rounds = int(ls_merge_rounds)
        # Accepted for compatibility with earlier demo drivers. Bare S gates still
        # use the generic special shell unless nonclifford_mode='qpd' selects an
        # S/SDG term inside a T decomposition.
        self.use_lattice_surgery_for_s = bool(use_lattice_surgery_for_s)
        self.use_lattice_surgery_for_t_correction = bool(use_lattice_surgery_for_t_correction)
        cnot_mode = str(cnot_mode).lower()
        if cnot_mode not in ("transversal", "lattice_surgery"):
            raise ValueError("cnot_mode must be 'transversal' or 'lattice_surgery'")
        self.cnot_mode = cnot_mode

    # ------------------------- planning -------------------------
    def plan(self, qc_or_gates: Any) -> List[PlanEvent]:
        gates = parse_qulacs_or_gate_list(qc_or_gates)
        events: List[PlanEvent] = []
        chunk: List[Gate] = []
        chunk_id = 0

        def flush(reason: str) -> None:
            nonlocal chunk, chunk_id
            if chunk:
                events.append(CliffordChunk(gates=chunk, label=f"clifford_chunk_{chunk_id}_{reason}"))
                chunk = []
                chunk_id += 1

        for gi, (name, a, b) in enumerate(gates):
            name = normalize_gate_name(name)
            if name in ("CNOT", "CX") and self.cnot_mode == "lattice_surgery":
                flush(f"before_CNOT_{gi}")
                events.append(LSCnotEvent(control=a, target=b, label=f"CNOT_{gi}"))
            elif name in CLIFFORD_CHUNK_GATES:
                chunk.append((name, a, b))
            elif name in ("S", "SDG"):
                flush(f"before_{name}_{gi}")
                events.append(SpecialGate(name=name, target=a, label=f"{name}_{gi}"))
            elif name in ("T", "T_DAG"):
                flush(f"before_{name}_{gi}")
                events.append(SpecialGate(name=name, target=a, label=f"{name}_{gi}"))
            else:
                raise ValueError(f"Unsupported gate in hybrid plan: {name}")
        flush("tail")
        return events

    # ------------------------- Stim QEC sampling -------------------------
    def _sample_qec_residuals(
        self,
        chunk: CliffordChunk,
        shots: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
        """Return decoded residual logical X faults from a Stim Clifford chunk.

        The returned matrix has shape (shots, n_logical), where 1 means a
        residual X_L fault should be applied to that logical qubit.
        """
        try:
            import pymatching  # type: ignore
            from ftqc.clifford_ft_full import CliffordFTCompiler, layer_gates  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Stim/PyMatching and clifford_ft_full.py are required for QEC sampling. "
                "Install stim, pymatching, and run from the directory containing the uploaded files."
            ) from exc

        comp = CliffordFTCompiler(self.n, p=self.p_phys, tail_rounds=self.tail_rounds)
        layers = layer_gates(chunk.gates, self.n)
        circ, _ = comp.compile(layers, noise=True)
        det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
        try:
            dem = circ.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            pred = matcher.decode_batch(det)
            decoder = "MWPM"
        except ValueError:
            from stimbposd import BPOSD  # type: ignore
            dem = circ.detector_error_model(decompose_errors=False)
            pred = BPOSD(dem, max_bp_iters=40).decode_batch(det)
            decoder = "BP+OSD"
        x_residuals = (obs ^ pred).astype(np.uint8)

        z_residuals = None
        note = "Stim compiler observable flips are mapped to residual X_L faults."
        if self.mirror_z_residuals:
            rates = x_residuals.mean(axis=0)
            z_residuals = (rng.random((shots, self.n)) < rates).astype(np.uint8)
            note += " mirror_z_residuals=True added approximate independent Z_L faults."
        return x_residuals, decoder, z_residuals, note

    def _sample_memory_shell(self, label: str, shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
        """Stim QEC shell with no logical gate, used around S/T handoffs."""
        return self._sample_qec_residuals(CliffordChunk(gates=[], label=label), shots, rng)

    def _sample_ls_s_shell_residuals(
        self,
        label: str,
        target: int,
        shots: int,
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
        """Sample/decode the LS-S shell and map its observable to a full residual matrix."""
        try:
            from ftqc.ls_handoff_shells import build_ls_s_shell, decode_shell  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ls_handoff_shells.py, lattice_surgery.py, stim, and pymatching are required "
                "for LS-S shell sampling."
            ) from exc

        shell = build_ls_s_shell(
            p=self.p_phys,
            merge_rounds=self.ls_merge_rounds,
            noise=True,
            logical_target=target,
        )
        decoded = decode_shell(shell, shots=shots)
        full = np.zeros((shots, self.n), dtype=np.uint8)
        for col, logical_q in enumerate(shell.observable_to_logical):
            if col < decoded.residuals.shape[1]:
                full[:, int(logical_q)] ^= decoded.residuals[:, col].astype(np.uint8)
        note = decoded.note + f" Routed by {label}."
        return full, decoded.decoder, None, note

    def _sample_ls_cnot_shell_residuals(
        self,
        label: str,
        control: int,
        target: int,
        shots: int,
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
        """Sample/decode the LS CNOT shell and map its two observables to a
        full residual matrix (obs0 -> control X_L fault, obs1 -> target X_L fault).

        The shell is the circuit-level LS CNOT (ls_cnot_ft.LSCnotFT): zero gauge
        detectors, CNOT truth table verified, decoded without
        allow_gauge_detectors. The published Stage 03 tests check graphlike
        distance 3; this check is not a proof for all ungraphlike mechanisms.
        """
        try:
            from ftqc.ls_handoff_shells import build_ls_cnot_shell, decode_shell  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ls_handoff_shells.py, ls_cnot_ft.py, stim, and pymatching are required "
                "for LS CNOT shell sampling."
            ) from exc

        shell = build_ls_cnot_shell(
            p=self.p_phys,
            merge_rounds=self.ls_merge_rounds,
            noise=True,
            logical_control=control,
            logical_target=target,
        )
        decoded = decode_shell(shell, shots=shots)
        full = np.zeros((shots, self.n), dtype=np.uint8)
        for col, logical_q in enumerate(shell.observable_to_logical):
            if col < decoded.residuals.shape[1]:
                full[:, int(logical_q)] ^= decoded.residuals[:, col].astype(np.uint8)
        note = decoded.note + f" Routed by {label}."
        return full, decoded.decoder, None, note

    @staticmethod
    def _qpd_terms_for_gate(name: str) -> Tuple[Tuple[str, float], ...]:
        name = normalize_gate_name(name)
        if name == "T":
            return T_QPD_TERMS
        if name == "T_DAG":
            return TDG_QPD_TERMS
        raise ValueError(f"No T-QPD decomposition for {name}")

    # ------------------------- magic-state model -------------------------
    def _estimate_magic_error(self) -> float:
        if self.magic_distillation_shots <= 0:
            return float(min(1.0, 35.0 * self.p_phys ** 3))
        try:
            from ftqc.t_gate import build_15to1, distill  # type: ignore
        except Exception as exc:
            raise RuntimeError("t_gate.py is required for magic-state distillation estimates.") from exc
        HX, HZ = build_15to1()
        _, p_magic = distill(HX, HZ, self.p_phys, shots=self.magic_distillation_shots, seed=self.seed)
        return float(p_magic)

    # ------------------------- trajectory update helpers -------------------------
    @staticmethod
    def _apply_logical_residuals(state: Any, x_row: np.ndarray, z_row: Optional[np.ndarray] = None) -> None:
        for q, bit in enumerate(x_row):
            if int(bit):
                state.apply_gate("X", q)
        if z_row is not None:
            for q, bit in enumerate(z_row):
                if int(bit):
                    state.apply_gate("Z", q)

    @staticmethod
    def _apply_chunk_to_state(state: Any, gates: Sequence[Gate]) -> None:
        for name, a, b in gates:
            state.apply_gate(name, a, b)

    # ------------------------- public execution -------------------------
    def run(
        self,
        qc_or_gates: Any,
        shots: int = 1000,
        seed: Optional[int] = None,
        sample_final_z: bool = True,
    ) -> HybridRunResult:
        """Run noisy hybrid trajectories and return final Z-basis samples.

        In nonclifford_mode='statevector', T/Tdag are applied directly at the
        logical state-vector layer and protected by a generic Stim QEC handoff
        shell.  In nonclifford_mode='qpd', every T/Tdag is sampled as a signed
        Clifford-channel branch.  The S/SDG branch is routed through the LS-S
        shell, while the I/Z branches use a generic Stim memory shell.  The
        reconstructed QPD probabilities are available via
        result.qpd_reconstructed_distribution().
        """
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        events = self.plan(qc_or_gates)

        # Batch-sample all stochastic data once, then replay per shot.
        qec_samples: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        qec_reports: List[QECReport] = []
        magic_faults: Dict[str, np.ndarray] = {}
        magic_reports: List[MagicReport] = []
        qpd_term_samples: Dict[str, np.ndarray] = {}
        qpd_weight_samples: Dict[str, np.ndarray] = {}
        qpd_reports: List[QPDBranchReport] = []
        p_magic_cache: Optional[float] = None

        def zero_qec_sample(note: str) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
            x0 = np.zeros((shots, self.n), dtype=np.uint8)
            return x0, "skipped", None, note

        def guarded_qec_sample(chunk: CliffordChunk, kind_note: str) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
            if self.qec_mode == "off":
                return zero_qec_sample("qec_mode=off; no Stim sampling performed")
            try:
                return self._sample_qec_residuals(chunk, shots, rng)
            except Exception as exc:
                if self.qec_mode == "required":
                    raise
                return zero_qec_sample(
                    f"qec_mode=optional; Stim QEC skipped for {kind_note}: "
                    f"{type(exc).__name__}: {exc}"
                )

        def guarded_ls_s_sample(label: str, target: int) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
            if self.qec_mode == "off":
                return zero_qec_sample("qec_mode=off; LS-S shell sampling disabled")
            try:
                return self._sample_ls_s_shell_residuals(label, target, shots)
            except Exception as exc:
                if self.qec_mode == "required":
                    raise
                return zero_qec_sample(
                    f"qec_mode=optional; LS-S shell skipped for {label}: "
                    f"{type(exc).__name__}: {exc}"
                )

        def guarded_ls_cnot_sample(label: str, control: int, target: int) -> Tuple[np.ndarray, str, Optional[np.ndarray], str]:
            if self.qec_mode == "off":
                return zero_qec_sample("qec_mode=off; LS CNOT shell sampling disabled")
            try:
                return self._sample_ls_cnot_shell_residuals(label, control, target, shots)
            except Exception as exc:
                if self.qec_mode == "required":
                    raise
                return zero_qec_sample(
                    f"qec_mode=optional; LS CNOT shell skipped for {label}: "
                    f"{type(exc).__name__}: {exc}"
                )

        for ev in events:
            if isinstance(ev, LSCnotEvent):
                x, decoder, z, note = guarded_ls_cnot_sample(ev.label, ev.control, ev.target)
                qec_samples[ev.label] = (x, z)
                qec_reports.append(QECReport(
                    label=ev.label,
                    kind="ls_cnot_shell",
                    n_gates=1,
                    decoder=decoder,
                    samples_shape=tuple(x.shape),
                    x_residual_rate=x.mean(axis=0).round(8).tolist(),
                    z_residual_rate=None if z is None else z.mean(axis=0).round(8).tolist(),
                    note=note,
                ))
                continue

            if isinstance(ev, CliffordChunk):
                x, decoder, z, note = guarded_qec_sample(ev, ev.label)
                qec_samples[ev.label] = (x, z)
                qec_reports.append(QECReport(
                    label=ev.label,
                    kind="clifford_chunk",
                    n_gates=len(ev.gates),
                    decoder=decoder,
                    samples_shape=tuple(x.shape),
                    x_residual_rate=x.mean(axis=0).round(8).tolist(),
                    z_residual_rate=None if z is None else z.mean(axis=0).round(8).tolist(),
                    note=note,
                ))
                continue

            if not isinstance(ev, SpecialGate):
                raise AssertionError(f"Unexpected plan event {ev!r}")

            if ev.name in ("T", "T_DAG") and self.nonclifford_mode == "qpd":
                terms = self._qpd_terms_for_gate(ev.name)
                coeffs = np.array([float(c) for _, c in terms], dtype=float)
                gamma = float(np.sum(np.abs(coeffs)))
                probs = np.abs(coeffs) / gamma
                choice = rng.choice(len(terms), size=shots, p=probs)
                selected = np.array([terms[int(i)][0] for i in choice], dtype=object)
                weights = np.array([gamma * (1.0 if coeffs[int(i)] >= 0 else -1.0) for i in choice], dtype=float)
                qpd_term_samples[ev.label] = selected
                qpd_weight_samples[ev.label] = weights

                unique, counts_arr = np.unique(selected, return_counts=True)
                term_counts = {str(k): int(v) for k, v in zip(unique, counts_arr)}
                s_mask = np.isin(selected, ["S", "SDG"])
                active_s = int(s_mask.sum())

                # I/Z branches remain Clifford memory handoffs.
                clifford_shell_label = ev.label + "_qpd_I_Z_stim_qec_shell"
                x, decoder, z, note = guarded_qec_sample(CliffordChunk(gates=[], label=clifford_shell_label), clifford_shell_label)
                qec_samples[clifford_shell_label] = (x, z)
                qec_reports.append(QECReport(
                    label=clifford_shell_label,
                    kind=f"{ev.name}_qpd_I_Z_shell",
                    n_gates=0,
                    decoder=decoder,
                    samples_shape=tuple(x.shape),
                    x_residual_rate=x.mean(axis=0).round(8).tolist(),
                    z_residual_rate=None if z is None else z.mean(axis=0).round(8).tolist(),
                    note=note + " Active for QPD I/Z branches.",
                    active_shots=shots - active_s,
                ))

                # The requested routing: S/SDG branch goes through LS-S shell.
                s_shell_label = ev.label + "_qpd_S_ls_s_shell"
                if self.route_qpd_s_via_ls:
                    x, decoder, z, note = guarded_ls_s_sample(s_shell_label, ev.target)
                    kind = f"{ev.name}_qpd_S_LS_S_shell"
                else:
                    x, decoder, z, note = guarded_qec_sample(CliffordChunk(gates=[], label=s_shell_label), s_shell_label)
                    kind = f"{ev.name}_qpd_S_generic_shell"
                qec_samples[s_shell_label] = (x, z)
                qec_reports.append(QECReport(
                    label=s_shell_label,
                    kind=kind,
                    n_gates=0,
                    decoder=decoder,
                    samples_shape=tuple(x.shape),
                    x_residual_rate=x.mean(axis=0).round(8).tolist(),
                    z_residual_rate=None if z is None else z.mean(axis=0).round(8).tolist(),
                    note=note + " Active only when the sampled T-QPD term is S/SDG.",
                    active_shots=active_s,
                ))

                qpd_reports.append(QPDBranchReport(
                    label=ev.label,
                    target=ev.target,
                    gate=ev.name,
                    gamma=gamma,
                    terms=[(str(name), float(coeff)) for name, coeff in terms],
                    term_counts=term_counts,
                    signed_weight_mean=float(weights.mean()),
                    signed_weight_abs_mean=float(np.abs(weights).mean()),
                    s_branch_shell_label=s_shell_label,
                ))
                continue

            if ev.name in ("S", "SDG", "T", "T_DAG"):
                # Generic handoff path used by the existing state-vector mode.
                shell_label = ev.label + "_stim_qec_shell"
                x, decoder, z, note = guarded_qec_sample(CliffordChunk(gates=[], label=shell_label), shell_label)
                qec_samples[shell_label] = (x, z)
                qec_reports.append(QECReport(
                    label=shell_label,
                    kind=f"{ev.name}_handoff_shell",
                    n_gates=0,
                    decoder=decoder,
                    samples_shape=tuple(x.shape),
                    x_residual_rate=x.mean(axis=0).round(8).tolist(),
                    z_residual_rate=None if z is None else z.mean(axis=0).round(8).tolist(),
                    note=note,
                ))
                if ev.name in ("T", "T_DAG"):
                    if p_magic_cache is None:
                        p_magic_cache = self._estimate_magic_error()
                    faults = (rng.random(shots) < p_magic_cache).astype(np.uint8)
                    magic_faults[ev.label] = faults
                    magic_reports.append(MagicReport(
                        label=ev.label,
                        target=ev.target,
                        p_magic=p_magic_cache,
                        sampled_faults=int(faults.sum()),
                        shots=shots,
                    ))
                continue

            raise AssertionError(f"Unexpected special gate {ev}")

        counts: Dict[Tuple[int, ...], int] = {}
        qpd_weighted_counts: Dict[Tuple[int, ...], float] = {}
        qpd_weight_sum = 0.0
        qpd_weight_abs_sum = 0.0
        final_state: Optional[np.ndarray] = None
        backend_used = "unknown"

        for sidx in range(shots):
            state = make_backend(self.n, self.backend_name)
            backend_used = state.name
            trajectory_weight = 1.0

            for ev in events:
                if isinstance(ev, LSCnotEvent):
                    state.apply_gate("CNOT", ev.control, ev.target)
                    x, z = qec_samples[ev.label]
                    self._apply_logical_residuals(state, x[sidx], None if z is None else z[sidx])
                    continue

                if isinstance(ev, CliffordChunk):
                    self._apply_chunk_to_state(state, ev.gates)
                    x, z = qec_samples[ev.label]
                    self._apply_logical_residuals(state, x[sidx], None if z is None else z[sidx])
                    continue

                if not isinstance(ev, SpecialGate):
                    raise AssertionError(f"Unexpected plan event {ev!r}")

                if ev.name in ("T", "T_DAG") and self.nonclifford_mode == "qpd":
                    term = str(qpd_term_samples[ev.label][sidx])
                    trajectory_weight *= float(qpd_weight_samples[ev.label][sidx])
                    if term != "I":
                        state.apply_gate(term, ev.target)
                    if term in ("S", "SDG"):
                        shell_label = ev.label + "_qpd_S_ls_s_shell"
                    else:
                        shell_label = ev.label + "_qpd_I_Z_stim_qec_shell"
                    x, z = qec_samples[shell_label]
                    self._apply_logical_residuals(state, x[sidx], None if z is None else z[sidx])
                    continue

                if ev.name == "T":
                    state.apply_gate("T", ev.target)
                    if int(magic_faults[ev.label][sidx]):
                        # Bad distilled |A> = Z|A> makes the corrected gadget apply T^dag.
                        # T^dag = S^dag T, so after ideal T we apply S^dag.
                        state.apply_gate("SDG", ev.target)
                elif ev.name == "T_DAG":
                    state.apply_gate("T_DAG", ev.target)
                    if int(magic_faults[ev.label][sidx]):
                        # Symmetric model: a bad |A^dag> applies T instead of T^dag.
                        state.apply_gate("S", ev.target)
                elif ev.name == "S":
                    state.apply_gate("S", ev.target)
                elif ev.name == "SDG":
                    state.apply_gate("SDG", ev.target)
                else:
                    raise AssertionError(f"Unexpected special gate {ev}")
                shell_label = ev.label + "_stim_qec_shell"
                x, z = qec_samples[shell_label]
                self._apply_logical_residuals(state, x[sidx], None if z is None else z[sidx])

            if sample_final_z:
                key = state.sample_z(rng)
            else:
                key = tuple()
            counts[key] = counts.get(key, 0) + 1
            if self.nonclifford_mode == "qpd":
                qpd_weighted_counts[key] = qpd_weighted_counts.get(key, 0.0) + trajectory_weight
                qpd_weight_sum += trajectory_weight
                qpd_weight_abs_sum += abs(trajectory_weight)
            if shots == 1:
                final_state = state.vector()

        plan_summary = []
        for ev in events:
            if isinstance(ev, LSCnotEvent):
                plan_summary.append(
                    f"{ev.label}: logical CNOT({ev.control},{ev.target}) via circuit-level LS CNOT shell (distance 2)"
                )
            elif isinstance(ev, CliffordChunk):
                plan_summary.append(f"{ev.label}: Stim Clifford/QEC chunk with {len(ev.gates)} gates")
            elif isinstance(ev, SpecialGate):
                if ev.name in ("T", "T_DAG") and self.nonclifford_mode == "qpd":
                    plan_summary.append(
                        f"{ev.label}: {ev.name} via T-QPD Clifford branches; S/SDG branch -> LS-S shell; branch_backend={self.backend_name}"
                    )
                else:
                    plan_summary.append(f"{ev.label}: {ev.name} via state-vector handoff + Stim QEC shell")

        result_backend = f"qpd+{backend_used}" if self.nonclifford_mode == "qpd" else backend_used

        return HybridRunResult(
            n_logical=self.n,
            shots=shots,
            backend=result_backend,
            measurement_counts=counts,
            qec_reports=qec_reports,
            magic_reports=magic_reports,
            qpd_reports=qpd_reports,
            qpd_weighted_counts=qpd_weighted_counts,
            qpd_weight_sum=float(qpd_weight_sum),
            qpd_weight_abs_sum=float(qpd_weight_abs_sum),
            qpd_execution_backend=result_backend if self.nonclifford_mode == "qpd" else None,
            final_state_if_single_shot=final_state,
            plan_summary=plan_summary,
        )

    def ideal_state(self, qc_or_gates: Any, backend: str = "numpy") -> np.ndarray:
        """Compute the noiseless ideal logical state for comparison."""
        gates = parse_qulacs_or_gate_list(qc_or_gates)
        state = make_backend(infer_n_logical(gates, self.n), backend)
        for name, a, b in gates:
            state.apply_gate(name, a, b)
        return state.vector()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """Small demo. Requires stim, pymatching, qulacs, and the uploaded modules."""
    try:
        from qulacs import QuantumCircuit  # type: ignore
    except Exception:
        QuantumCircuit = None

    if QuantumCircuit is not None:
        qc = QuantumCircuit(2)
        qc.add_H_gate(0)
        qc.add_T_gate(0)
        qc.add_H_gate(1)
        qc.add_CNOT_gate(0, 1)
        qc.add_T_gate(1)
        qc.add_S_gate(0)
        program: Any = qc
    else:
        program = [("H", 0, None), ("T", 0, None), ("H", 1, None), ("CNOT", 0, 1), ("T", 1, None), ("S", 0, None)]

    rt = HybridFTQCRuntime(
        n_logical=2,
        p_phys=0.005,
        backend="qpd",
        magic_distillation_shots=0,
        seed=7,
        qec_mode="optional",
        route_qpd_s_via_ls=True,
    )
    result = rt.run(program, shots=2000, seed=11)

    print("# Hybrid FTQC runtime demo")
    print("backend:", result.backend)
    print("plan:")
    for line in result.plan_summary:
        print("  -", line)
    print("QEC reports:")
    for r in result.qec_reports:
        print(f"  - {r.label}: {r.kind}, decoder={r.decoder}, X_L rates={r.x_residual_rate}")
    print("Magic reports:")
    for r in result.magic_reports:
        print(f"  - {r.label}: target=q{r.target}, p_magic={r.p_magic:.3e}, faults={r.sampled_faults}/{r.shots}")
    print("QPD reports:")
    for r in result.qpd_reports:
        print(f"  - {r.label}: gamma={r.gamma:.6f}, counts={r.term_counts}, S-shell={r.s_branch_shell_label}")
    print("Final sampled branch Z distribution:")
    for k, v in result.logical_distribution().items():
        print(f"  {''.join(map(str, k))}: {v:.4f}")
    print("QPD reconstructed Z distribution:")
    for k, v in result.qpd_distribution().items():
        print(f"  {''.join(map(str, k))}: {v:.4f}")


if __name__ == "__main__":
    demo()
