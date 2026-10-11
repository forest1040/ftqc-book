"""
Lattice-surgery handoff shells for the hybrid FTQC runtime.

This module supplies detector-bearing Stim circuits for the S/T handoff points
used by hybrid_ftqc_runtime.py.

Boundary of the model
---------------------
Stim cannot represent the non-stabilizer magic state |A>.  Therefore the true T
state update is still performed by the logical state-vector backend.  The Stim
circuit constructed here is the stabilizer/QEC skeleton around the handoff:
logical patches, repeated joint measurements, detector constraints, decoding,
and an observable on the data patch.  Magic-state quality is modeled separately
by t_gate.py and sampled by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ShellCircuit:
    circuit: object
    observable_to_logical: List[int]
    kind: str
    note: str


@dataclass
class ShellDecodeResult:
    residuals: np.ndarray
    decoder: str
    note: str


def _append_observable(ls, rec_idx: int, obs_index: int = 0) -> None:
    import stim  # type: ignore
    rr = lambda a: stim.target_rec(a - ls.mc)
    ls.c.append("OBSERVABLE_INCLUDE", [rr(rec_idx)], obs_index)


def build_ls_s_shell(
    p: float = 0.005,
    merge_rounds: int = 3,
    noise: bool = True,
    logical_target: int = 0,
) -> ShellCircuit:
    """Build a detector-bearing logical-level LS-S shell.

    The data patch is initialized as |0_L>, LS-S is applied with a |Y_L>
    ancilla, and the data patch is measured in Z_L.  Since S|0>=|0>, the Z_L
    observable is deterministic in the noiseless reference and can be used as a
    residual X_L channel estimate for the target logical qubit.
    """
    from ftqc.lattice_surgery import LatticeSurgery  # type: ignore

    ls = LatticeSurgery(2, merge_rounds=merge_rounds, p=p, noise=noise)
    ls.prep(0, "0")
    ls.ls_s(0, anc=1)
    z_rec = ls.measure_logical(0, "Z")
    _append_observable(ls, z_rec, obs_index=0)
    ls.append_detectors()
    return ShellCircuit(
        circuit=ls.c,
        observable_to_logical=[logical_target],
        kind="ls_s_shell",
        note="Detector-bearing logical-level LS-S shell; observable is target Z_L residual.",
    )


def build_ls_t_injection_shell(
    p: float = 0.005,
    merge_rounds: int = 3,
    noise: bool = True,
    logical_target: int = 0,
    include_s_correction_shell: bool = True,
) -> ShellCircuit:
    """Build a detector-bearing stabilizer skeleton for a T handoff.

    True |A_L> is non-stabilizer and is not represented in Stim.  Instead this
    circuit protects the Clifford/stabilizer part of the injection boundary:

      data patch |0_L>
      placeholder magic patch |+_L>
      LS-CNOT(data -> placeholder)
      magic-patch Z_L measurement
      optional always-on LS-S correction shell on the data patch
      data-patch Z_L observable

    The optional LS-S shell is conservative: a real teleportation branch applies
    S only when the injection measurement requires it.  The runtime records this
    in the report note.  Magic-state logical faults are sampled separately.
    """
    from ftqc.lattice_surgery import LatticeSurgery  # type: ignore

    n_patches = 4 if include_s_correction_shell else 3
    ls = LatticeSurgery(n_patches, merge_rounds=merge_rounds, p=p, noise=noise)
    ls.prep(0, "0")
    ls.prep(1, "+")
    ls.ls_cnot(0, 1, anc=2)
    ls.measure_logical(1, "Z")
    if include_s_correction_shell:
        ls.ls_s(0, anc=3)
    z_rec = ls.measure_logical(0, "Z")
    _append_observable(ls, z_rec, obs_index=0)
    ls.append_detectors()
    return ShellCircuit(
        circuit=ls.c,
        observable_to_logical=[logical_target],
        kind="ls_t_injection_shell",
        note=(
            "Detector-bearing LS T-injection stabilizer skeleton. True |A_L> and "
            "non-Clifford update are handled by the state-vector backend; "
            "magic-state faults are sampled separately. "
            + ("Includes conservative always-on LS-S correction shell." if include_s_correction_shell else "No LS-S correction shell included.")
        ),
    )


def decode_shell(shell: ShellCircuit, shots: int = 1000) -> ShellDecodeResult:
    """Sample and decode a ShellCircuit.

    Returns an array of shape (shots, len(shell.observable_to_logical)) where a
    1 marks a decoded residual observable flip.  The runtime maps those shell
    observables back to full logical-qubit X_L faults.
    """
    import pymatching  # type: ignore

    circ = shell.circuit
    det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
    try:
        dem = circ.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=False,
        )
        matcher = pymatching.Matching.from_detector_error_model(dem)
        pred = matcher.decode_batch(det)
        decoder = "MWPM"
    except ValueError:
        from stimbposd import BPOSD  # type: ignore
        dem = circ.detector_error_model(
            decompose_errors=False,
            allow_gauge_detectors=False,
        )
        pred = BPOSD(dem, max_bp_iters=40).decode_batch(det)
        decoder = "BP+OSD"
    residuals = (obs ^ pred).astype(np.uint8)
    if residuals.ndim == 1:
        residuals = residuals.reshape((shots, 1))
    return ShellDecodeResult(residuals=residuals, decoder=decoder, note=shell.note)


def demo() -> None:
    """Minimal smoke demo. Requires stim and pymatching."""
    for label, builder in [
        ("S", build_ls_s_shell),
        ("T", build_ls_t_injection_shell),
    ]:
        shell = builder(p=0.0, noise=False)
        det, obs = shell.circuit.compile_detector_sampler().sample(100, separate_observables=True)
        print(f"{label} shell: detectors={shell.circuit.num_detectors}, obs={shell.circuit.num_observables}, noiseless_det_all_zero={not det.any()}")


if __name__ == "__main__":
    demo()
