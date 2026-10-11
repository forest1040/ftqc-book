"""
Lattice-surgery handoff shells for the hybrid FTQC runtime (circuit-level version).
================================================================================
Replaces the MPP-based logical-level LS shells (lattice_surgery.py) with
circuit-level (ancilla-based syndrome extraction) shells built on top of
surface_patch.py and circuit_level_merge.py.

Key improvements over the MPP-based version:
  - Syndrome extraction uses ancilla qubits with CX schedules (circuit-level
    noise model) instead of direct MPP measurements (phenomenological model).
  - Single-patch memory uses SurfacePatch wrapping
    stim.Circuit.generated("surface_code:rotated_memory_z").
  - Merge operations use CircuitLevelMerge which implements seam stabilizers
    with ancilla-based syndrome extraction. Its shortest graphlike logical
    error has length 3 (a circuit-distance upper bound, not an exact proof).
  - The CX schedule follows the FT ordering extracted from surface_patch
    (X-type: NE,NW,SE,SW / Z-type: NE,SE,NW,SW) to avoid hook errors.

Model boundary (unchanged from the MPP version):
  Stim cannot represent the non-stabilizer magic state |A>. Therefore the true T
  state update is still performed by the logical state-vector backend.  The Stim
  circuit constructed here is the stabilizer/QEC skeleton around the handoff:
  logical patches, seam stabilizer measurements, detector constraints, decoding,
  and an observable on the data patch.  Magic-state quality is modeled separately
  by t_gate.py and sampled by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ShellCircuit:
    """Detector-bearing Stim circuit for a single handoff point.

    Attributes:
        circuit: The stim.Circuit with DETECTOR and OBSERVABLE_INCLUDE.
        observable_to_logical: Mapping from observable index to logical qubit.
        kind: Human-readable circuit type label.
        note: Free-text description including parameters.
    """
    circuit: object
    observable_to_logical: List[int]
    kind: str
    note: str


@dataclass
class ShellDecodeResult:
    """Decoded residual logical flip rates from a ShellCircuit.

    Attributes:
        residuals: Array of shape (shots, num_mapped_observables) with 1 for
            a decoded residual flip.
        decoder: Name of the decoder used ("MWPM" or "BP+OSD").
        note: Free-text description forwarded from the source ShellCircuit.
    """
    residuals: np.ndarray
    decoder: str
    note: str


def decode_shell(shell: ShellCircuit, shots: int = 1000) -> ShellDecodeResult:
    """Sample and decode a ShellCircuit.

    Compiles the detector sampler, runs MWPM (with BP+OSD fallback for
    non-graphlike detector error models), and returns the decoded residual
    observable flips.

    Uses allow_gauge_detectors=False.  All shells produced here are built so
    that the noiseless circuit has zero non-deterministic detectors; if the DEM
    build raises a "non-deterministic detector" error, that signals a malformed
    shell (the gauge allowance would only hide the bug and collapse distance),
    so the error is allowed to propagate as a correctness guard.

    Args:
        shell: The ShellCircuit to sample and decode.
        shots: Number of detector sampling shots.

    Returns:
        ShellDecodeResult with residuals of shape (shots, num_observables).
        Each column corresponds to an OBSERVABLE_INCLUDE in the circuit.
        A 1 in the residual means the decoder predicts a logical flip.
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


# ---------------------------------------------------------------------------
# LS-S shell: circuit-level ZZ merge for S gate handoff
# ---------------------------------------------------------------------------

def build_ls_s_shell(
    p: float = 0.005,
    merge_rounds: int = 3,
    noise: bool = True,
    logical_target: int = 0,
) -> ShellCircuit:
    """Build a detector-bearing circuit-level LS-S shell.

    Uses CircuitLevelMerge("ZZ") to create a ZZ merge between two surface-code
    patches.  The data patch (patch A) is initialized as |0_L> and measured in
    Z_L after the merge rounds.  The ancilla patch (patch B) provides the
    partner for the M_ZZ joint measurement.

    Both patches use ancilla-based syndrome extraction (circuit-level noise
    model) with the FT CX schedule verified in circuit_level_merge.py
    (graphlike upper bound = 3, consistent with the intended patch distance).

    The Z_L observable on the data patch is deterministic (=0 for |0_L>) in the
    noiseless reference.  Decoded residuals capture the logical X_L fault rate
    introduced by the LS-S handoff, which the runtime applies to the logical
    state-vector trajectory.

    Args:
        p: Physical error probability for circuit-level noise.
        merge_rounds: Number of syndrome rounds in the merged configuration.
        noise: If True, apply circuit-level noise; if False, noiseless.
        logical_target: Logical qubit index mapped to observable 0.

    Returns:
        ShellCircuit with 2 observables (only observable 0 is mapped to
        logical_target).  The circuit includes DETECTOR constraints for
        all split/merge transition rounds.
    """
    from ftqc.circuit_level_merge import CircuitLevelMerge

    clm = CircuitLevelMerge("ZZ")
    c, _joint_recs, _drec = clm.build(
        aone=False,
        bone=False,
        merge_rounds=merge_rounds,
        split_rounds=2,
        p=p,
        noise=noise,
    )
    return ShellCircuit(
        circuit=c,
        observable_to_logical=[logical_target],
        kind="ls_s_shell_circuit_level",
        note=(
            "Circuit-level LS-S shell using ancilla syndrome extraction. "
            "Two patches merged via M_ZZ seam stabilizers. "
            f"merge_rounds={merge_rounds}, noise={noise}, p={p}. "
            "Observable 0 = Z_L(A) (data patch residual X_L fault channel). "
            "shortest_graphlike_error verified as 3 (FT preserved)."
        ),
    )


# ---------------------------------------------------------------------------
# LS CNOT shell: circuit-level ZZ+XX merge for CNOT handoff
# ---------------------------------------------------------------------------

def build_ls_cnot_shell(
    p: float = 0.005,
    merge_rounds: int = 3,
    split_rounds: int = 2,
    noise: bool = True,
    control_one: bool = False,
    target_one: bool = False,
    logical_control: int = 0,
    logical_target: int = 1,
) -> ShellCircuit:
    """Build a detector-bearing circuit-level LS CNOT shell.

    Uses ls_cnot_ft.LSCnotFT to create a 3-patch circuit (control, ancilla,
    target) with sequential ZZ and XX merges, built on the proven
    circuit_level_merge construction (_compose_in transition detectors, FT CX
    schedule, final spacelike-boundary detectors).

    Properties (verified in ls_cnot_ft.py):
      - Zero non-deterministic detectors -> allow_gauge_detectors NOT needed.
      - Computational-basis CNOT truth table matches with Pauli-frame-corrected
        observables (obs0 = Z_L(C), obs1 = Z_L(T) ^ m_zz ^ m_anc).
      - graphlike upper bound = 3 (consistent with patch distance 3): the
        merge->split seam-split closure detectors (conserved product of the two
        boundary stabilizers the seam splits into) remove the corner degeneracy
        at T's lower-left (XX seam) that would otherwise cap distance at 2.

    Args:
        p: Physical error probability for circuit-level noise.
        merge_rounds: Number of syndrome rounds per merge configuration.
        split_rounds: Number of split rounds before/after each merge.
        noise: If True, apply circuit-level noise.
        control_one: If True, prepare control in |1_L> (for truth-table tests).
        target_one: If True, prepare target in |1_L> (for truth-table tests).
        logical_control: Logical qubit index for the control observable.
        logical_target: Logical qubit index for the target observable.

    Returns:
        ShellCircuit with 2 observables: 0=Z_L(control), 1=frame-corrected
        Z_L(target).
    """
    from ftqc.ls_cnot_ft import LSCnotFT

    g = LSCnotFT()
    c, _drec, _zz_joint_recs = g.build(
        control_one=control_one,
        target_one=target_one,
        merge_rounds=merge_rounds,
        split_rounds=split_rounds,
        p=p,
        noise=noise,
    )
    return ShellCircuit(
        circuit=c,
        observable_to_logical=[logical_control, logical_target],
        kind="ls_cnot_shell_circuit_level",
        note=(
            "Circuit-level LS CNOT shell (ls_cnot_ft.LSCnotFT). "
            "L-shaped 3-patch layout: control + ancilla ZZ-merge, then "
            "ancilla + target XX-merge, ancilla measured in Z. "
            f"merge_rounds={merge_rounds}, split_rounds={split_rounds}, "
            f"noise={noise}, p={p}. "
            "Observables: 0=Z_L(control), 1=Z_L(target)^m_zz^m_anc (frame corrected). "
            "Zero gauge detectors (allow_gauge_detectors NOT used); CNOT truth "
            "table verified. graphlike upper bound=3 (consistent with patch distance; "
            "merge->split seam-split closure detectors remove the XX-seam corner "
            "degeneracy)."
        ),
    )


# ---------------------------------------------------------------------------
# LS-T injection shell: circuit-level stabilizer skeleton for T handoff
# ---------------------------------------------------------------------------

def build_ls_t_injection_shell(
    p: float = 0.005,
    merge_rounds: int = 3,
    split_rounds: int = 2,
    noise: bool = True,
    data_one: bool = False,
    logical_target: int = 0,
    include_s_correction_shell: bool = True,
) -> ShellCircuit:
    """Build a detector-bearing circuit-level skeleton for the T-teleportation.

    Faithful to the gate-teleportation structure (ls_cnot_ft.LSTeleportFT):
        T|psi> via  CNOT(data -> magic) = M_ZZ(data, cnot_anc) + M_XX(cnot_anc, magic),
        then measure magic in Z (outcome m), then S correction on data if m=1.
    The data (control) patch survives carrying T|psi>; the magic (target) is
    measured out.  With include_s_correction_shell=True, the S correction is
    realized as an additional M_ZZ merge on the data patch (the LS-S structure,
    matching circuit_level_merge), so the full chain is M_ZZ + M_XX + M_ZZ(S).

    The true |A_L> = T|+> is non-stabilizer and cannot be prepared in Stim; the
    magic patch is the Clifford skeleton (|+>) and the non-Clifford T-ness plus
    the magic-state fault are handled at the state-vector / magic layer
    (t_gate.py).  Likewise the |Y> content of the S ancilla is a logical-layer
    concern; here the S correction contributes only its merge QEC cost (the
    ancilla is |0>, as in circuit_level_merge's LS-S).  The S correction commutes
    with the data Z readout, so it changes only the data residual's noise, not
    its value.

    Verified: zero gauge detectors and graphlike upper bound 3 (consistent
    with patch distance 3) for both with/without the S correction merge.

    Args:
        p: Physical error probability for circuit-level noise.
        merge_rounds: Syndrome rounds per merge configuration.
        split_rounds: Split rounds between/around merges.
        noise: If True, apply circuit-level noise.
        data_one: If True, prepare the data patch in |1_L> (for skeleton checks).
        logical_target: Logical qubit index mapped to observable 0 (data residual).
        include_s_correction_shell: If True, append the S-correction M_ZZ merge.

    Returns:
        ShellCircuit with observable 0 = Z_L of the data patch (residual X_L
        fault channel).
    """
    from ftqc.ls_cnot_ft import LSTeleportFT

    g = LSTeleportFT(with_s=include_s_correction_shell)
    c, _drec, _info = g.build_teleport(
        data_one=data_one,
        merge_rounds=merge_rounds,
        split_rounds=split_rounds,
        p=p,
        noise=noise,
    )
    chain = "M_ZZ + M_XX + M_ZZ(S)" if include_s_correction_shell else "M_ZZ + M_XX"
    return ShellCircuit(
        circuit=c,
        observable_to_logical=[logical_target],
        kind="ls_t_injection_shell_circuit_level",
        note=(
            "Circuit-level T-teleportation skeleton (ls_cnot_ft.LSTeleportFT). "
            f"Gate teleportation: CNOT(data->magic) = M_ZZ(data,anc) + M_XX(anc,magic), "
            f"measure magic in Z, S correction on data ({chain}). "
            f"merge_rounds={merge_rounds}, split_rounds={split_rounds}, noise={noise}, p={p}. "
            "Observable 0 = Z_L(data) residual. True |A_L> and non-Clifford T-ness "
            "handled at the state-vector/magic layer (t_gate.py); magic-state faults "
            "sampled separately. Zero gauge detectors; graphlike upper bound=3. "
            f"S correction merge {'included' if include_s_correction_shell else 'omitted'}."
        ),
    )


def demo() -> None:
    """Smoke demo: verify every shell is gauge-free and decodes without
    allow_gauge_detectors.  Requires stim and pymatching."""
    for label, builder in [
        ("LS-S", build_ls_s_shell),
        ("LS-CNOT", lambda **kw: build_ls_cnot_shell(**kw)),
        ("LS-T (no S corr)", lambda **kw: build_ls_t_injection_shell(include_s_correction_shell=False, **kw)),
        ("LS-T (with S corr)", lambda **kw: build_ls_t_injection_shell(include_s_correction_shell=True, **kw)),
    ]:
        clean = builder(p=0.0, noise=False)
        det, _ = clean.circuit.compile_detector_sampler().sample(2000, separate_observables=True)
        nd = int(det.any(axis=0).sum())
        noisy = builder(p=0.005, noise=True)
        try:
            noisy.circuit.detector_error_model(
                decompose_errors=True,
                allow_gauge_detectors=False,
            )
            graphlike_bound = len(noisy.circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True))
            dem_ok = "OK"
        except Exception as e:
            graphlike_bound = "-"
            dem_ok = f"NG({str(e)[:25]})"
        print(
            f"{label:18s}: detectors={clean.circuit.num_detectors:3d}, "
            f"obs={clean.circuit.num_observables}, gauge検出器={nd}, "
            f"DEM(allow_gauge_detectors=False)={dem_ok}, graphlike上界={graphlike_bound}"
        )


if __name__ == "__main__":
    demo()
