"""Physical assignment and report contracts for the Stage 03 LS visualizers."""
import json

import pytest

from ftqc.circuit_level_merge import CircuitLevelMerge
from ftqc.ls_cnot_ft import LSCnotFT, LSTeleportFT
from ftqc.ftqc_tsim_backend import _parse_gates
from ftqc.viz.ftqc_mapping_visualizer import build_ls_assignment, render_svg, write_bundle
from ftqc.viz.ftqc_run_report import build_run_report_html


def _builders():
    return [
        CircuitLevelMerge("ZZ"),
        CircuitLevelMerge("XX"),
        LSCnotFT(),
        LSTeleportFT(with_s=True),
        LSTeleportFT(with_s=False),
    ]


def _data_geometry(builder):
    if hasattr(builder, "coords"):
        coords = list(builder.coords)
        patch_at = lambda q: builder.patch_of.get(tuple(coords[q]), "?")
        coordinate_at = lambda q: tuple(coords[q])
        n_data = int(builder.N)
    else:
        coords = builder.dcoord
        patch_by_coord = {}
        for cell in builder.g.A_cells:
            patch_by_coord[(2 * cell[0] + 1, 2 * cell[1] + 1)] = "A"
        for cell in builder.g.B_cells:
            patch_by_coord[(2 * cell[0] + 1, 2 * cell[1] + 1)] = "B"
        patch_at = lambda q: patch_by_coord.get(tuple(coords[q]), "?")
        coordinate_at = lambda q: tuple(coords[q])
        n_data = int(builder.n)
    return n_data, patch_at, coordinate_at


@pytest.mark.parametrize("builder", _builders(), ids=lambda b: type(b).__name__ + str(getattr(b, "kind", "")))
def test_ls_assignment_matches_real_builder_geometry_and_seams(builder, tmp_path):
    manifest = build_ls_assignment(builder)
    n_data, patch_at, coordinate_at = _data_geometry(builder)
    data = [q for patch in manifest.patches for q in patch.data_qubits]
    data_by_id = {q.physical: q for q in data}

    assert manifest.model == "ls"
    assert manifest.metadata["n_data"] == n_data == len(data)
    assert len(data_by_id) == n_data
    assert all(q.physical < n_data and q.coord == coordinate_at(q.physical) for q in data)
    assert all(q.patch == patch_at(q.physical) for q in data)

    seam_count = 0
    for key, ancilla_id in builder.anc_of.items():
        support = [int(q) for q in key[3] if int(q) < n_data]
        expected_patches = {patch_at(q) for q in support}
        ancilla = next(q for q in manifest.ancilla_qubits if q.physical == ancilla_id)
        is_seam = len(expected_patches) > 1
        seam_count += int(is_seam)
        assert ("seam" in ancilla.roles) is is_seam
        assert ancilla.patch == ("seam" if is_seam else next(iter(expected_patches)))
        assert ancilla.kind == "ancilla"
        assert ancilla.physical >= n_data
        assert ancilla.coord == pytest.approx(
            (sum(coordinate_at(q)[0] for q in support) / len(support),
             sum(coordinate_at(q)[1] for q in support) / len(support))
        )

    assert manifest.metadata["n_ancilla"] == len(manifest.ancilla_qubits)
    assert manifest.metadata["n_seam_ancilla"] == seam_count
    assert manifest.metadata["n_seam_ancilla"] == sum("seam" in q.roles for q in manifest.ancilla_qubits)

    out_html, out_json, out_svg = write_bundle(
        manifest, tmp_path / "assignment.html", tmp_path / "assignment.json", tmp_path / "assignment.svg"
    )
    serialized = json.loads(out_json.read_text(encoding="utf-8"))
    assert serialized == json.loads(json.dumps(manifest.to_json_dict()))
    assert out_svg.read_text(encoding="utf-8") == render_svg(manifest)
    assert "Lattice-surgery layout" in out_svg.read_text(encoding="utf-8")
    assert "Machine-readable manifest" in out_html.read_text(encoding="utf-8")


def test_cnot_lattice_surgery_report_renders_real_mapping_and_timeline():
    html = build_run_report_html(
        [("H", 0, None), ("CNOT", 0, 1)],
        n_logical=2,
        cnot_mode="lattice_surgery",
        p_phys=0.0,
        shots=32,
    )
    assert "<!doctype html>" in html.lower()
    assert "LSCnotFT" in html
    assert "Lattice-surgery layout" in html
    assert "CX" in html


def test_tsim_gate_parser_returns_normalized_package_api():
    assert _parse_gates("H0 T0 CNOT0,1") == [
        ("H", 0, None), ("T", 0, None), ("CNOT", 0, 1)
    ]
