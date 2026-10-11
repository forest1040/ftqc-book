"""
FTQC logical-to-physical assignment visualizer.

This module makes the first visualization layer for the hybrid FTQC stack:
"Which physical qubits implement each logical qubit?"

It supports three assignment models present in this repository:

1. mpp_d3   (no stim/qulacs dependency)
   The custom d=3 MPP patch model used by clifford_ft_full.py (the transversal
   Clifford / generator-tracking path).  A logical qubit q owns 9 data qubits:
       physical index = q * stride + local_index, local_index=0..8
   By default stride=9.  Logical supports XL=[0,3,6], ZL=[0,1,2].

2. generated_d3   (no stim/qulacs dependency)
   A Stim-generated rotated_memory_z layout.  It parses QUBIT_COORDS from a
   .stim file; a logical qubit q owns the block q*stride <= p < (q+1)*stride
   (default stride=40, XL=[1,8,15], ZL=[1,3,5]).

3. ls   (introspects the lattice-surgery builders; imports stim indirectly)
   The multi-patch lattice-surgery layouts.  It reads the real geometry of a
   builder instance and renders data qubits, syndrome-extraction ancillas, and
   seam (merge) ancillas in one coordinate frame, coloured by physical patch:
     * circuit_level_merge.CircuitLevelMerge("ZZ"|"XX")  -> patches A, B
     * ls_cnot_ft.LSCnotFT()                              -> patches C, A, T
     * ls_cnot_ft.LSTeleportFT(with_s=True|False)         -> patches D, A1, M(, As)
   Data coords come from builder.dcoord / builder.coords; ancillas are placed at
   the centroid of their data support (builder.anc_of); seam ancillas are those
   whose support spans more than one patch.

Outputs:
  * assignment JSON manifest (data + ancilla + seam)
  * self-contained HTML report with SVG patch diagrams
  * optional standalone SVG

Example:
  python ftqc_mapping_visualizer.py --n-logical 4 --model mpp_d3 \
      --cnot 0:1 --out-html ftqc_assignment.html --out-json ftqc_assignment.json

  python ftqc_mapping_visualizer.py --model ls --ls-kind cnot \
      --src-path ./src --out-html ls_cnot_assignment.html

Programmatic use of the ls model:
  from ftqc.ls_cnot_ft import LSCnotFT
  from ftqc.viz.ftqc_mapping_visualizer import build_ls_assignment, write_bundle
  write_bundle(build_ls_assignment(LSCnotFT(), label="LS CNOT"), "ls_cnot.html")
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Shared d=3 MPP patch constants used by clifford_ft_full.py and
# logical_clifford_ftqc.py.
STABS: List[Tuple[str, List[int]]] = [
    ("X", [0, 1]),
    ("Z", [0, 1, 3, 4]),
    ("X", [1, 2, 4, 5]),
    ("Z", [2, 5]),
    ("Z", [3, 6]),
    ("X", [3, 4, 6, 7]),
    ("Z", [4, 5, 7, 8]),
    ("X", [7, 8]),
]
XL = [0, 3, 6]
ZL = [0, 1, 2]
NDATA = 9

# d=3 generated rotated_memory_z supports used by logical_to_physical.py.
GENERATED_XL_SUPPORT = [1, 8, 15]
GENERATED_ZL_SUPPORT = [1, 3, 5]
DEFAULT_GENERATED_STRIDE = 40


@dataclass
class PhysicalQubit:
    physical: int
    local: Optional[int]
    x: float
    y: float
    kind: str = "data"
    roles: List[str] = field(default_factory=list)
    coord: Optional[Tuple[float, ...]] = None
    patch: Optional[str] = None


@dataclass
class StabilizerAssignment:
    index: int
    pauli: str
    local_support: List[int]
    physical_support: List[int]


@dataclass
class LogicalPatchAssignment:
    logical: int
    base: int
    stride: int
    model: str
    data_qubits: List[PhysicalQubit]
    logical_x_support: List[int]
    logical_z_support: List[int]
    stabilizers: List[StabilizerAssignment] = field(default_factory=list)
    note: str = ""
    patch_label: Optional[str] = None


@dataclass
class CNOTAssignment:
    control_logical: int
    target_logical: int
    physical_pairs: List[Tuple[int, int]]
    note: str


@dataclass
class AssignmentManifest:
    model: str
    n_logical: int
    patches: List[LogicalPatchAssignment]
    cnot_assignments: List[CNOTAssignment] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    ancilla_qubits: List[PhysicalQubit] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Assignment builders
# ---------------------------------------------------------------------------


def _mpp_local_coord(local: int) -> Tuple[int, int]:
    """Return visual col,row for local data index 0..8."""
    return local % 3, local // 3


def build_mpp_assignment(
    n_logical: int,
    stride: int = NDATA,
    patches_per_row: Optional[int] = None,
    cnot_pairs: Sequence[Tuple[int, int]] = (),
) -> AssignmentManifest:
    """Build assignment manifest for the custom 9-data-qubit MPP patch model."""
    if n_logical < 1:
        raise ValueError("n_logical must be positive")
    if stride < NDATA:
        raise ValueError(f"stride must be >= {NDATA} for mpp_d3")
    patches_per_row = patches_per_row or max(1, min(n_logical, 4))

    patches: List[LogicalPatchAssignment] = []
    for q in range(n_logical):
        base = q * stride
        qs: List[PhysicalQubit] = []
        for local in range(NDATA):
            col, row = _mpp_local_coord(local)
            roles = []
            if local in XL:
                roles.append("XL")
            if local in ZL:
                roles.append("ZL")
            qs.append(
                PhysicalQubit(
                    physical=base + local,
                    local=local,
                    x=float(col),
                    y=float(row),
                    kind="data",
                    roles=roles,
                    coord=(float(col), float(row)),
                )
            )
        stabs = [
            StabilizerAssignment(
                index=i,
                pauli=pauli,
                local_support=list(support),
                physical_support=[base + s for s in support],
            )
            for i, (pauli, support) in enumerate(STABS)
        ]
        patches.append(
            LogicalPatchAssignment(
                logical=q,
                base=base,
                stride=stride,
                model="mpp_d3",
                data_qubits=qs,
                logical_x_support=[base + s for s in XL],
                logical_z_support=[base + s for s in ZL],
                stabilizers=stabs,
                note="q_i owns 9 data qubits: physical=q_i*stride+local. Local grid is a visualization convention.",
            )
        )

    cnot_assignments: List[CNOTAssignment] = []
    for c, t in cnot_pairs:
        if c < 0 or t < 0 or c >= n_logical or t >= n_logical:
            raise ValueError(f"CNOT pair out of range for n_logical={n_logical}: {c}->{t}")
        pairs = [(c * stride + i, t * stride + i) for i in range(NDATA)]
        cnot_assignments.append(
            CNOTAssignment(
                control_logical=c,
                target_logical=t,
                physical_pairs=pairs,
                note="Transversal CX pairs matching local data index i: CX(qc_base+i, qt_base+i).",
            )
        )

    return AssignmentManifest(
        model="mpp_d3",
        n_logical=n_logical,
        patches=patches,
        cnot_assignments=cnot_assignments,
        metadata={
            "NDATA": NDATA,
            "XL_local": XL,
            "ZL_local": ZL,
            "STABS": [{"pauli": p, "support": s} for p, s in STABS],
            "patches_per_row": patches_per_row,
            "description": "Custom d=3 rotated-surface-code-like MPP patch assignment used by the Clifford/QEC compiler.",
        },
    )


_QUBIT_COORD_RE = re.compile(r"^\s*QUBIT_COORDS\(([^)]*)\)\s+([0-9]+)\s*$")


def parse_qubit_coords_from_stim_text(text: str) -> Dict[int, Tuple[float, ...]]:
    coords: Dict[int, Tuple[float, ...]] = {}
    for line in text.splitlines():
        m = _QUBIT_COORD_RE.match(line)
        if not m:
            continue
        args = tuple(float(x.strip()) for x in m.group(1).split(",") if x.strip())
        q = int(m.group(2))
        coords[q] = args
    return coords


def build_generated_assignment_from_stim(
    stim_file: str | Path,
    n_logical: Optional[int] = None,
    stride: int = DEFAULT_GENERATED_STRIDE,
) -> AssignmentManifest:
    """Build assignment manifest from QUBIT_COORDS in a generated Stim circuit file."""
    text = Path(stim_file).read_text(encoding="utf-8")
    coords = parse_qubit_coords_from_stim_text(text)
    if not coords:
        raise ValueError(f"No QUBIT_COORDS found in {stim_file}")

    patch_ids = sorted({q // stride for q in coords})
    if n_logical is not None:
        patch_ids = [q for q in patch_ids if q < n_logical]
    else:
        n_logical = max(patch_ids) + 1

    patches: List[LogicalPatchAssignment] = []
    for logical in range(n_logical):
        base = logical * stride
        block_qubits = sorted(q for q in coords if base <= q < base + stride)
        pq: List[PhysicalQubit] = []
        for q in block_qubits:
            local = q - base
            roles: List[str] = []
            if local in GENERATED_XL_SUPPORT:
                roles.append("XL")
            if local in GENERATED_ZL_SUPPORT:
                roles.append("ZL")
            coord = coords[q]
            x = coord[0] if len(coord) >= 1 else 0.0
            y = coord[1] if len(coord) >= 2 else 0.0
            # Use a simple heuristic: generated d=3 data qubits are odd/odd coords.
            kind = "data" if len(coord) >= 2 and int(coord[0]) % 2 == 1 and int(coord[1]) % 2 == 1 else "measure"
            pq.append(
                PhysicalQubit(
                    physical=q,
                    local=local,
                    x=float(x),
                    y=float(y),
                    kind=kind,
                    roles=roles,
                    coord=tuple(float(v) for v in coord),
                )
            )
        patches.append(
            LogicalPatchAssignment(
                logical=logical,
                base=base,
                stride=stride,
                model="generated_d3",
                data_qubits=pq,
                logical_x_support=[base + s for s in GENERATED_XL_SUPPORT],
                logical_z_support=[base + s for s in GENERATED_ZL_SUPPORT],
                stabilizers=[],
                note="Parsed QUBIT_COORDS from Stim file; block assignment uses q//stride. kind=data is inferred from odd/odd coordinates.",
            )
        )

    return AssignmentManifest(
        model="generated_d3",
        n_logical=n_logical,
        patches=patches,
        cnot_assignments=[],
        metadata={
            "source_stim_file": str(stim_file),
            "stride": stride,
            "XL_local": GENERATED_XL_SUPPORT,
            "ZL_local": GENERATED_ZL_SUPPORT,
            "description": "Stim-generated rotated_memory_z physical-qubit layout parsed from QUBIT_COORDS.",
        },
    )


# ---------------------------------------------------------------------------
# Lattice-surgery assignment builder (introspects the LS builders)
# ---------------------------------------------------------------------------

# Per-patch fill colours used by the ls renderer (kept here so JSON consumers
# can reproduce the colouring).
LS_PATCH_COLORS: Dict[str, str] = {
    "A": "#bfdbfe", "B": "#fecaca", "C": "#bfdbfe", "T": "#fecaca",
    "D": "#bbf7d0", "A1": "#fde68a", "M": "#fed7aa", "As": "#e9d5ff",
    "seam": "#fee2e2", "?": "#e5e7eb",
}


def _ls_extract(builder):
    """Introspect an LS builder and return (data_records, anc_records, n_data).

    data_records: list of (index, (x,y), patch, roles)
    anc_records:  list of (anc_index, (x,y), type, patches_set, support_indices)
    Ancillas are placed at the centroid of their data support; an ancilla is a
    seam ancilla when its support spans more than one physical patch.
    """
    if hasattr(builder, "coords") and hasattr(builder, "patch_of"):
        # ls_cnot_ft.LSCnotFT / LSTeleportFT: coords is a list, patch_of maps coord->patch.
        coords = list(builder.coords)
        n_data = int(getattr(builder, "N", len(coords)))
        patch_of = builder.patch_of
        anc_of = builder.anc_of
        data_xy = lambda i: tuple(coords[i])
        data_patch = lambda i: patch_of.get(tuple(coords[i]), "?")
        role_sets = {"XL": set(), "ZL": set()}
        for attr in dir(builder):
            m = re.fullmatch(r"(XL|ZL)_\w+", attr)
            if not m:
                continue
            try:
                for s in getattr(builder, attr):
                    role_sets[m.group(1)].add(int(s))
            except TypeError:
                pass
    elif hasattr(builder, "dcoord") and hasattr(builder, "g"):
        # circuit_level_merge.CircuitLevelMerge: dcoord maps data index->coord.
        import numpy as _np
        g = builder.g
        dcoord = builder.dcoord
        n_data = int(getattr(builder, "n", len(dcoord)))
        anc_of = builder.anc_of
        coord2patch: Dict[Tuple[int, int], str] = {}
        for (c, r) in getattr(g, "A_cells", []):
            coord2patch[(2 * c + 1, 2 * r + 1)] = "A"
        for (c, r) in getattr(g, "B_cells", []):
            coord2patch[(2 * c + 1, 2 * r + 1)] = "B"
        data_xy = lambda i: tuple(dcoord[i])
        data_patch = lambda i: coord2patch.get(tuple(dcoord[i]), "?")
        role_sets = {"XL": set(), "ZL": set()}
        for vecattr in ("ZLA", "ZLB", "XLA", "XLB"):
            v = getattr(g, vecattr, None)
            if v is None:
                continue
            v = _np.asarray(v).ravel()
            half = len(v) // 2
            role = "ZL" if vecattr.startswith("ZL") else "XL"
            for j in range(half):
                if v[half + j]:
                    role_sets["ZL"].add(j)
                if v[j]:
                    role_sets["XL"].add(j)
    else:
        raise TypeError(
            "Unsupported builder for model=ls: expected CircuitLevelMerge / LSCnotFT / LSTeleportFT"
        )

    coord_lookup: Dict[int, Tuple[float, float]] = {}
    data_records = []
    for i in range(n_data):
        xy = data_xy(i)
        coord_lookup[i] = xy
        roles = []
        if i in role_sets["XL"]:
            roles.append("XL")
        if i in role_sets["ZL"]:
            roles.append("ZL")
        data_records.append((i, xy, data_patch(i), roles))

    patch_of_index = {i: p for (i, _xy, p, _r) in data_records}
    anc_records = []
    for key, anc in anc_of.items():
        # key = (cx, cy, type, support_indices)
        typ = key[2]
        support = [int(s) for s in key[3] if int(s) < n_data]
        if not support:
            continue
        xs = [coord_lookup[s][0] for s in support]
        ys = [coord_lookup[s][1] for s in support]
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        patches = {patch_of_index.get(s, "?") for s in support}
        anc_records.append((int(anc), centroid, typ, patches, support))
    return data_records, anc_records, n_data


def build_ls_assignment(builder, label: Optional[str] = None) -> AssignmentManifest:
    """Build a lattice-surgery assignment manifest from an LS builder instance.

    Accepts circuit_level_merge.CircuitLevelMerge, ls_cnot_ft.LSCnotFT, or
    ls_cnot_ft.LSTeleportFT.  Data qubits keep their real lattice coordinates;
    syndrome-extraction ancillas are placed at the centroid of their data
    support, and seam (cross-patch) ancillas are flagged.
    """
    data_records, anc_records, n_data = _ls_extract(builder)
    kind = type(builder).__name__
    label = label or kind

    patch_order: List[str] = []
    by_patch: Dict[str, List[PhysicalQubit]] = {}
    for (i, xy, patch, roles) in data_records:
        if patch not in by_patch:
            by_patch[patch] = []
            patch_order.append(patch)
        by_patch[patch].append(
            PhysicalQubit(
                physical=i, local=None, x=float(xy[0]), y=float(xy[1]),
                kind="data", roles=list(roles),
                coord=(float(xy[0]), float(xy[1])), patch=patch,
            )
        )

    patches: List[LogicalPatchAssignment] = []
    for li, p in enumerate(patch_order):
        qs = by_patch[p]
        patches.append(
            LogicalPatchAssignment(
                logical=li, base=min(q.physical for q in qs), stride=0, model="ls",
                data_qubits=qs,
                logical_x_support=[q.physical for q in qs if "XL" in q.roles],
                logical_z_support=[q.physical for q in qs if "ZL" in q.roles],
                stabilizers=[],
                note=f"physical patch {p} ({len(qs)} data qubits)",
                patch_label=p,
            )
        )

    ancillas: List[PhysicalQubit] = []
    n_seam = 0
    for (anc, (cx, cy), typ, pset, _support) in anc_records:
        seam = len(pset) > 1
        n_seam += int(seam)
        roles = [typ] + (["seam"] if seam else [])
        ancillas.append(
            PhysicalQubit(
                physical=anc, local=None, x=float(cx), y=float(cy),
                kind="ancilla", roles=roles, coord=(float(cx), float(cy)),
                patch=("seam" if seam else next(iter(pset))),
            )
        )

    return AssignmentManifest(
        model="ls",
        n_logical=len(patches),
        patches=patches,
        cnot_assignments=[],
        ancilla_qubits=ancillas,
        metadata={
            "builder": kind,
            "label": label,
            "n_data": n_data,
            "n_ancilla": len(ancillas),
            "n_seam_ancilla": n_seam,
            "patches": patch_order,
            "description": (
                f"Lattice-surgery physical layout introspected from {kind}: "
                f"{len(patch_order)} patches, {n_data} data, {len(ancillas)} ancilla "
                f"({n_seam} seam)."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _role_class(roles: Sequence[str]) -> str:
    if "XL" in roles and "ZL" in roles:
        return "both"
    if "XL" in roles:
        return "xl"
    if "ZL" in roles:
        return "zl"
    return "plain"


def _svg_patch_mpp(patch: LogicalPatchAssignment, ox: float, oy: float, scale: float = 58.0) -> str:
    pad = 18
    w = 2 * scale + 2 * pad
    h = 2 * scale + 2 * pad
    parts = [f'<g class="patch" transform="translate({ox},{oy})">']
    parts.append(f'<rect class="patch-box" x="0" y="0" width="{w}" height="{h}" rx="12"/>')
    parts.append(f'<text class="patch-title" x="{w/2}" y="-8" text-anchor="middle">logical q{patch.logical} → physical {patch.base}..{patch.base+NDATA-1}</text>')
    # Grid lines
    for k in range(3):
        x = pad + k * scale
        y = pad + k * scale
        parts.append(f'<line class="grid" x1="{pad}" y1="{y}" x2="{pad+2*scale}" y2="{y}"/>')
        parts.append(f'<line class="grid" x1="{x}" y1="{pad}" x2="{x}" y2="{pad+2*scale}"/>')
    # Stabilizer plaquettes as translucent support hulls (rough cell backgrounds).
    for stab in patch.stabilizers:
        coords = [_mpp_local_coord(i) for i in stab.local_support]
        if not coords:
            continue
        xs = [pad + c * scale for c, _ in coords]
        ys = [pad + r * scale for _, r in coords]
        x0, x1 = min(xs) - 15, max(xs) + 15
        y0, y1 = min(ys) - 15, max(ys) + 15
        klass = "stab-x" if stab.pauli == "X" else "stab-z"
        parts.append(f'<rect class="{klass}" x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="6"/>')
    # Qubits
    for qubit in patch.data_qubits:
        x = pad + qubit.x * scale
        y = pad + qubit.y * scale
        cls = _role_class(qubit.roles)
        label = str(qubit.physical)
        local = qubit.local if qubit.local is not None else "?"
        tooltip = f"logical q{patch.logical}; local {local}; physical {qubit.physical}; roles={','.join(qubit.roles) or '-'}"
        parts.append(f'<g><title>{html.escape(tooltip)}</title><circle class="q {cls}" cx="{x}" cy="{y}" r="16"/>')
        parts.append(f'<text class="q-label" x="{x}" y="{y+4}" text-anchor="middle">{label}</text>')
        parts.append(f'<text class="local-label" x="{x}" y="{y+28}" text-anchor="middle">l{local}</text></g>')
    parts.append("</g>")
    return "\n".join(parts)


def _svg_patch_generated(patch: LogicalPatchAssignment, ox: float, oy: float, scale: float = 30.0) -> str:
    if not patch.data_qubits:
        return ""
    minx = min(q.x for q in patch.data_qubits)
    maxx = max(q.x for q in patch.data_qubits)
    miny = min(q.y for q in patch.data_qubits)
    maxy = max(q.y for q in patch.data_qubits)
    pad = 32
    w = (maxx - minx) * scale + 2 * pad
    h = (maxy - miny) * scale + 2 * pad
    parts = [f'<g class="patch" transform="translate({ox},{oy})">']
    parts.append(f'<rect class="patch-box" x="0" y="0" width="{w}" height="{h}" rx="12"/>')
    parts.append(f'<text class="patch-title" x="{w/2}" y="-8" text-anchor="middle">logical q{patch.logical} block base={patch.base}</text>')
    for qubit in patch.data_qubits:
        x = pad + (qubit.x - minx) * scale
        y = pad + (qubit.y - miny) * scale
        role_cls = _role_class(qubit.roles)
        kind_cls = "data-kind" if qubit.kind == "data" else "measure-kind"
        label = str(qubit.physical)
        tooltip = f"logical q{patch.logical}; local {qubit.local}; physical {qubit.physical}; coord={qubit.coord}; kind={qubit.kind}; roles={','.join(qubit.roles) or '-'}"
        parts.append(f'<g><title>{html.escape(tooltip)}</title><circle class="q {kind_cls} {role_cls}" cx="{x}" cy="{y}" r="12"/>')
        parts.append(f'<text class="q-label small" x="{x}" y="{y+4}" text-anchor="middle">{label}</text></g>')
    parts.append("</g>")
    return "\n".join(parts)


LS_SVG_CSS = """
.ls-data { stroke: #222; stroke-width: 1.5; }
.ls-anc  { stroke: #555; stroke-width: 1.2; }
.ls-seam { stroke: #b91c1c; stroke-width: 2.4; }
.ls-xl   { stroke: #1d4ed8; stroke-width: 3.2; }
.ls-zl   { stroke: #c2410c; stroke-width: 3.2; }
.ls-both { stroke: #7e22ce; stroke-width: 3.6; }
.ls-q-label { font-size: 10px; font-weight: 700; fill: #111; pointer-events: none; }
.ls-anc-label { font-size: 8px; fill: #374151; pointer-events: none; }
.ls-patch-label { font-size: 13px; font-weight: 700; }
"""


def _svg_layout_ls(manifest: AssignmentManifest, scale: float = 34.0) -> str:
    """Render all patches' data + ancillas in one coordinate frame."""
    all_q = [q for p in manifest.patches for q in p.data_qubits] + list(manifest.ancilla_qubits)
    if not all_q:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40"></svg>'
    minx = min(q.x for q in all_q); maxx = max(q.x for q in all_q)
    miny = min(q.y for q in all_q); maxy = max(q.y for q in all_q)
    pad = 48
    W = (maxx - minx) * scale + 2 * pad
    H = (maxy - miny) * scale + 2 * pad + 34
    sx = lambda x: pad + (x - minx) * scale
    sy = lambda y: pad + (maxy - y) * scale + 34  # flip y so small y is at the bottom
    label = html.escape(str(manifest.metadata.get("label", manifest.model)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" class="ftqc-svg">',
        '<style><![CDATA[' + SVG_CSS + LS_SVG_CSS + ']]></style>',
        f'<text class="main-title" x="{W/2:.0f}" y="24" text-anchor="middle">Lattice-surgery layout ({label})</text>',
    ]
    # patch hulls + labels
    for p in manifest.patches:
        if not p.data_qubits:
            continue
        xs = [sx(q.x) for q in p.data_qubits]; ys = [sy(q.y) for q in p.data_qubits]
        x0, x1 = min(xs) - 18, max(xs) + 18
        y0, y1 = min(ys) - 18, max(ys) + 18
        color = LS_PATCH_COLORS.get(p.patch_label or "?", "#e5e7eb")
        parts.append(f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{x1-x0:.0f}" height="{y1-y0:.0f}" rx="10" fill="{color}" fill-opacity="0.30" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<text class="ls-patch-label" x="{(x0+x1)/2:.0f}" y="{y0-6:.0f}" text-anchor="middle" fill="#111">patch {html.escape(str(p.patch_label))}</text>')
    # data qubits
    for p in manifest.patches:
        color = LS_PATCH_COLORS.get(p.patch_label or "?", "#ffffff")
        for q in p.data_qubits:
            x, y = sx(q.x), sy(q.y)
            cls = ("ls-both" if ("XL" in q.roles and "ZL" in q.roles)
                   else "ls-xl" if "XL" in q.roles
                   else "ls-zl" if "ZL" in q.roles else "ls-data")
            tip = f"patch {p.patch_label}; data {q.physical}; coord {q.coord}; roles={','.join(q.roles) or '-'}"
            parts.append(f'<g><title>{html.escape(tip)}</title><circle class="{cls}" cx="{x:.0f}" cy="{y:.0f}" r="13" fill="{color}"/>')
            parts.append(f'<text class="ls-q-label" x="{x:.0f}" y="{y+4:.0f}" text-anchor="middle">{q.physical}</text></g>')
    # ancillas as squares
    for q in manifest.ancilla_qubits:
        x, y = sx(q.x), sy(q.y)
        seam = "seam" in q.roles
        s = 9
        cls = "ls-seam" if seam else "ls-anc"
        fill = "#fee2e2" if seam else ("#dbeafe" if "X" in q.roles else "#ffedd5")
        typ = ",".join(r for r in q.roles if r in ("X", "Z")) or "?"
        tip = f"ancilla {q.physical}; type {typ}; {'SEAM ' if seam else ''}patch {q.patch}"
        parts.append(f'<g><title>{html.escape(tip)}</title><rect class="{cls}" x="{x-s:.0f}" y="{y-s:.0f}" width="{2*s}" height="{2*s}" rx="2" fill="{fill}"/>')
        parts.append(f'<text class="ls-anc-label" x="{x:.0f}" y="{y+3:.0f}" text-anchor="middle">{q.physical}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(manifest: AssignmentManifest, max_patch_cols: Optional[int] = None) -> str:
    """Render a standalone SVG of the assignment."""
    if manifest.model == "ls":
        return _svg_layout_ls(manifest)
    n = len(manifest.patches)
    max_patch_cols = max_patch_cols or int(manifest.metadata.get("patches_per_row", min(max(n, 1), 4)))
    max_patch_cols = max(1, max_patch_cols)
    if manifest.model == "mpp_d3":
        cell_w, cell_h = 190, 210
        renderer = _svg_patch_mpp
    else:
        cell_w, cell_h = 270, 270
        renderer = _svg_patch_generated
    rows = math.ceil(n / max_patch_cols)
    width = max_patch_cols * cell_w + 30
    height = rows * cell_h + 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="ftqc-svg">',
        '<style><![CDATA[' + SVG_CSS + ']]></style>',
        f'<text class="main-title" x="{width/2}" y="28" text-anchor="middle">Logical → Physical Assignment ({html.escape(manifest.model)})</text>',
        '<g transform="translate(18,70)">',
    ]
    for i, patch in enumerate(manifest.patches):
        col = i % max_patch_cols
        row = i // max_patch_cols
        parts.append(renderer(patch, col * cell_w, row * cell_h))
    parts.append("</g>")
    if manifest.cnot_assignments:
        y0 = rows * cell_h + 55
        parts.append(f'<text class="legend" x="20" y="{y0}">CNOT assignment: local-index matched transversal CX pairs are listed in the HTML/JSON manifest.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


SVG_CSS = """
.ftqc-svg { background: #fff; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.main-title { font-size: 18px; font-weight: 700; fill: #222; }
.patch-title { font-size: 12px; font-weight: 700; fill: #222; }
.patch-box { fill: #fafafa; stroke: #444; stroke-width: 1.5; }
.grid { stroke: #aaa; stroke-width: 1; stroke-dasharray: 3 3; }
.stab-x { fill: #dbeafe; stroke: #60a5fa; stroke-width: 0.5; opacity: 0.25; }
.stab-z { fill: #fed7aa; stroke: #fb923c; stroke-width: 0.5; opacity: 0.25; }
.q { stroke: #222; stroke-width: 1.5; fill: #fff; }
.q.plain { fill: #fff; }
.q.xl { fill: #bfdbfe; stroke: #1d4ed8; stroke-width: 3; }
.q.zl { fill: #fed7aa; stroke: #c2410c; stroke-width: 3; }
.q.both { fill: #e9d5ff; stroke: #7e22ce; stroke-width: 3.5; }
.q.data-kind { stroke-width: 2; }
.q.measure-kind { fill: #f2f2f2; stroke-dasharray: 3 2; }
.q-label { font-size: 11px; font-weight: 700; fill: #111; pointer-events: none; }
.q-label.small { font-size: 8px; }
.local-label { font-size: 9px; fill: #555; }
.legend { font-size: 12px; fill: #333; }
"""


HTML_CSS = """
:root { color-scheme: light; }
body { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #1f2937; background: #f6f7fb; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin-top: 28px; font-size: 20px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); margin: 14px 0; }
.meta { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }
.pill { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 999px; padding: 5px 10px; font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; font-size: 13px; }
.swatch { display: inline-block; width: 14px; height: 14px; border-radius: 50%; vertical-align: -2px; margin-right: 4px; border: 2px solid #222; }
.swatch.xl { background: #bfdbfe; border-color: #1d4ed8; }
.swatch.zl { background: #fed7aa; border-color: #c2410c; }
.swatch.both { background: #e9d5ff; border-color: #7e22ce; }
.swatch.meas { background: #f2f2f2; border-style: dashed; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #f9fafb; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; border-radius: 10px; padding: 14px; overflow: auto; max-height: 420px; }
.note { color: #4b5563; line-height: 1.55; }
a { color: #1d4ed8; }
"""


def _patch_table(manifest: AssignmentManifest) -> str:
    is_ls = manifest.model == "ls"
    rows = []
    for p in manifest.patches:
        name = f"patch {p.patch_label}" if p.patch_label else f"q{p.logical}"
        stride = "-" if is_ls else str(p.stride)
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{p.base}</td>"
            f"<td>{stride}</td>"
            f"<td>{', '.join(str(q.physical) for q in p.data_qubits)}</td>"
            f"<td>{', '.join(map(str, p.logical_x_support))}</td>"
            f"<td>{', '.join(map(str, p.logical_z_support))}</td>"
            "</tr>"
        )
    head = "patch" if is_ls else "logical"
    return ("""
<table>
<thead><tr><th>""" + head + """</th><th>base</th><th>stride</th><th>owned physical qubits</th><th>X_L support</th><th>Z_L support</th></tr></thead>
<tbody>
""" + "\n".join(rows) + "\n</tbody></table>")


def _ls_ancilla_section(manifest: AssignmentManifest) -> str:
    """Ancilla / seam summary for the lattice-surgery model."""
    if manifest.model != "ls" or not manifest.ancilla_qubits:
        return ""
    rows = []
    for q in manifest.ancilla_qubits:
        typ = ", ".join(r for r in q.roles if r in ("X", "Z")) or "-"
        seam = "yes" if "seam" in q.roles else ""
        rows.append(
            "<tr>"
            f"<td>{q.physical}</td>"
            f"<td>{html.escape(str(q.patch))}</td>"
            f"<td>{typ}</td>"
            f"<td>{seam}</td>"
            f"<td>({q.x:.1f}, {q.y:.1f})</td>"
            "</tr>"
        )
    meta = manifest.metadata
    table = ("""
<table>
<thead><tr><th>ancilla</th><th>patch</th><th>type</th><th>seam</th><th>centroid (x,y)</th></tr></thead>
<tbody>
""" + "\n".join(rows) + "\n</tbody></table>")
    return (
        "<section class='card'>"
        "<h2>Ancilla &amp; seam qubits</h2>"
        f"<p class='note'>data={meta.get('n_data')}, ancilla={meta.get('n_ancilla')}, "
        f"seam ancilla={meta.get('n_seam_ancilla')}. "
        "Ancillas are placed at the centroid of their data support; seam ancillas span more than one patch.</p>"
        f"{table}</section>"
    )


def _stabilizer_table(manifest: AssignmentManifest) -> str:
    if manifest.model == "ls":
        return ("<p class='note'>For the lattice-surgery model, stabilizers are realised by the "
                "ancilla qubits listed above (one ancilla per stabilizer). Use the circuit's "
                "DETECTOR/DEM and the timeline visualizer for the syndrome-extraction schedule.</p>")
    if manifest.model != "mpp_d3":
        return "<p class='note'>This model is parsed from a generated Stim circuit, so stabilizer supports are not reconstructed here. Use the circuit's DETECTOR/DEM for the decoding graph view.</p>"
    rows = []
    for p in manifest.patches:
        for s in p.stabilizers:
            rows.append(
                "<tr>"
                f"<td>q{p.logical}</td>"
                f"<td>{s.index}</td>"
                f"<td>{s.pauli}</td>"
                f"<td>{', '.join(map(str, s.local_support))}</td>"
                f"<td>{', '.join(map(str, s.physical_support))}</td>"
                "</tr>"
            )
    return """
<table>
<thead><tr><th>logical</th><th>stab index</th><th>type</th><th>local support</th><th>physical support</th></tr></thead>
<tbody>
""" + "\n".join(rows) + "\n</tbody></table>"


def _cnot_table(manifest: AssignmentManifest) -> str:
    if not manifest.cnot_assignments:
        return "<p class='note'>No CNOT pairs were requested for this assignment report.</p>"
    sections = []
    for ca in manifest.cnot_assignments:
        pair_text = ", ".join(f"{a}->{b}" for a, b in ca.physical_pairs)
        sections.append(
            f"<h3>CNOT q{ca.control_logical} → q{ca.target_logical}</h3>"
            f"<p class='note'>{html.escape(ca.note)}</p>"
            f"<pre>{html.escape(pair_text)}</pre>"
        )
    return "\n".join(sections)


def render_html(manifest: AssignmentManifest, title: str = "FTQC Logical-to-Physical Assignment") -> str:
    svg = render_svg(manifest)
    jd = json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2)
    model_desc = html.escape(str(manifest.metadata.get("description", manifest.model)))
    ls_section = _ls_ancilla_section(manifest)
    if manifest.model == "ls":
        chips = "".join(
            "<span><span class='swatch' style='background:%s;border-color:#444'></span>patch %s</span>"
            % (LS_PATCH_COLORS.get(p, "#e5e7eb"), html.escape(str(p)))
            for p in manifest.metadata.get("patches", [])
        )
        ls_legend = (
            chips
            + "<span><span class='swatch' style='background:#dbeafe;border-radius:3px;border-color:#555'></span>X ancilla</span>"
            + "<span><span class='swatch' style='background:#ffedd5;border-radius:3px;border-color:#555'></span>Z ancilla</span>"
            + "<span><span class='swatch' style='background:#fee2e2;border-radius:3px;border-color:#b91c1c'></span>seam ancilla</span>"
        )
    else:
        ls_legend = "<span><span class='swatch meas'></span>generated-circuit measurement/ancilla qubit</span>"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{HTML_CSS}</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="note">{model_desc}</p>
  <div class="meta">
    <span class="pill">model={html.escape(manifest.model)}</span>
    <span class="pill">logical qubits={manifest.n_logical}</span>
    <span class="pill">patches={len(manifest.patches)}</span>
  </div>
  <div class="card legend">
    <span><span class="swatch xl"></span>X_L support</span>
    <span><span class="swatch zl"></span>Z_L support</span>
    <span><span class="swatch both"></span>X_L ∩ Z_L</span>
    {ls_legend}
  </div>
  <section class="card">
    <h2>Patch diagram</h2>
    {svg}
  </section>
  <section class="card">
    <h2>Logical → physical table</h2>
    {_patch_table(manifest)}
  </section>
  {ls_section}
  <section class="card">
    <h2>Stabilizer supports</h2>
    {_stabilizer_table(manifest)}
  </section>
  <section class="card">
    <h2>CNOT physical assignment</h2>
    {_cnot_table(manifest)}
  </section>
  <section class="card">
    <h2>Machine-readable manifest</h2>
    <pre>{html.escape(jd)}</pre>
  </section>
</main>
</body>
</html>
"""


def write_bundle(
    manifest: AssignmentManifest,
    out_html: str | Path,
    out_json: Optional[str | Path] = None,
    out_svg: Optional[str | Path] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    out_html = Path(out_html)
    out_html.write_text(render_html(manifest), encoding="utf-8")
    json_path: Optional[Path] = None
    svg_path: Optional[Path] = None
    if out_json is not None:
        json_path = Path(out_json)
        manifest.write_json(json_path)
    if out_svg is not None:
        svg_path = Path(out_svg)
        svg_path.write_text(render_svg(manifest), encoding="utf-8")
    return out_html, json_path, svg_path


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------


def _parse_cnot(values: Sequence[str]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for v in values:
        if ":" in v:
            a, b = v.split(":", 1)
        elif "->" in v:
            a, b = v.split("->", 1)
        elif "," in v:
            a, b = v.split(",", 1)
        else:
            raise ValueError(f"CNOT must look like 0:1 or 0->1, got {v!r}")
        pairs.append((int(a), int(b)))
    return pairs


def _load_ls_builder(ls_kind: str, src_path: Optional[str]):
    """Import and construct the requested lattice-surgery builder."""
    if ls_kind in ("merge_zz", "merge_xx"):
        from ftqc.circuit_level_merge import CircuitLevelMerge
        return CircuitLevelMerge("ZZ" if ls_kind == "merge_zz" else "XX"), ls_kind
    if ls_kind == "cnot":
        from ftqc.ls_cnot_ft import LSCnotFT
        return LSCnotFT(), "LS CNOT"
    if ls_kind in ("t", "t_nos"):
        from ftqc.ls_cnot_ft import LSTeleportFT
        return LSTeleportFT(with_s=(ls_kind == "t")), ("LS-T (with S)" if ls_kind == "t" else "LS-T (no S)")
    raise SystemExit(f"unknown --ls-kind: {ls_kind}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize logical-to-physical qubit assignment for the FTQC stack.")
    p.add_argument("--model", choices=["mpp_d3", "generated_d3", "ls"], default="mpp_d3")
    p.add_argument("--n-logical", type=int, default=4)
    p.add_argument("--stride", type=int, default=None, help="Patch stride. Defaults to 9 for mpp_d3, 40 for generated_d3.")
    p.add_argument("--stim-file", type=str, default=None, help="Stim circuit file for model=generated_d3.")
    p.add_argument("--ls-kind", choices=["merge_zz", "merge_xx", "cnot", "t", "t_nos"], default="cnot",
                   help="Which lattice-surgery builder to introspect for model=ls.")
    p.add_argument("--src-path", type=str, default=None,
                   help="Path to the simulator src/ directory (added to sys.path for model=ls).")
    p.add_argument("--cnot", action="append", default=[], help="CNOT assignment to display, e.g. --cnot 0:1. Can repeat.")
    p.add_argument("--out-html", type=str, default="ftqc_assignment.html")
    p.add_argument("--out-json", type=str, default="ftqc_assignment.json")
    p.add_argument("--out-svg", type=str, default="ftqc_assignment.svg")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cnot_pairs = _parse_cnot(args.cnot)
    if args.model == "mpp_d3":
        stride = args.stride if args.stride is not None else NDATA
        manifest = build_mpp_assignment(args.n_logical, stride=stride, cnot_pairs=cnot_pairs)
    elif args.model == "ls":
        builder, label = _load_ls_builder(args.ls_kind, args.src_path)
        manifest = build_ls_assignment(builder, label=label)
    else:
        if not args.stim_file:
            raise SystemExit("--stim-file is required for --model generated_d3")
        stride = args.stride if args.stride is not None else DEFAULT_GENERATED_STRIDE
        manifest = build_generated_assignment_from_stim(args.stim_file, n_logical=args.n_logical, stride=stride)
    html_path, json_path, svg_path = write_bundle(manifest, args.out_html, args.out_json, args.out_svg)
    print(f"wrote HTML: {html_path}")
    if json_path:
        print(f"wrote JSON: {json_path}")
    if svg_path:
        print(f"wrote SVG : {svg_path}")


if __name__ == "__main__":
    main()
