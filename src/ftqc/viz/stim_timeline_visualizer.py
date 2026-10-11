#!/usr/bin/env python3
"""
Stim physical-qubit timeline / timeslice visualizer.

This is a dependency-light visualization layer for FTQC experiments.  It reads a
Stim text circuit, splits it into time slices, and emits:

  * a JSON manifest with per-tick instruction/qubit activity
  * a self-contained HTML report with a physical-qubit timeline SVG
  * an optional standalone SVG
  * an optional normalized .stim file with explicit TICKs inserted

Tick policies:
  explicit     Use TICK instructions already present in the Stim file.
  auto         Ignore existing TICKs and greedily pack instructions into
               non-overlapping qubit timeslices.  Useful for circuits generated
               without TICKs, e.g. MPP-level toy circuits.
  instruction  Put each activity instruction into its own tick.

The parser intentionally does not import stim.  It handles common Stim text
syntax including QUBIT_COORDS, TICK, Clifford gates, M/MR, MPP, noise, DETECTOR,
OBSERVABLE_INCLUDE, and simple REPEAT blocks by unrolling them.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Stim-ish instruction categorization
# ---------------------------------------------------------------------------

SINGLE_Q_GATES = {
    "I", "X", "Y", "Z", "H", "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "C_XYZ", "C_ZYX", "H_XY", "H_XZ", "H_YZ", "H_NXY", "H_NXZ", "H_NYZ",
}
TWO_Q_GATES = {
    "CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SWAP", "ISWAP",
    "ISWAP_DAG", "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG",
    "SQRT_ZZ", "SQRT_ZZ_DAG",
}
MEASURE_GATES = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}
RESET_GATES = {"R", "RX", "RY", "RZ"}
META_GATES = {"QUBIT_COORDS", "SHIFT_COORDS"}
NON_ACTIVITY_GATES = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}

CATEGORY_COLORS = {
    "reset": "#a6cee3",
    "single_qubit": "#1f78b4",
    "two_qubit": "#6a3d9a",
    "mpp": "#b15928",
    "measure": "#33a02c",
    "noise": "#fb9a99",
    "detector": "#fdbf6f",
    "observable": "#ff7f00",
    "meta": "#bdbdbd",
    "other": "#cab2d6",
    "mixed": "#4d4d4d",
}

CATEGORY_SHORT = {
    "reset": "R",
    "single_qubit": "1Q",
    "two_qubit": "2Q",
    "mpp": "MPP",
    "measure": "M",
    "noise": "N",
    "detector": "D",
    "observable": "O",
    "meta": "meta",
    "other": "op",
    "mixed": "mix",
}

# Priority used when a qubit has multiple operations in the same tick, e.g.
# H and DEPOLARIZE1.  The label shows all ops; the color uses the highest-priority
# category to avoid noise-only colors hiding the primary operation.
CATEGORY_PRIORITY = {
    "measure": 9,
    "reset": 8,
    "mpp": 7,
    "two_qubit": 6,
    "single_qubit": 5,
    "noise": 2,
    "other": 1,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StimInstruction:
    order: int
    name: str
    args: List[float]
    target_text: str
    raw: str
    qubits: List[int] = field(default_factory=list)
    category: str = "other"
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    measurement_count: int = 0
    detector_count: int = 0
    observable_count: int = 0
    tick_boundary: bool = False

    def to_json_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # tuples become lists in json naturally, but make it explicit.
        d["pairs"] = [[a, b] for a, b in self.pairs]
        return d


@dataclass
class TickSlice:
    tick: int
    instructions: List[StimInstruction] = field(default_factory=list)
    explicit_tick_after: bool = False

    @property
    def qubits(self) -> List[int]:
        s = set()
        for inst in self.instructions:
            s.update(inst.qubits)
        return sorted(s)

    @property
    def category_counts(self) -> Dict[str, int]:
        c = Counter(inst.category for inst in self.instructions)
        return dict(sorted(c.items()))

    @property
    def instruction_counts(self) -> Dict[str, int]:
        c = Counter(inst.name for inst in self.instructions)
        return dict(sorted(c.items()))

    @property
    def measurement_count(self) -> int:
        return sum(inst.measurement_count for inst in self.instructions)

    @property
    def detector_count(self) -> int:
        return sum(inst.detector_count for inst in self.instructions)

    @property
    def observable_count(self) -> int:
        return sum(inst.observable_count for inst in self.instructions)

    @property
    def pairs(self) -> List[Tuple[int, int, str]]:
        out = []
        for inst in self.instructions:
            for a, b in inst.pairs:
                out.append((a, b, inst.name))
        return out

    def qubit_activity(self) -> Dict[int, List[StimInstruction]]:
        d: Dict[int, List[StimInstruction]] = defaultdict(list)
        for inst in self.instructions:
            for q in inst.qubits:
                d[q].append(inst)
        return d

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "tick": self.tick,
            "num_instructions": len(self.instructions),
            "instruction_counts": self.instruction_counts,
            "category_counts": self.category_counts,
            "qubits": self.qubits,
            "num_qubits_touched": len(self.qubits),
            "measurement_count": self.measurement_count,
            "detector_count": self.detector_count,
            "observable_count": self.observable_count,
            "pairs": [[a, b, name] for a, b, name in self.pairs],
            "instructions": [inst.to_json_dict() for inst in self.instructions],
        }


@dataclass
class TimelineManifest:
    source_file: str
    title: str
    tick_policy: str
    num_ticks: int
    num_qubits: int
    used_qubits: List[int]
    qubit_coords: Dict[int, Tuple[float, ...]]
    stride: Optional[int]
    ticks: List[TickSlice]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "source_file": self.source_file,
            "title": self.title,
            "tick_policy": self.tick_policy,
            "num_ticks": self.num_ticks,
            "num_qubits": self.num_qubits,
            "used_qubits": self.used_qubits,
            "qubit_coords": {str(k): list(v) for k, v in sorted(self.qubit_coords.items())},
            "stride": self.stride,
            "qubits": [qubit_descriptor(q, self.qubit_coords.get(q), self.stride) for q in self.used_qubits],
            "ticks": [t.to_json_dict() for t in self.ticks],
            "metadata": self.metadata,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def strip_comment(line: str) -> str:
    # Stim comments are line comments introduced by #.  There is no quoted string
    # syntax in regular circuit text, so a simple split is adequate.
    return line.split("#", 1)[0].strip()


def expand_repeat_blocks(lines: Sequence[str], max_unroll: int = 10000) -> List[str]:
    """Expand simple Stim REPEAT blocks into flat text lines.

    The implementation supports common forms:
      REPEAT 3 {
        ...
      }
    It is intentionally conservative: if a repeat count is too large, it raises
    instead of producing a gigantic visualization accidentally.
    """

    clean = [strip_comment(x) for x in lines]
    clean = [x for x in clean if x]

    def parse_block(i: int) -> Tuple[List[str], int]:
        out: List[str] = []
        while i < len(clean):
            line = clean[i]
            if line == "}":
                return out, i + 1
            m = re.match(r"^REPEAT\s+(\d+)\s*\{\s*$", line, flags=re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if n > max_unroll:
                    raise ValueError(f"REPEAT {n} exceeds max_unroll={max_unroll}")
                block, j = parse_block(i + 1)
                if len(block) * n > max_unroll * 100:
                    raise ValueError("expanded REPEAT block would be too large")
                for _ in range(n):
                    out.extend(block)
                i = j
                continue
            # Rare variant: REPEAT n on one line then { on next line.
            m2 = re.match(r"^REPEAT\s+(\d+)\s*$", line, flags=re.IGNORECASE)
            if m2 and i + 1 < len(clean) and clean[i + 1] == "{":
                n = int(m2.group(1))
                if n > max_unroll:
                    raise ValueError(f"REPEAT {n} exceeds max_unroll={max_unroll}")
                block, j = parse_block(i + 2)
                for _ in range(n):
                    out.extend(block)
                i = j
                continue
            out.append(line)
            i += 1
        return out, i

    out, end = parse_block(0)
    if end != len(clean):
        raise ValueError("unexpected parser state while expanding REPEAT blocks")
    return out


def parse_args(arg_text: Optional[str]) -> List[float]:
    if not arg_text:
        return []
    vals: List[float] = []
    for part in arg_text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            # Stim also permits symbolic-ish args in some contexts.  Keep NaN so
            # the JSON remains structurally consistent.
            vals.append(float("nan"))
    return vals


def parse_stim_line(line: str, order: int) -> StimInstruction:
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*(.*)$", line)
    if not m:
        raise ValueError(f"Could not parse Stim line: {line!r}")
    name = m.group(1).upper()
    args = parse_args(m.group(2))
    target_text = (m.group(3) or "").strip()
    inst = StimInstruction(order=order, name=name, args=args, target_text=target_text, raw=line)
    if name == "TICK":
        inst.category = "meta"
        inst.tick_boundary = True
        return inst
    if name in META_GATES:
        inst.category = "meta"
        inst.qubits = parse_qubit_targets(target_text)
        return inst
    inst.qubits = parse_qubit_targets(target_text)
    inst.category = categorize_instruction(name)
    inst.measurement_count = measurement_count(name, target_text, inst.qubits)
    inst.detector_count = 1 if name == "DETECTOR" else 0
    inst.observable_count = 1 if name == "OBSERVABLE_INCLUDE" else 0
    inst.pairs = parse_pairs(name, inst.qubits)
    return inst


def parse_qubit_targets(target_text: str) -> List[int]:
    """Extract qubit indices from Stim target text.

    Examples:
      "2 3 16 17" -> [2,3,16,17]
      "X0*Z1 Y2" -> [0,1,2]
      "rec[-4]" -> []
    """
    if not target_text:
        return []
    qubits: List[int] = []
    # Split on whitespace, then on MPP combiners.
    atoms: List[str] = []
    for tok in target_text.split():
        atoms.extend(tok.split("*"))
    for atom in atoms:
        atom = atom.strip().strip(",")
        if not atom:
            continue
        if atom.startswith("!"):
            atom = atom[1:]
        # Ignore non-qubit targets.
        if atom.startswith("rec[") or atom.startswith("D") or atom.startswith("L") or atom == "combiner":
            continue
        if re.fullmatch(r"-?\d+", atom):
            q = int(atom)
            if q >= 0:
                qubits.append(q)
            continue
        pm = re.fullmatch(r"[XYZ](-?\d+)", atom)
        if pm:
            q = int(pm.group(1))
            if q >= 0:
                qubits.append(q)
            continue
        # Some dumped Stim targets can look like sweep[0], ignore them.
    # Preserve order but drop duplicates inside the same instruction target list
    # for visualization row activity.  Pairs are built before dedup?  Here pairs
    # need ordering, so return ordered list with duplicates removed only if exact
    # repeated target appears.  This is okay for normal paired gates.
    return qubits


def categorize_instruction(name: str) -> str:
    if name in RESET_GATES:
        return "reset"
    if name in MEASURE_GATES:
        return "measure"
    if name in SINGLE_Q_GATES:
        return "single_qubit"
    if name in TWO_Q_GATES:
        return "two_qubit"
    if name == "MPP":
        return "mpp"
    if name == "DETECTOR":
        return "detector"
    if name == "OBSERVABLE_INCLUDE":
        return "observable"
    if name in META_GATES or name == "TICK":
        return "meta"
    if is_noise_instruction(name):
        return "noise"
    return "other"


def is_noise_instruction(name: str) -> bool:
    return (
        name.endswith("_ERROR")
        or name.startswith("DEPOLARIZE")
        or name.startswith("PAULI_CHANNEL")
        or name in {"CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"}
    )


def measurement_count(name: str, target_text: str, qubits: Sequence[int]) -> int:
    if name in MEASURE_GATES:
        return len(qubits)
    if name == "MPP":
        # Each whitespace-separated Pauli product is one measurement.
        return len([tok for tok in target_text.split() if tok])
    return 0


def parse_pairs(name: str, qubits: Sequence[int]) -> List[Tuple[int, int]]:
    if name in TWO_Q_GATES or name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"}:
        return [(int(qubits[i]), int(qubits[i + 1])) for i in range(0, len(qubits) - 1, 2)]
    return []


def parse_stim_file(path: str | Path, max_unroll: int = 10000) -> Tuple[List[StimInstruction], Dict[int, Tuple[float, ...]], List[str]]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    expanded = expand_repeat_blocks(lines, max_unroll=max_unroll)
    instructions: List[StimInstruction] = []
    coords: Dict[int, Tuple[float, ...]] = {}
    meta_lines: List[str] = []
    for order, line in enumerate(expanded):
        inst = parse_stim_line(line, order)
        instructions.append(inst)
        if inst.name == "QUBIT_COORDS" and inst.qubits:
            coords[inst.qubits[0]] = tuple(float(x) for x in inst.args)
            meta_lines.append(inst.raw)
        elif inst.name == "SHIFT_COORDS":
            meta_lines.append(inst.raw)
    return instructions, coords, meta_lines


# ---------------------------------------------------------------------------
# Tick construction
# ---------------------------------------------------------------------------


def is_activity_instruction(inst: StimInstruction) -> bool:
    return inst.name not in {"TICK", "QUBIT_COORDS", "SHIFT_COORDS"}


def build_ticks(instructions: Sequence[StimInstruction], tick_policy: str = "explicit") -> List[TickSlice]:
    if tick_policy not in {"explicit", "auto", "instruction"}:
        raise ValueError("tick_policy must be one of: explicit, auto, instruction")
    activity = [inst for inst in instructions if is_activity_instruction(inst)]

    if tick_policy == "instruction":
        return [TickSlice(tick=i, instructions=[inst]) for i, inst in enumerate(activity)]

    if tick_policy == "auto":
        ticks: List[TickSlice] = []
        cur = TickSlice(tick=0)
        used: set[int] = set()
        for inst in activity:
            qs = set(inst.qubits)
            # Detector/observable instructions have no qubits; attach them to the
            # current slice so measurement bookkeeping is visible next to nearby ops.
            # Noise instructions are also treated as decorations of the preceding
            # operation layer, matching the way Stim typically places
            # DEPOLARIZE*/X_ERROR immediately after the operation they model.
            decorative = inst.category in {"noise", "detector", "observable"}
            if (not decorative) and qs and used.intersection(qs) and cur.instructions:
                ticks.append(cur)
                cur = TickSlice(tick=len(ticks))
                used = set()
            cur.instructions.append(inst)
            if not decorative:
                used.update(qs)
        if cur.instructions or not ticks:
            ticks.append(cur)
        return ticks

    # explicit TICK policy
    ticks = [TickSlice(tick=0)]
    saw_tick = False
    for inst in instructions:
        if inst.name == "TICK":
            saw_tick = True
            ticks[-1].explicit_tick_after = True
            # Keep empty ticks if TICKs are consecutive?  Usually not useful for
            # visualization; add next slice and prune trailing empty below.
            ticks.append(TickSlice(tick=len(ticks)))
            continue
        if not is_activity_instruction(inst):
            continue
        ticks[-1].instructions.append(inst)
    # Remove a trailing empty slice after the last TICK.
    while len(ticks) > 1 and not ticks[-1].instructions:
        ticks.pop()
    if not saw_tick:
        # Explicit policy on a file without TICKs would put everything into one
        # huge tick.  Keep this behavior but mark it in metadata later.
        return ticks
    # Re-number in case trailing ticks were pruned.
    for i, t in enumerate(ticks):
        t.tick = i
    return ticks


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def qubit_descriptor(q: int, coord: Optional[Tuple[float, ...]], stride: Optional[int]) -> Dict[str, object]:
    d: Dict[str, object] = {"physical": q}
    if stride and stride > 0:
        d["logical"] = q // stride
        d["local"] = q % stride
    if coord is not None:
        d["coord"] = list(coord)
    return d


def build_manifest(
    source_file: str | Path,
    tick_policy: str = "explicit",
    stride: Optional[int] = None,
    title: Optional[str] = None,
    used_only: bool = True,
    max_unroll: int = 10000,
) -> Tuple[TimelineManifest, List[str]]:
    instructions, coords, meta_lines = parse_stim_file(source_file, max_unroll=max_unroll)
    ticks = build_ticks(instructions, tick_policy=tick_policy)
    used_qubits = sorted({q for t in ticks for inst in t.instructions for q in inst.qubits})
    if not used_only:
        max_q = max(max(used_qubits or [0]), max(coords.keys() or [0]))
        used_qubits = list(range(max_q + 1))
    num_measurements = sum(t.measurement_count for t in ticks)
    num_detectors = sum(t.detector_count for t in ticks)
    num_observables = sum(t.observable_count for t in ticks)
    num_ticks = len(ticks)
    explicit_tick_count = sum(1 for inst in instructions if inst.name == "TICK")
    active_logicals_per_tick = {}
    patch_concurrency_counts = {}
    if stride and stride > 0:
        c = Counter()
        for t in ticks:
            logicals = sorted({q // stride for q in t.qubits})
            active_logicals_per_tick[str(t.tick)] = logicals
            c[len(logicals)] += 1
        patch_concurrency_counts = dict(sorted(c.items()))
    manifest = TimelineManifest(
        source_file=str(source_file),
        title=title or f"Stim timeline: {Path(source_file).name}",
        tick_policy=tick_policy,
        num_ticks=num_ticks,
        num_qubits=len(used_qubits),
        used_qubits=used_qubits,
        qubit_coords=coords,
        stride=stride,
        ticks=ticks,
        metadata={
            "num_instructions": sum(len(t.instructions) for t in ticks),
            "num_measurements": num_measurements,
            "num_detectors": num_detectors,
            "num_observables": num_observables,
            "explicit_tick_count_in_source": explicit_tick_count,
            "used_only": used_only,
            "active_logicals_per_tick": active_logicals_per_tick,
            "patch_concurrency_counts": patch_concurrency_counts,
            "note": "Each tick is a timeslice; explicit uses Stim TICK boundaries, auto greedily packs non-overlapping qubit instructions.",
        },
    )
    return manifest, meta_lines


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def choose_activity(insts: Sequence[StimInstruction]) -> Tuple[str, str, str]:
    """Return label, category, title for cell activity."""
    if not insts:
        return "", "", ""
    names = [inst.name for inst in insts]
    # Deduplicate while preserving order.
    label_names: List[str] = []
    for n in names:
        if n not in label_names:
            label_names.append(n)
    label = "+".join(short_gate_name(n) for n in label_names[:3])
    if len(label_names) > 3:
        label += "+..."
    best = max(insts, key=lambda inst: CATEGORY_PRIORITY.get(inst.category, 0))
    title = "; ".join(inst.raw for inst in insts)
    return label, best.category, title


def short_gate_name(name: str) -> str:
    mapping = {
        "DEPOLARIZE1": "D1",
        "DEPOLARIZE2": "D2",
        "X_ERROR": "XE",
        "Z_ERROR": "ZE",
        "Y_ERROR": "YE",
        "PAULI_CHANNEL_1": "P1",
        "PAULI_CHANNEL_2": "P2",
        "OBSERVABLE_INCLUDE": "OBS",
        "DETECTOR": "DET",
    }
    return mapping.get(name, name)


def color_for_category(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])


def escape(s: object) -> str:
    return html.escape(str(s), quote=True)


def render_timeline_svg(
    manifest: TimelineManifest,
    max_label_chars: int = 18,
    cell_w: int = 26,
    row_h: int = 18,
    left_w: int = 128,
    top_h: int = 68,
    right_pad: int = 20,
    bottom_h: int = 60,
) -> str:
    qubits = manifest.used_qubits
    ticks = manifest.ticks
    width = left_w + max(1, len(ticks)) * cell_w + right_pad
    height = top_h + max(1, len(qubits)) * row_h + bottom_h

    # Precompute qubit activity per tick.
    per_tick_activity = [t.qubit_activity() for t in ticks]
    y_of = {q: top_h + i * row_h + row_h / 2 for i, q in enumerate(qubits)}

    out: List[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">')
    out.append("<style>")
    out.append("text{font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:10px;fill:#222}")
    out.append(".axis{stroke:#999;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.pair{stroke:#7f7f7f;stroke-width:1.2;opacity:.55}.cell{stroke:#fff;stroke-width:.8;rx:2;ry:2}.small{font-size:8px}.ticklabel{font-size:9px}.qlabel{font-size:9px}.title{font-size:13px;font-weight:700}")
    out.append("</style>")
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    out.append(f'<text x="10" y="20" class="title">{escape(manifest.title)}</text>')
    out.append(f'<text x="10" y="38" class="small">policy={escape(manifest.tick_policy)} ticks={manifest.num_ticks} qubits={manifest.num_qubits} measurements={manifest.metadata.get("num_measurements",0)} detectors={manifest.metadata.get("num_detectors",0)}</text>')

    # Tick labels and vertical grid.
    for ti, t in enumerate(ticks):
        x = left_w + ti * cell_w
        if ti % 5 == 0 or len(ticks) <= 40:
            out.append(f'<text x="{x + cell_w / 2:.1f}" y="{top_h - 18}" text-anchor="middle" class="ticklabel">{ti}</text>')
        out.append(f'<line x1="{x}" y1="{top_h - 10}" x2="{x}" y2="{top_h + len(qubits) * row_h}" class="grid"/>')
    out.append(f'<line x1="{left_w + len(ticks)*cell_w}" y1="{top_h - 10}" x2="{left_w + len(ticks)*cell_w}" y2="{top_h + len(qubits)*row_h}" class="grid"/>')

    # Qubit rows.
    for ri, q in enumerate(qubits):
        y = top_h + ri * row_h
        if ri % 2 == 0:
            out.append(f'<rect x="0" y="{y}" width="{width}" height="{row_h}" fill="#f9fafb"/>')
        label = qubit_label(q, manifest.stride)
        coord = manifest.qubit_coords.get(q)
        if coord is not None:
            label += f" @{','.join(format_coord(c) for c in coord)}"
        if len(label) > max_label_chars:
            label = label[: max_label_chars - 1] + "…"
        out.append(f'<text x="{left_w - 6}" y="{y + row_h * 0.68:.1f}" text-anchor="end" class="qlabel"><title>{escape(qubit_label(q, manifest.stride))}</title>{escape(label)}</text>')
        out.append(f'<line x1="{left_w}" y1="{y}" x2="{left_w + len(ticks)*cell_w}" y2="{y}" class="grid"/>')

    # Draw 2q pair connectors first, behind cells.
    for ti, t in enumerate(ticks):
        x = left_w + ti * cell_w + cell_w / 2
        for a, b, name in t.pairs:
            if a in y_of and b in y_of:
                y1, y2 = y_of[a], y_of[b]
                out.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" class="pair"><title>{escape(name)} {a} {b} at tick {ti}</title></line>')

    # Activity cells.
    for ti, t in enumerate(ticks):
        x = left_w + ti * cell_w + 2
        act = per_tick_activity[ti]
        for ri, q in enumerate(qubits):
            insts = act.get(q, [])
            if not insts:
                continue
            label, cat, title = choose_activity(insts)
            y = top_h + ri * row_h + 2
            color = color_for_category(cat)
            out.append(f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{row_h-4}" class="cell" fill="{color}"><title>tick {ti}, q{q}: {escape(title)}</title></rect>')
            if cell_w >= 24:
                txt = escape(label[:4])
                out.append(f'<text x="{x + (cell_w-4)/2:.1f}" y="{y + row_h*0.58:.1f}" text-anchor="middle" class="small" fill="#111">{txt}</text>')

    # Measurement / detector count mini-bars below the matrix.
    base_y = top_h + len(qubits) * row_h + 22
    max_m = max([t.measurement_count for t in ticks] + [1])
    max_d = max([t.detector_count for t in ticks] + [1])
    out.append(f'<text x="{left_w - 6}" y="{base_y}" text-anchor="end" class="small">meas</text>')
    out.append(f'<text x="{left_w - 6}" y="{base_y + 16}" text-anchor="end" class="small">det</text>')
    for ti, t in enumerate(ticks):
        x = left_w + ti * cell_w + 3
        mh = 10 * (t.measurement_count / max_m) if max_m else 0
        dh = 10 * (t.detector_count / max_d) if max_d else 0
        if t.measurement_count:
            out.append(f'<rect x="{x}" y="{base_y - mh:.1f}" width="{cell_w-6}" height="{mh:.1f}" fill="{CATEGORY_COLORS["measure"]}"><title>tick {ti}: measurements={t.measurement_count}</title></rect>')
        if t.detector_count:
            out.append(f'<rect x="{x}" y="{base_y + 16 - dh:.1f}" width="{cell_w-6}" height="{dh:.1f}" fill="{CATEGORY_COLORS["detector"]}"><title>tick {ti}: detectors={t.detector_count}</title></rect>')

    out.append(f'<line x1="{left_w}" y1="{top_h - 10}" x2="{left_w}" y2="{top_h + len(qubits)*row_h}" class="axis"/>')
    out.append(f'<line x1="{left_w}" y1="{top_h + len(qubits)*row_h}" x2="{left_w + len(ticks)*cell_w}" y2="{top_h + len(qubits)*row_h}" class="axis"/>')
    out.append("</svg>")
    return "\n".join(out)


def qubit_label(q: int, stride: Optional[int]) -> str:
    if stride and stride > 0:
        return f"q{q}  L{q // stride}:l{q % stride}"
    return f"q{q}"


def format_coord(x: float) -> str:
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:g}"


def render_legend() -> str:
    items = []
    for cat in ["reset", "single_qubit", "two_qubit", "mpp", "measure", "noise", "detector", "observable", "other"]:
        items.append(
            f'<span class="legend-item"><span class="swatch" style="background:{CATEGORY_COLORS[cat]}"></span>{escape(cat)}</span>'
        )
    return "".join(items)


def render_tick_table(manifest: TimelineManifest, max_rows: int = 500) -> str:
    rows = []
    for t in manifest.ticks[:max_rows]:
        ops = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in t.instruction_counts.items())
        cats = ", ".join(f"{k}:{v}" for k, v in t.category_counts.items())
        qs = compact_int_ranges(t.qubits)
        raw_lines = "\n".join(inst.raw for inst in t.instructions)
        rows.append(
            "<tr>"
            f"<td>{t.tick}</td>"
            f"<td>{escape(ops)}</td>"
            f"<td>{escape(cats)}</td>"
            f"<td>{escape(qs)}</td>"
            f"<td>{t.measurement_count}</td>"
            f"<td>{t.detector_count}</td>"
            f"<td><details><summary>{len(t.instructions)} inst</summary><pre>{escape(raw_lines)}</pre></details></td>"
            "</tr>"
        )
    note = ""
    if len(manifest.ticks) > max_rows:
        note = f"<p class='muted'>Table truncated to {max_rows} of {len(manifest.ticks)} ticks. JSON contains all ticks.</p>"
    return f"""
{note}
<table class="tick-table">
<thead><tr><th>tick</th><th>ops</th><th>categories</th><th>qubits</th><th>meas</th><th>det</th><th>raw</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
"""


def compact_int_ranges(vals: Sequence[int]) -> str:
    if not vals:
        return ""
    vals = sorted(set(vals))
    ranges = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = v
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def render_patch_summary(manifest: TimelineManifest) -> str:
    if not manifest.stride:
        return ""
    groups: Dict[int, List[int]] = defaultdict(list)
    for q in manifest.used_qubits:
        groups[q // manifest.stride].append(q)
    cards = []
    for logical, qs in sorted(groups.items()):
        cards.append(
            f"<div class='patch-card'><h3>L{logical}</h3>"
            f"<p><b>{len(qs)}</b> used physical qubits</p>"
            f"<p class='mono'>{escape(compact_int_ranges(qs))}</p></div>"
        )
    return f"<section><h2>Logical patch grouping by stride={manifest.stride}</h2><div class='patch-grid'>{''.join(cards)}</div></section>"



def render_concurrency_note(manifest: TimelineManifest) -> str:
    counts = manifest.metadata.get("patch_concurrency_counts")
    if not counts:
        return ""
    return (
        "<p class='muted'><b>Logical-patch concurrency:</b> "
        f"active logical patches per tick = <span class='mono'>{escape(counts)}</span>. "
        "値がほぼ 1 なら、パッチは並列ではなく順番に連結されています。</p>"
    )

def render_html(manifest: TimelineManifest, svg: str) -> str:
    summary = manifest.metadata
    warning = ""
    if manifest.tick_policy == "explicit" and summary.get("explicit_tick_count_in_source", 0) == 0:
        warning = "<p class='warn'>このファイルには TICK が無かったため、explicit policy では全命令が1 tickに入っています。--tick-policy auto を使うと qubit 衝突を避けて自動 timeslice 化できます。</p>"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{escape(manifest.title)}</title>
<style>
:root {{ --border:#d0d7de; --muted:#57606a; --bg:#f6f8fa; }}
body {{ margin:0; padding:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#24292f; background:#fff; }}
h1 {{ margin:0 0 8px; font-size:24px; }}
h2 {{ margin-top:28px; font-size:18px; }}
h3 {{ margin:.2rem 0; font-size:15px; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:16px 0; }}
.card, .patch-card {{ border:1px solid var(--border); border-radius:10px; padding:12px; background:var(--bg); }}
.metric {{ font-size:24px; font-weight:700; }}
.muted {{ color:var(--muted); }}
.warn {{ border-left:4px solid #d1242f; background:#fff8f8; padding:10px; }}
.legend {{ margin:14px 0; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
.legend-item {{ display:inline-flex; align-items:center; gap:5px; font-size:13px; }}
.swatch {{ width:14px; height:14px; display:inline-block; border-radius:3px; border:1px solid #9994; }}
.timeline-scroll {{ overflow:auto; border:1px solid var(--border); border-radius:12px; padding:8px; background:#fff; }}
.tick-table {{ border-collapse:collapse; width:100%; font-size:12px; }}
.tick-table th, .tick-table td {{ border:1px solid var(--border); padding:6px 8px; vertical-align:top; }}
.tick-table th {{ background:var(--bg); position:sticky; top:0; }}
pre {{ white-space:pre-wrap; margin:8px 0 0; font-size:11px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
.patch-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
</style>
</head>
<body>
<h1>{escape(manifest.title)}</h1>
<p class="muted">Source: <span class="mono">{escape(manifest.source_file)}</span></p>
{warning}
<section class="summary">
  <div class="card"><div class="muted">tick policy</div><div class="metric">{escape(manifest.tick_policy)}</div></div>
  <div class="card"><div class="muted">ticks</div><div class="metric">{manifest.num_ticks}</div></div>
  <div class="card"><div class="muted">used physical qubits</div><div class="metric">{manifest.num_qubits}</div></div>
  <div class="card"><div class="muted">measurements</div><div class="metric">{summary.get('num_measurements', 0)}</div></div>
  <div class="card"><div class="muted">detectors</div><div class="metric">{summary.get('num_detectors', 0)}</div></div>
  <div class="card"><div class="muted">observables</div><div class="metric">{summary.get('num_observables', 0)}</div></div>
</section>
<section>
<h2>Physical qubit timeline</h2>
<p class="muted">横方向が tick / timeslice、縦方向が物理 qubit です。2量子ビットゲートは同 tick 内で縦線として接続しています。セルにマウスを置くと元の Stim 命令が見えます。</p>
{render_concurrency_note(manifest)}
<div class="legend">{render_legend()}</div>
<div class="timeline-scroll">{svg}</div>
</section>
{render_patch_summary(manifest)}
<section>
<h2>Tick table</h2>
{render_tick_table(manifest)}
</section>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Normalized ticked Stim output
# ---------------------------------------------------------------------------


def write_ticked_stim(manifest: TimelineManifest, meta_lines: Sequence[str], path: str | Path) -> None:
    lines: List[str] = []
    lines.extend(meta_lines)
    if meta_lines:
        lines.append("")
    for i, t in enumerate(manifest.ticks):
        lines.append(f"# tick {i}")
        for inst in t.instructions:
            lines.append(inst.raw)
        if i != len(manifest.ticks) - 1:
            lines.append("TICK")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def derive_out(path: str | Path, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + suffix))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visualize physical qubit activity in a Stim circuit timeline.")
    ap.add_argument("--stim-file", required=True, help="Input .stim file")
    ap.add_argument("--tick-policy", choices=["explicit", "auto", "instruction"], default="explicit", help="How to split timeslices")
    ap.add_argument("--stride", type=int, default=None, help="Optional logical patch stride for row labels, e.g. 40 for generated d=3 blocks or 9 for MPP patches")
    ap.add_argument("--title", default=None, help="Report title")
    ap.add_argument("--all-qubits", action="store_true", help="Show rows for all qubits 0..max instead of only used qubits")
    ap.add_argument("--max-unroll", type=int, default=10000, help="Maximum REPEAT count to unroll")
    ap.add_argument("--out-html", default=None, help="Output HTML path")
    ap.add_argument("--out-json", default=None, help="Output JSON manifest path")
    ap.add_argument("--out-svg", default=None, help="Output standalone SVG path")
    ap.add_argument("--out-ticked-stim", default=None, help="Optional normalized .stim with explicit TICK between generated timeslices")
    args = ap.parse_args(argv)

    manifest, meta_lines = build_manifest(
        args.stim_file,
        tick_policy=args.tick_policy,
        stride=args.stride,
        title=args.title,
        used_only=not args.all_qubits,
        max_unroll=args.max_unroll,
    )
    svg = render_timeline_svg(manifest)

    out_html = args.out_html or derive_out(args.stim_file, f"_timeline_{args.tick_policy}.html")
    out_json = args.out_json or derive_out(args.stim_file, f"_timeline_{args.tick_policy}.json")
    out_svg = args.out_svg or derive_out(args.stim_file, f"_timeline_{args.tick_policy}.svg")

    Path(out_html).write_text(render_html(manifest, svg), encoding="utf-8")
    manifest.write_json(out_json)
    Path(out_svg).write_text(svg, encoding="utf-8")
    if args.out_ticked_stim:
        write_ticked_stim(manifest, meta_lines, args.out_ticked_stim)

    print(f"wrote {out_html}")
    print(f"wrote {out_json}")
    print(f"wrote {out_svg}")
    if args.out_ticked_stim:
        print(f"wrote {args.out_ticked_stim}")
    print(
        f"summary: ticks={manifest.num_ticks}, used_qubits={manifest.num_qubits}, "
        f"measurements={manifest.metadata.get('num_measurements')}, "
        f"detectors={manifest.metadata.get('num_detectors')}, "
        f"observables={manifest.metadata.get('num_observables')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
